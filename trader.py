#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""업비트 변동성 돌파 자동매매.

전략 (래리 윌리엄스 변동성 돌파를 업비트 일봉에 맞춘 형태)
----------------------------------------------------------------
- 대상: KRW-BTC (환경 변수 UPBIT_TICKER 로 변경 가능)
- 매수: 현재가가 당일 시가 + (전일 고가 - 전일 저가) * 0.5 를 돌파하면
        보유 원화의 50% 로 시장가 매수. 거래일(09:00~다음날 09:00)당 1회.
- 매도: 익절 +3%, 손절 -2%, 또는 다음 거래일 09:00(KST) 에 봇이 산 수량 전량 매도.
- 기본 실행은 모의매매다. 실주문은 ``python trader.py --live`` 로만 나간다.

설치
----
    pip install pyupbit

실행
----
    export UPBIT_ACCESS_KEY="발급받은 액세스 키"
    export UPBIT_SECRET_KEY="발급받은 시크릿 키"
    python trader.py            # 모의매매, 계속 실행
    python trader.py --once     # 한 번만 판단
    python trader.py --live     # 실주문
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import pyupbit


# ---------------------------------------------------------------------------
# 설정
# API 키는 소스에 적지 않는다. 셸 환경 변수로만 읽는다.
# ---------------------------------------------------------------------------

KST = ZoneInfo("Asia/Seoul")

# 업비트 Open API 키. 비어 있으면 실거래를 시작하지 않는다.
ACCESS_KEY = os.environ.get("UPBIT_ACCESS_KEY", "").strip()
SECRET_KEY = os.environ.get("UPBIT_SECRET_KEY", "").strip()

# 업비트 원화 마켓 수수료 0.05%. 모의매매 체결가 계산에만 사용한다.
# 시장가 매수 주문 금액에는 수수료가 포함되므로, 실주문 금액은 잔고를 넘기지 않게만 맞춘다.
FEE_RATE = Decimal("0.0005")

# 시세 API 초당 10회, 주문 API 초당 8회 한도를 넘지 않도록 요청 사이에 둔다.
REQUEST_INTERVAL_SEC = 0.2

# 일봉은 자주 변하지 않는다. 현재가만 매 루프 조회한다.
OHLCV_CACHE_SEC = 30.0

# 주문 실패 시 같은 조건을 매초 다시 치지 않도록 잠시 쉰다.
ORDER_RETRY_SEC = 10.0

VOLUME_DECIMALS = 8


@dataclass(frozen=True)
class Config:
    """전략 숫자와 실행 모드. 기본값은 요청 예시와 같다."""

    ticker: str = "KRW-BTC"
    k: Decimal = Decimal("0.5")
    take_profit: Decimal = Decimal("0.03")
    stop_loss: Decimal = Decimal("0.02")
    invest_ratio: Decimal = Decimal("0.5")
    min_order_krw: int = 5000
    no_buy_minutes: int = 5
    loop_seconds: float = 1.0
    paper_krw: int = 1_000_000
    live: bool = False
    once: bool = False
    state_path: Path = Path("state/paper_KRW-BTC.json")


@dataclass
class Position:
    """봇이 직접 연 포지션. 계좌에 원래 있던 코인과 구분한다."""

    trading_day: str
    entry_price: str
    volume: str


@dataclass
class BotState:
    ticker: str
    last_buy_day: Optional[str] = None
    paper_krw: Optional[int] = None
    paper_volume: str = "0"
    position: Optional[Position] = None


@dataclass(frozen=True)
class Snapshot:
    """한 번의 판단에 필요한 시세와 잔고."""

    now: datetime
    price: Decimal
    today_open: Optional[Decimal]
    prev_high: Optional[Decimal]
    prev_low: Optional[Decimal]
    candle_day: Optional[str]
    krw_balance: Decimal
    coin_volume: Decimal


@dataclass(frozen=True)
class Decision:
    action: str  # "hold" | "buy" | "sell"
    reason: str
    target_price: Optional[Decimal] = None
    order_krw: Optional[int] = None
    sell_volume: Optional[Decimal] = None
    pnl_rate: Optional[Decimal] = None


@dataclass(frozen=True)
class Fill:
    success: bool
    message: str
    entry_price: Optional[Decimal] = None
    volume: Optional[Decimal] = None


# ---------------------------------------------------------------------------
# 전략 계산. 네트워크 없이 시험할 수 있게 순수 함수로 둔다.
# ---------------------------------------------------------------------------

def as_kst(now: datetime) -> datetime:
    """시간대 정보가 없으면 한국시간으로 간주한다."""
    if now.tzinfo is None:
        return now.replace(tzinfo=KST)
    return now.astimezone(KST)


def trading_day_key(now: datetime) -> str:
    """업비트 일봉이 바뀌는 09:00 KST 를 하루의 경계로 쓴다.

    예) 2026-10-07 08:59 KST -> "2026-10-06" (전날 09:00에 시작한 장)
        2026-10-07 09:00 KST -> "2026-10-07"
    """
    local = as_kst(now)
    if local.hour < 9:
        local = local - timedelta(days=1)
    return local.strftime("%Y-%m-%d")


def minutes_until_session_close(now: datetime) -> float:
    """다음 09:00 KST 까지 남은 분. 장 마감 직전 신규 매수를 거를 때 쓴다."""
    local = as_kst(now)
    close = local.replace(hour=9, minute=0, second=0, microsecond=0)
    if local >= close:
        close = close + timedelta(days=1)
    return (close - local).total_seconds() / 60.0


def calc_target_price(
    today_open: Decimal,
    prev_high: Decimal,
    prev_low: Decimal,
    k: Decimal,
) -> Decimal:
    """변동성 돌파 목표가.

    목표가 = 당일 시가 + (전일 고가 - 전일 저가) * K
    예) 시가 100, 전일 고가 110, 전일 저가 90, K 0.5
        변동폭 20 * 0.5 = 10, 목표가 110
    """
    return today_open + (prev_high - prev_low) * k


def calc_order_krw(krw_balance: Decimal, ratio: Decimal, min_order_krw: int) -> Optional[int]:
    """보유 원화의 ratio 만큼을 원 단위로 내림한다. 최소 주문 금액 미만이면 None.

    잔고 전액을 주문하면 수수료/반올림 때문에 거절될 수 있어, 계산 금액이
    주문 가능 원화와 같으면 1원을 남긴다. 기본 비율 50% 에서는 해당되지 않는다.
    """
    if krw_balance <= 0 or ratio <= 0:
        return None
    spendable = int(krw_balance.to_integral_value(rounding=ROUND_DOWN))
    amount = int((krw_balance * ratio).to_integral_value(rounding=ROUND_DOWN))
    amount = min(amount, spendable)
    if amount >= spendable and spendable > min_order_krw:
        amount = spendable - 1
    if amount < min_order_krw:
        return None
    return amount


def floor_volume(volume: Decimal, places: int = VOLUME_DECIMALS) -> Decimal:
    """업비트에 보낼 수량을 소수 places 자리에서 내림한다."""
    quant = Decimal("1").scaleb(-places)
    return volume.quantize(quant, rounding=ROUND_DOWN)


def volume_to_str(volume: Decimal, places: int = VOLUME_DECIMALS) -> str:
    """과학적 표기(1e-8)를 피하고 고정 소수점 문자열로 만든다."""
    return format(floor_volume(volume, places), "f")


def price_change_rate(price: Decimal, entry: Decimal) -> Decimal:
    return (price - entry) / entry


def _sell_decision(
    position: Position,
    reason: str,
    target: Optional[Decimal],
    pnl: Optional[Decimal] = None,
) -> Decision:
    volume = floor_volume(Decimal(position.volume))
    return Decision(
        action="sell",
        reason=reason,
        target_price=target,
        sell_volume=volume if volume > 0 else None,
        pnl_rate=pnl,
    )


def evaluate(config: Config, snap: Snapshot, state: BotState) -> Decision:
    """지금 사야 하는지, 팔아야 하는지, 기다려야 하는지 정한다."""
    day = trading_day_key(snap.now)
    target: Optional[Decimal] = None
    if (
        snap.candle_day == day
        and snap.today_open is not None
        and snap.prev_high is not None
        and snap.prev_low is not None
        and snap.prev_high >= snap.prev_low
    ):
        target = calc_target_price(snap.today_open, snap.prev_high, snap.prev_low, config.k)

    position = state.position
    if position is not None and Decimal(position.volume) > 0:
        entry = Decimal(position.entry_price)
        pnl = price_change_rate(snap.price, entry) if entry > 0 else None
        # 다음 날 09:00 이 되면 손익과 관계없이 봇이 산 수량을 모두 판다.
        if position.trading_day != day:
            return _sell_decision(position, "다음 날 09:00 장 시작, 보유 수량 전량 매도", target, pnl)
        if pnl is not None and pnl >= config.take_profit:
            return _sell_decision(position, f"익절 조건 도달 ({_format_rate(pnl)})", target, pnl)
        if pnl is not None and pnl <= -config.stop_loss:
            return _sell_decision(position, f"손절 조건 도달 ({_format_rate(pnl)})", target, pnl)
        return Decision(
            action="hold",
            reason="보유 중, 익절/손절/장 마감 조건 미충족",
            target_price=target,
            pnl_rate=pnl,
        )

    if state.last_buy_day == day:
        return Decision("hold", "오늘은 이미 매수해서 다시 사지 않습니다", target)

    if target is None:
        return Decision("hold", "당일 일봉이 아직 없어 목표가를 계산하지 않습니다", None)

    # 08:55~09:00 에 사면 직후 장 시작 매도로 바로 청산된다.
    if minutes_until_session_close(snap.now) < config.no_buy_minutes:
        return Decision("hold", "장 마감 직전이라 신규 매수하지 않습니다", target)

    if snap.price < target:
        return Decision("hold", "현재가가 목표가보다 낮습니다", target)

    order_krw = calc_order_krw(snap.krw_balance, config.invest_ratio, config.min_order_krw)
    if order_krw is None:
        return Decision(
            "hold",
            f"주문 가능 원화가 최소 주문 금액 {config.min_order_krw:,}원보다 작습니다",
            target,
        )

    return Decision(
        action="buy",
        reason="현재가가 변동성 돌파 목표가를 넘어 매수합니다",
        target_price=target,
        order_krw=order_krw,
    )


# ---------------------------------------------------------------------------
# 출력 / 상태 파일
# ---------------------------------------------------------------------------

def log(message: str, now: Optional[datetime] = None) -> None:
    stamp = as_kst(now or datetime.now(KST)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _format_rate(rate: Decimal) -> str:
    return f"{(rate * Decimal('100')):+.2f}%"


def format_price(value: Decimal) -> str:
    if value == value.to_integral():
        return f"{int(value):,}"
    return f"{value:,.4f}"


def format_krw(value: Decimal) -> str:
    return f"{int(value.to_integral_value(rounding=ROUND_DOWN)):,} KRW"


def log_balances(snap: Snapshot, ticker: str, now: datetime) -> None:
    coin = ticker.split("-", 1)[1]
    log(
        f"잔고 조회 | 원화 {format_krw(snap.krw_balance)} | {coin} {volume_to_str(snap.coin_volume)}",
        now,
    )


def load_state(path: Path, ticker: str, paper_krw: int) -> BotState:
    if not path.exists():
        return BotState(ticker=ticker, paper_krw=paper_krw, paper_volume="0")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        position = None
        if raw.get("position"):
            position = Position(**raw["position"])
        state = BotState(
            ticker=raw.get("ticker", ticker),
            last_buy_day=raw.get("last_buy_day"),
            paper_krw=raw.get("paper_krw", paper_krw),
            paper_volume=str(raw.get("paper_volume", "0")),
            position=position,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log(f"상태 파일을 읽지 못해 새로 시작합니다: {exc}")
        return BotState(ticker=ticker, paper_krw=paper_krw, paper_volume="0")
    if state.ticker != ticker:
        log(f"상태 파일의 종목({state.ticker})이 현재 종목과 달라 포지션을 무시합니다.")
        return BotState(ticker=ticker, paper_krw=paper_krw, paper_volume="0")
    return state


def save_state(path: Path, state: BotState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": state.ticker,
        "last_buy_day": state.last_buy_day,
        "paper_krw": state.paper_krw,
        "paper_volume": state.paper_volume,
        "position": None if state.position is None else asdict(state.position),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# 시세 / 주문
# ---------------------------------------------------------------------------

class MarketDataError(RuntimeError):
    """시세 조회 실패. 루프는 이 오류로 종료하지 않는다."""


class MarketData:
    """일봉은 짧게 캐시하고, 호출 뒤에는 항상 쉬어 요청 한도를 지킨다."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep
        self._ohlcv_frame = None
        self._ohlcv_at = 0.0

    def daily_candles(self, ticker: str):
        elapsed = time.monotonic() - self._ohlcv_at
        if self._ohlcv_frame is not None and elapsed < OHLCV_CACHE_SEC:
            return self._ohlcv_frame
        frame = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        self._sleep(REQUEST_INTERVAL_SEC)
        if frame is None or len(frame) < 2:
            raise MarketDataError("일봉 조회에 실패했거나 캔들이 2개보다 적습니다.")
        self._ohlcv_frame = frame
        self._ohlcv_at = time.monotonic()
        return frame

    def current_price(self, ticker: str) -> Decimal:
        price = pyupbit.get_current_price(ticker)
        self._sleep(REQUEST_INTERVAL_SEC)
        if not isinstance(price, (int, float)):
            raise MarketDataError(f"현재가 조회에 실패했습니다: {price!r}")
        return Decimal(str(price))


def _candle_day(index_value) -> str:
    """pyupbit 일봉 인덱스는 KST 09:00 의 naive 시각이다. 날짜가 거래일 키와 같다."""
    if hasattr(index_value, "to_pydatetime"):
        index_value = index_value.to_pydatetime()
    return index_value.strftime("%Y-%m-%d")


def snapshot_from_candles(frame, price: Decimal, krw: Decimal, coin: Decimal, now: datetime) -> Snapshot:
    prev = frame.iloc[-2]
    today = frame.iloc[-1]
    return Snapshot(
        now=now,
        price=price,
        today_open=Decimal(str(today["open"])),
        prev_high=Decimal(str(prev["high"])),
        prev_low=Decimal(str(prev["low"])),
        candle_day=_candle_day(frame.index[-1]),
        krw_balance=krw,
        coin_volume=coin,
    )


def is_order_success(result: object) -> bool:
    """pyupbit 는 성공 시 uuid 가 있는 dict, 실패 시 None 또는 error dict 를 준다."""
    return isinstance(result, dict) and bool(result.get("uuid")) and "error" not in result


def describe_order(result: object) -> str:
    if result is None:
        return "응답 없음 (pyupbit 가 예외를 삼킨 경우 직전 줄에 예외 클래스명이 찍힙니다)"
    if not isinstance(result, dict):
        return str(result)
    error = result.get("error")
    if isinstance(error, dict):
        return f"{error.get('name', 'error')}: {error.get('message', error)}"
    parts = [
        f"uuid={result.get('uuid')}",
        f"state={result.get('state')}",
        f"executed_volume={result.get('executed_volume')}",
        f"executed_funds={result.get('executed_funds')}",
        f"paid_fee={result.get('paid_fee')}",
    ]
    return ", ".join(parts)


class PaperExchange:
    """공개 시세만 조회하고 주문은 가상 잔고에 반영한다."""

    def __init__(self, state: BotState, market: MarketData, initial_krw: int) -> None:
        self.state = state
        self.market = market
        if state.paper_krw is None:
            state.paper_krw = initial_krw

    def fetch(self, ticker: str, now: datetime) -> Snapshot:
        frame = self.market.daily_candles(ticker)
        price = self.market.current_price(ticker)
        return snapshot_from_candles(
            frame,
            price,
            Decimal(state_krw(self.state)),
            Decimal(self.state.paper_volume),
            now,
        )

    def buy(self, ticker: str, order_krw: int, price: Decimal) -> Fill:
        del ticker  # 모의 체결은 종목명 없이 가격만 사용한다.
        krw = state_krw(self.state)
        if order_krw > krw:
            return Fill(False, f"모의 매수 실패 | 주문 금액 {order_krw:,}원이 잔고 {krw:,}원보다 큽니다")
        net = Decimal(order_krw) * (Decimal("1") - FEE_RATE)
        volume = floor_volume(net / price)
        if volume <= 0:
            return Fill(False, "모의 매수 실패 | 주문 금액으로 살 수 있는 수량이 최소 단위보다 작습니다")
        self.state.paper_krw = krw - order_krw
        self.state.paper_volume = volume_to_str(volume)
        return Fill(
            True,
            f"모의 매수 성공 | 주문 {order_krw:,} KRW | 체결가 {format_price(price)} | 수량 {volume_to_str(volume)}",
            entry_price=price,
            volume=volume,
        )

    def sell(self, ticker: str, volume: Decimal, price: Decimal) -> Fill:
        del ticker
        held = Decimal(self.state.paper_volume)
        sell_volume = floor_volume(min(volume, held))
        if sell_volume <= 0:
            return Fill(False, "모의 매도 실패 | 매도할 수량이 없습니다")
        gross = sell_volume * price
        proceeds = int((gross * (Decimal("1") - FEE_RATE)).to_integral_value(rounding=ROUND_DOWN))
        self.state.paper_krw = state_krw(self.state) + proceeds
        self.state.paper_volume = "0"
        return Fill(
            True,
            f"모의 매도 성공 | 수량 {volume_to_str(sell_volume)} | 체결가 {format_price(price)} | 정산 {proceeds:,} KRW",
            entry_price=price,
            volume=sell_volume,
        )


def state_krw(state: BotState) -> int:
    return 0 if state.paper_krw is None else int(state.paper_krw)


class LiveExchange:
    """pyupbit.Upbit 로 잔고를 조회하고 시장가 주문을 낸다."""

    def __init__(self, upbit: "pyupbit.Upbit", market: MarketData, sleep: Callable[[float], None] = time.sleep) -> None:
        self.upbit = upbit
        self.market = market
        self._sleep = sleep

    def fetch(self, ticker: str, now: datetime) -> Snapshot:
        frame = self.market.daily_candles(ticker)
        price = self.market.current_price(ticker)
        krw = self._balance("KRW")
        coin = self._balance(ticker)
        return snapshot_from_candles(frame, price, krw, coin, now)

    def _balance(self, ticker: str) -> Decimal:
        # get_balance 는 실패 시 None, 보유하지 않으면 0 을 반환한다.
        amount = self.upbit.get_balance(ticker)
        self._sleep(REQUEST_INTERVAL_SEC)
        if amount is None:
            raise MarketDataError(f"잔고 조회 실패: {ticker}")
        return Decimal(str(amount))

    def buy(self, ticker: str, order_krw: int, price: Decimal) -> Fill:
        # 시장가 매수는 원화 금액을 정수로 보낸다. 소수점이 들어가면 주문이 거절된다.
        result = self.upbit.buy_market_order(ticker, int(order_krw))
        self._sleep(REQUEST_INTERVAL_SEC)
        if not is_order_success(result):
            return Fill(False, f"매수 실패 | {describe_order(result)}")
        filled = self._resolve_order(result)
        if filled.get("state") == "cancel":
            return Fill(False, f"매수 주문이 취소되어 체결되지 않았습니다 | {describe_order(filled)}")
        volume = Decimal(str(filled.get("executed_volume") or "0"))
        funds = Decimal(str(filled.get("executed_funds") or "0"))
        if volume > 0 and funds > 0:
            entry = funds / volume
            return Fill(
                True,
                f"매수 성공 | {describe_order(filled)} | 평단 {format_price(entry)}",
                entry_price=entry,
                volume=floor_volume(volume),
            )
        # 주문 번호는 있는데 체결 조회가 늦으면, 같은 금액을 한 번 더 사지 않도록 추정 포지션을 남긴다.
        estimated = floor_volume(Decimal(order_krw) * (Decimal("1") - FEE_RATE) / price)
        if estimated <= 0:
            return Fill(False, f"매수 주문은 접수됐지만 수량을 계산하지 못했습니다 | {describe_order(filled)}")
        return Fill(
            True,
            f"매수 주문은 접수됐지만 체결 확인이 지연되어 현재가 기준으로 기록합니다 | {describe_order(filled)}",
            entry_price=price,
            volume=estimated,
        )

    def sell(self, ticker: str, volume: Decimal, price: Decimal) -> Fill:
        del price
        volume_text = volume_to_str(volume)
        if Decimal(volume_text) <= 0:
            return Fill(False, "매도 실패 | 매도 수량이 0 입니다")
        result = self.upbit.sell_market_order(ticker, volume_text)
        self._sleep(REQUEST_INTERVAL_SEC)
        if not is_order_success(result):
            return Fill(False, f"매도 실패 | {describe_order(result)}")
        filled = self._resolve_order(result)
        sold = Decimal(str(filled.get("executed_volume") or "0"))
        if sold <= 0 and filled.get("state") != "done":
            return Fill(False, f"매도 주문 접수 후 체결을 확인하지 못했습니다 | {describe_order(filled)}")
        if sold <= 0:
            sold = Decimal(volume_text)
        return Fill(True, f"매도 성공 | {describe_order(filled)}", volume=floor_volume(sold))

    def _resolve_order(self, first: dict) -> dict:
        """시장가 체결 내역이 채워질 때까지 주문 단건을 몇 번 조회한다."""
        current = first
        uuid = first.get("uuid")
        if not uuid:
            return current
        for _ in range(3):
            volume = Decimal(str(current.get("executed_volume") or "0"))
            if volume > 0 or current.get("state") in {"done", "cancel"}:
                return current
            self._sleep(0.5)
            fetched = self.upbit.get_order(uuid)
            self._sleep(REQUEST_INTERVAL_SEC)
            if is_order_success(fetched):
                current = fetched
        return current


# ---------------------------------------------------------------------------
# 1회 판단
# ---------------------------------------------------------------------------

def apply_fill(state: BotState, decision: Decision, fill: Fill, day: str) -> None:
    if decision.action == "buy" and fill.success and fill.entry_price is not None and fill.volume is not None:
        state.position = Position(
            trading_day=day,
            entry_price=format(fill.entry_price, "f"),
            volume=volume_to_str(fill.volume),
        )
        state.last_buy_day = day
        return
    if decision.action == "sell" and fill.success and state.position is not None:
        sold = fill.volume if fill.volume is not None else Decimal(state.position.volume)
        remaining = floor_volume(Decimal(state.position.volume) - sold)
        # 부분 체결이면 남은 수량만 다음 루프에서 다시 판다.
        state.position = None if remaining <= 0 else Position(
            trading_day=state.position.trading_day,
            entry_price=state.position.entry_price,
            volume=volume_to_str(remaining),
        )


def run_once(config: Config, exchange, state: BotState, now: datetime, log_status: bool) -> str:
    """시세를 보고 필요하면 주문한다.

    반환값은 ``hold``, ``filled``, ``rejected`` 중 하나다.
    조회 실패는 호출한 쪽으로 올려 보내고, 그 쪽에서 프로그램을 유지한다.
    """
    snap = exchange.fetch(config.ticker, now)
    decision = evaluate(config, snap, state)
    day = trading_day_key(now)

    if decision.action == "hold":
        if log_status:
            _log_status(config, snap, decision, now)
        return "hold"

    log_balances(snap, config.ticker, now)
    target_text = format_price(decision.target_price) if decision.target_price is not None else "-"
    if decision.action == "buy":
        log(
            f"매수 시도 | {decision.reason} | 현재가 {format_price(snap.price)} | "
            f"목표가 {target_text} | 주문금액 {decision.order_krw:,} KRW",
            now,
        )
        fill = exchange.buy(config.ticker, int(decision.order_krw or 0), snap.price)
    else:
        pnl = _format_rate(decision.pnl_rate) if decision.pnl_rate is not None else "-"
        # 계좌에 실제로 있는 수량만 판다. 봇이 사지 않은 코인은 여기 포함되지 않는다.
        volume = floor_volume(min(decision.sell_volume or Decimal("0"), snap.coin_volume))
        if volume <= 0:
            log("매도할 잔고가 없어 포지션 기록을 정리합니다.", now)
            state.position = None
            save_state(config.state_path, state)
            return "filled"
        log(
            f"매도 시도 | {decision.reason} | 현재가 {format_price(snap.price)} | "
            f"수익률 {pnl} | 수량 {volume_to_str(volume)}",
            now,
        )
        fill = exchange.sell(config.ticker, volume, snap.price)

    if fill.success:
        log(f"주문 성공 | {fill.message}", now)
        apply_fill(state, decision, fill, day)
        outcome = "filled"
    else:
        log(f"주문 실패 | {fill.message}", now)
        outcome = "rejected"
    save_state(config.state_path, state)
    return outcome


def _log_status(config: Config, snap: Snapshot, decision: Decision, now: datetime) -> None:
    target = format_price(decision.target_price) if decision.target_price is not None else "-"
    coin = config.ticker.split("-", 1)[1]
    extra = ""
    if decision.pnl_rate is not None:
        extra = f" | 평가손익 {_format_rate(decision.pnl_rate)}"
    log(
        f"상태 | {decision.reason} | 현재가 {format_price(snap.price)} | 목표가 {target} | "
        f"원화 {format_krw(snap.krw_balance)} | {coin} {volume_to_str(snap.coin_volume)}{extra}",
        now,
    )


def build_exchange(config: Config, state: BotState, market: Optional[MarketData] = None):
    market = market or MarketData()
    if not config.live:
        return PaperExchange(state, market, config.paper_krw)
    if not ACCESS_KEY or not SECRET_KEY:
        raise SystemExit(
            "실거래에는 환경 변수 UPBIT_ACCESS_KEY 와 UPBIT_SECRET_KEY 가 필요합니다. "
            "키는 코드에 넣지 말고 셸에서 export 하세요."
        )
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    return LiveExchange(upbit, market)


def run(
    config: Config,
    now_fn: Callable[[], datetime] = lambda: datetime.now(KST),
    exchange_builder: Optional[Callable[[BotState], object]] = None,
) -> int:
    state = load_state(config.state_path, config.ticker, config.paper_krw)
    _print_banner(config)
    exchange = build_exchange(config, state) if exchange_builder is None else exchange_builder(state)
    last_status = 0.0
    failures = 0

    while True:
        now = now_fn()
        show_status = config.once or (time.monotonic() - last_status) >= 60
        try:
            outcome = run_once(config, exchange, state, now, log_status=show_status)
            failures = 0
            if outcome == "hold" and show_status:
                last_status = time.monotonic()
            if outcome == "filled":
                # 체결 직후 잔고가 바뀌었으므로 다음 상태 로그를 바로 찍게 한다.
                last_status = 0.0
            if config.once:
                save_state(config.state_path, state)
                return 0
            # 거절된 주문은 잔고 부족 같은 이유로 매초 반복되지 않게 더 쉰다.
            pause = ORDER_RETRY_SEC if outcome == "rejected" else config.loop_seconds
            time.sleep(max(pause, REQUEST_INTERVAL_SEC))
        except KeyboardInterrupt:
            log("사용자 중단으로 종료합니다.", now)
            save_state(config.state_path, state)
            return 0
        except Exception as exc:
            # 네트워크, 응답 형식, 일시적 시세 오류가 나도 프로세스를 유지한다.
            failures += 1
            log(f"오류가 발생했지만 종료하지 않습니다 ({failures}회): {type(exc).__name__}: {exc}", now)
            if config.once:
                return 1
            time.sleep(max(config.loop_seconds, ORDER_RETRY_SEC))


def _print_banner(config: Config) -> None:
    mode = "실거래 (실제 주문이 나갑니다)" if config.live else "모의매매 (실제 주문 없음)"
    print("=" * 62, flush=True)
    print("업비트 변동성 돌파 자동매매", flush=True)
    print(f"모드     : {mode}", flush=True)
    print(f"종목     : {config.ticker}", flush=True)
    print(
        f"매개변수 : K={config.k} | 익절={_format_rate(config.take_profit)} | "
        f"손절=-{config.stop_loss * Decimal('100'):.2f}% | 투자비율={config.invest_ratio * Decimal('100'):.0f}%",
        flush=True,
    )
    print(f"상태파일 : {config.state_path}", flush=True)
    print("중지     : Ctrl+C", flush=True)
    print("=" * 62, flush=True)
    if config.live:
        print("주의: 실거래 모드입니다. 손실이 날 수 있습니다.", flush=True)


def _env_decimal(name: str, default: str) -> Decimal:
    raw = os.environ.get(name, default).strip()
    try:
        return Decimal(raw)
    except Exception as exc:
        raise SystemExit(f"환경 변수 {name} 값이 숫자가 아닙니다: {raw}") from exc


def _env_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"환경 변수 {name} 값이 정수가 아닙니다: {raw}") from exc


def _env_flag_is_live() -> bool:
    raw = os.environ.get("UPBIT_DRY_RUN", "1").strip().lower()
    return raw in {"0", "false", "no"}


def parse_args(argv: Optional[list[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="업비트 변동성 돌파 자동매매")
    parser.add_argument("--live", action="store_true", help="실제 주문을 전송한다")
    parser.add_argument("--paper", action="store_true", help="모의매매로 강제한다")
    parser.add_argument("--once", action="store_true", help="한 번만 판단하고 종료한다")
    parser.add_argument("--ticker", default=os.environ.get("UPBIT_TICKER", "KRW-BTC"))
    args = parser.parse_args(argv)

    if args.live and args.paper:
        parser.error("--live 와 --paper 는 함께 쓸 수 없습니다.")
    if args.live:
        live = True
    elif args.paper:
        live = False
    else:
        live = _env_flag_is_live()

    ticker = args.ticker.strip().upper()
    if not ticker.startswith("KRW-") or len(ticker.split("-")) != 2:
        raise SystemExit("종목은 KRW-BTC 처럼 원화 마켓 코드여야 합니다.")

    config = Config(
        ticker=ticker,
        k=_env_decimal("UPBIT_K", "0.5"),
        take_profit=_env_decimal("UPBIT_TAKE_PROFIT", "0.03"),
        stop_loss=_env_decimal("UPBIT_STOP_LOSS", "0.02"),
        invest_ratio=_env_decimal("UPBIT_INVEST_RATIO", "0.5"),
        min_order_krw=_env_int("UPBIT_MIN_ORDER_KRW", "5000"),
        no_buy_minutes=_env_int("UPBIT_NO_BUY_MINUTES", "5"),
        loop_seconds=float(_env_decimal("UPBIT_LOOP_SECONDS", "1")),
        paper_krw=_env_int("UPBIT_PAPER_KRW", "1000000"),
        live=live,
        once=args.once,
        state_path=Path("state") / f"{'live' if live else 'paper'}_{ticker}.json",
    )
    _validate_config(config)
    return config


def _validate_config(config: Config) -> None:
    if config.k <= 0:
        raise SystemExit("UPBIT_K 는 0보다 커야 합니다.")
    if config.take_profit <= 0:
        raise SystemExit("UPBIT_TAKE_PROFIT 는 0보다 커야 합니다. 예: 0.03")
    if not Decimal("0") < config.stop_loss < 1:
        raise SystemExit("UPBIT_STOP_LOSS 는 0과 1 사이여야 합니다. 예: 0.02")
    if not Decimal("0") < config.invest_ratio <= 1:
        raise SystemExit("UPBIT_INVEST_RATIO 는 0 초과 1 이하여야 합니다. 예: 0.5")
    if config.min_order_krw < 5000:
        raise SystemExit("UPBIT_MIN_ORDER_KRW 는 업비트 최소 주문 금액 5000원 이상이어야 합니다.")
    if config.loop_seconds < 0.2:
        raise SystemExit("UPBIT_LOOP_SECONDS 는 0.2초 이상이어야 합니다.")
    if config.no_buy_minutes < 0:
        raise SystemExit("UPBIT_NO_BUY_MINUTES 는 0 이상이어야 합니다.")
    if config.paper_krw < 0:
        raise SystemExit("UPBIT_PAPER_KRW 는 0 이상이어야 합니다.")


def main(argv: Optional[list[str]] = None) -> int:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
