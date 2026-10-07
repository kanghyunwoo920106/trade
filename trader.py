#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""업비트 변동성 돌파 자동매매.

전략 (래리 윌리엄스 변동성 돌파를 업비트 일봉에 맞춘 형태)
----------------------------------------------------------------
- 대상: 업비트 원화(KRW) 마켓 전체. ``--ticker KRW-BTC`` 로 한 종목만 볼 수 있다.
- 매수: 현재가가 당일 시가 + (전일 고가 - 전일 저가) * 0.5 를 돌파하면 시장가 매수.
        거래일(09:00~다음날 09:00)과 종목당 1회.
        전체 마켓에서는 원화의 50%를 최대 10종목에 나눠 쓴다.
        한 종목만 보면 그 종목에 원화의 50%를 쓴다.
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
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
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
REQUEST_INTERVAL_SEC = 0.12

# 목표가에 쓰는 전일 고저가는 장중에는 바뀌지 않는다. 일봉은 10분마다 다시 받는다.
OHLCV_CACHE_SEC = 600.0

# 원화 마켓이 수백 개라 일봉은 여러 요청을 겹치되, 시작 간격은 위 한도를 지킨다.
CANDLE_WORKERS = 4

# 주문 실패 시 같은 조건을 매초 다시 치지 않도록 잠시 쉰다.
ORDER_RETRY_SEC = 10.0

VOLUME_DECIMALS = 8


@dataclass(frozen=True)
class Config:
    """전략 숫자와 실행 모드.

    ticker 가 None 이면 업비트 원화 마켓 전체다.
    """

    ticker: Optional[str] = None
    k: Decimal = Decimal("0.5")
    take_profit: Decimal = Decimal("0.03")
    stop_loss: Decimal = Decimal("0.02")
    invest_ratio: Decimal = Decimal("0.5")
    max_positions: int = 10
    min_order_krw: int = 5000
    no_buy_minutes: int = 5
    loop_seconds: float = 2.0
    paper_krw: int = 1_000_000
    live: bool = False
    once: bool = False
    state_path: Path = Path("state/paper_KRW-ALL.json")


@dataclass
class Position:
    """봇이 직접 연 포지션. 계좌에 원래 있던 코인과 구분한다."""

    trading_day: str
    entry_price: str
    volume: str


@dataclass
class BotState:
    """여러 종목의 포지션을 한 계좌 잔고로 관리한다."""

    positions: dict[str, Position] = field(default_factory=dict)
    last_buy_days: dict[str, str] = field(default_factory=dict)
    paper_krw: Optional[int] = None
    paper_volumes: dict[str, str] = field(default_factory=dict)

    def open_position_count(self) -> int:
        return sum(1 for position in self.positions.values() if Decimal(position.volume) > 0)


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
    ticker: str = "KRW-BTC"


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


def position_slots(config: Config) -> int:
    """한 종목 모드는 그 종목 하나에 투자 비율을 모두 쓴다."""
    if config.ticker:
        return 1
    return config.max_positions


def order_budget_ratio(config: Config) -> Decimal:
    """종목 하나에 쓸 원화 비율. 전체 마켓에서는 50%를 슬롯 수로 나눈다."""
    return config.invest_ratio / Decimal(position_slots(config))


def planned_order_krw(krw_balance: Decimal, config: Config) -> Optional[int]:
    """종목당 주문 금액.

    전체 마켓은 투자 비율을 슬롯 수로 나눈다. 5만 원처럼 잔고가 작아
    그 금액이 업비트 최소 주문 5,000원보다 작으면, 투자 비율 한도 안에서
    최소 주문 금액 이상으로 한 건을 낸다.
    """
    split = calc_order_krw(krw_balance, order_budget_ratio(config), config.min_order_krw)
    if split is not None:
        return split
    return calc_order_krw(krw_balance, config.invest_ratio, config.min_order_krw)


def evaluate(config: Config, snap: Snapshot, state: BotState) -> Decision:
    """한 종목을 두고 지금 살지, 팔지, 기다릴지 정한다."""
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

    position = state.positions.get(snap.ticker)
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

    if state.last_buy_days.get(snap.ticker) == day:
        return Decision("hold", "오늘은 이미 매수해서 다시 사지 않습니다", target)

    if target is None:
        return Decision("hold", "당일 일봉이 아직 없어 목표가를 계산하지 않습니다", None)

    # 08:55~09:00 에 사면 직후 장 시작 매도로 바로 청산된다.
    if minutes_until_session_close(snap.now) < config.no_buy_minutes:
        return Decision("hold", "장 마감 직전이라 신규 매수하지 않습니다", target)

    if snap.price < target:
        return Decision("hold", "현재가가 목표가보다 낮습니다", target)

    if config.ticker is None and state.open_position_count() >= config.max_positions:
        return Decision("hold", "최대 보유 종목 수에 도달해서 신규 매수하지 않습니다", target)

    order_krw = planned_order_krw(snap.krw_balance, config)
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


def load_state(path: Path, paper_krw: int) -> BotState:
    if not path.exists():
        return BotState(paper_krw=paper_krw)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        positions = {
            ticker: Position(**payload)
            for ticker, payload in (raw.get("positions") or {}).items()
        }
        # 예전 단일 종목 파일도 읽는다.
        if not positions and raw.get("position") and raw.get("ticker"):
            positions[str(raw["ticker"])] = Position(**raw["position"])
        last_buy_days = {str(key): str(value) for key, value in (raw.get("last_buy_days") or {}).items()}
        if not last_buy_days and raw.get("last_buy_day") and raw.get("ticker"):
            last_buy_days[str(raw["ticker"])] = str(raw["last_buy_day"])
        paper_volumes = {str(key): str(value) for key, value in (raw.get("paper_volumes") or {}).items()}
        if not paper_volumes and raw.get("paper_volume") not in (None, "0") and raw.get("ticker"):
            paper_volumes[str(raw["ticker"])] = str(raw["paper_volume"])
        return BotState(
            positions=positions,
            last_buy_days=last_buy_days,
            paper_krw=raw.get("paper_krw", paper_krw),
            paper_volumes=paper_volumes,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log(f"상태 파일을 읽지 못해 새로 시작합니다: {exc}")
        return BotState(paper_krw=paper_krw)


def save_state(path: Path, state: BotState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_buy_days": state.last_buy_days,
        "paper_krw": state.paper_krw,
        "paper_volumes": state.paper_volumes,
        "positions": {ticker: asdict(position) for ticker, position in state.positions.items()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def quote_path_for(state_path: Path) -> Path:
    """포지션 파일과 겹치지 않게 시세 스냅샷 경로를 만든다."""
    return state_path.with_name(f"prices_{state_path.name}")


# ---------------------------------------------------------------------------
# 시세 / 주문
# ---------------------------------------------------------------------------

class MarketDataError(RuntimeError):
    """시세 조회 실패. 일시적인 오류는 루프를 끝내지 않는다."""


class IpNotAllowed(MarketDataError):
    """허용 IP가 아니면 재조회할 때마다 업비트가 알림을 보낸다."""


def explain_balance_response(rows: object) -> None:
    """잔고 응답이 계좌 목록이 아니면 이유를 담아 예외를 낸다."""
    if isinstance(rows, list):
        return
    name = ""
    message = ""
    if isinstance(rows, dict):
        error = rows.get("error")
        if isinstance(error, dict):
            name = str(error.get("name") or "")
            message = str(error.get("message") or "")
    if name == "no_authorization_ip":
        found = re.search(r"(\d+\.\d+\.\d+\.\d+)", message)
        ip_text = found.group(1) if found else "카카오톡 안내에 적힌 IP"
        raise IpNotAllowed(
            "업비트 Open API 허용 IP에 이 컴퓨터가 없습니다. "
            f"요청 IP: {ip_text}. "
            "업비트 Open API 관리에서 이 IP를 등록한 뒤 다시 실행하세요. "
            "등록 전에는 잔고를 다시 조회하지 않습니다."
        )
    detail = f"{name} {message}".strip() or f"응답 형식 {type(rows).__name__}"
    raise MarketDataError(f"잔고 조회에 실패했습니다. {detail}")


class MarketData:
    """일봉은 종목별로 캐시하고, 호출 뒤에는 항상 쉬어 요청 한도를 지킨다."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep
        self._candles: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def krw_tickers(self) -> list[str]:
        tickers = pyupbit.get_tickers(fiat="KRW")
        self._sleep(REQUEST_INTERVAL_SEC)
        if not tickers:
            raise MarketDataError("원화 마켓 목록을 가져오지 못했습니다.")
        return sorted(ticker for ticker in tickers if isinstance(ticker, str) and ticker.startswith("KRW-"))

    def daily_candles(self, ticker: str):
        cached = self._candles.get(ticker)
        if cached is not None and time.monotonic() - cached[0] < OHLCV_CACHE_SEC:
            return cached[1]
        frame = pyupbit.get_ohlcv(ticker, interval="day", count=2)
        self._sleep(REQUEST_INTERVAL_SEC)
        if frame is None or len(frame) < 2:
            raise MarketDataError(f"{ticker} 일봉 조회에 실패했거나 캔들이 2개보다 적습니다.")
        self._candles[ticker] = (time.monotonic(), frame)
        return frame

    def current_price(self, ticker: str) -> Decimal:
        price = pyupbit.get_current_price(ticker)
        self._sleep(REQUEST_INTERVAL_SEC)
        if not isinstance(price, (int, float)):
            raise MarketDataError(f"현재가 조회에 실패했습니다: {price!r}")
        return Decimal(str(price))

    def current_prices(self, tickers: list[str]) -> dict[str, Decimal]:
        """현재가는 한 요청에 여러 종목을 실어 원화 마켓 전체를 빨리 본다."""
        if len(tickers) == 1:
            return {tickers[0]: self.current_price(tickers[0])}
        found: dict[str, Decimal] = {}
        for start in range(0, len(tickers), 100):
            chunk = tickers[start:start + 100]
            raw = pyupbit.get_current_price(chunk if len(chunk) > 1 else chunk[0])
            self._sleep(REQUEST_INTERVAL_SEC)
            if isinstance(raw, dict):
                for market, price in raw.items():
                    if isinstance(price, (int, float)):
                        found[str(market)] = Decimal(str(price))
            elif len(chunk) == 1 and isinstance(raw, (int, float)):
                found[chunk[0]] = Decimal(str(raw))
        return found

    def load_levels(self, tickers: list[str], progress: Optional[Callable[[int, int, str], None]] = None) -> dict[str, "CandleLevels"]:
        """전일 고저가와 당일 시가. 없는 종목만 다시 받고, 실패한 종목은 건너뛴다."""
        missing = [ticker for ticker in tickers if not self._has_fresh_candles(ticker)]
        if len(missing) > 1:
            self._fetch_many(missing, progress)
        elif len(missing) == 1:
            ticker = missing[0]
            try:
                self.daily_candles(ticker)
            except MarketDataError as exc:
                log(str(exc))
            if progress is not None:
                progress(1, 1, ticker)
        levels: dict[str, CandleLevels] = {}
        for ticker in tickers:
            cached = self._candles.get(ticker)
            if cached is None:
                continue
            try:
                levels[ticker] = levels_from_frame(cached[1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return levels

    def _has_fresh_candles(self, ticker: str) -> bool:
        cached = self._candles.get(ticker)
        return cached is not None and time.monotonic() - cached[0] < OHLCV_CACHE_SEC

    def _reserve_request_slot(self) -> float:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request_at)
            self._next_request_at = scheduled + REQUEST_INTERVAL_SEC
            return scheduled - now

    def _fetch_many(self, tickers: list[str], progress: Optional[Callable[[int, int, str], None]]) -> None:
        total = len(tickers)

        def fetch_one(ticker: str):
            delay = self._reserve_request_slot()
            if delay > 0:
                self._sleep(delay)
            frame = pyupbit.get_ohlcv(ticker, interval="day", count=2)
            return ticker, frame

        done = 0
        with ThreadPoolExecutor(max_workers=CANDLE_WORKERS) as pool:
            futures = [pool.submit(fetch_one, ticker) for ticker in tickers]
            for future in as_completed(futures):
                done += 1
                try:
                    ticker, frame = future.result()
                except Exception as exc:
                    log(f"일봉 조회 오류: {type(exc).__name__}: {exc}")
                    continue
                if progress is not None and (done == 1 or done == total or done % 20 == 0):
                    progress(done, total, ticker)
                if frame is None or len(frame) < 2:
                    log(f"일봉 건너뜀 | {ticker}")
                    continue
                with self._lock:
                    self._candles[ticker] = (time.monotonic(), frame)


def _candle_day(index_value) -> str:
    """pyupbit 일봉 인덱스는 KST 09:00 의 naive 시각이다. 날짜가 거래일 키와 같다."""
    if hasattr(index_value, "to_pydatetime"):
        index_value = index_value.to_pydatetime()
    return index_value.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class CandleLevels:
    today_open: Decimal
    prev_high: Decimal
    prev_low: Decimal
    candle_day: str


def levels_from_frame(frame) -> CandleLevels:
    prev = frame.iloc[-2]
    today = frame.iloc[-1]
    return CandleLevels(
        today_open=Decimal(str(today["open"])),
        prev_high=Decimal(str(prev["high"])),
        prev_low=Decimal(str(prev["low"])),
        candle_day=_candle_day(frame.index[-1]),
    )


def snapshot_from_candles(frame, price: Decimal, krw: Decimal, coin: Decimal, now: datetime, ticker: str = "KRW-BTC") -> Snapshot:
    levels = levels_from_frame(frame)
    return Snapshot(
        now=now,
        price=price,
        today_open=levels.today_open,
        prev_high=levels.prev_high,
        prev_low=levels.prev_low,
        candle_day=levels.candle_day,
        krw_balance=krw,
        coin_volume=coin,
        ticker=ticker,
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


def build_snapshots(
    market: MarketData,
    tickers: list[str],
    now: datetime,
    krw: Decimal,
    volumes: dict[str, str],
) -> list[Snapshot]:
    """현재가와 일봉으로 종목별 판단 재료를 만든다. 실패한 종목은 건너뛴다."""
    prices = market.current_prices(tickers)

    def progress(done: int, total: int, ticker: str) -> None:
        log(f"일봉 수집 {done}/{total} | {ticker}", now)

    levels = market.load_levels(tickers, progress=progress)
    snapshots: list[Snapshot] = []
    for ticker in tickers:
        price = prices.get(ticker)
        level = levels.get(ticker)
        if price is None or level is None:
            continue
        snapshots.append(
            Snapshot(
                now=now,
                price=price,
                today_open=level.today_open,
                prev_high=level.prev_high,
                prev_low=level.prev_low,
                candle_day=level.candle_day,
                krw_balance=krw,
                coin_volume=Decimal(volumes.get(ticker, "0")),
                ticker=ticker,
            )
        )
    if tickers and not snapshots:
        raise MarketDataError("시세를 가져오지 못했습니다.")
    return snapshots


class PaperExchange:
    """공개 시세만 조회하고 주문은 가상 잔고에 반영한다."""

    def __init__(self, state: BotState, market: MarketData, initial_krw: int) -> None:
        self.state = state
        self.market = market
        if state.paper_krw is None:
            state.paper_krw = initial_krw

    def fetch_market(self, config: Config, now: datetime) -> list[Snapshot]:
        return build_snapshots(self.market, self._tickers(config), now, Decimal(state_krw(self.state)), self.state.paper_volumes)

    def _tickers(self, config: Config) -> list[str]:
        if config.ticker:
            return [config.ticker]
        return self.market.krw_tickers()

    def available_krw(self) -> Decimal:
        return Decimal(state_krw(self.state))

    def buy(self, ticker: str, order_krw: int, price: Decimal) -> Fill:
        krw = state_krw(self.state)
        if order_krw > krw:
            return Fill(False, f"모의 매수 실패 | {ticker} | 주문 금액 {order_krw:,}원이 잔고 {krw:,}원보다 큽니다")
        net = Decimal(order_krw) * (Decimal("1") - FEE_RATE)
        volume = floor_volume(net / price)
        if volume <= 0:
            return Fill(False, f"모의 매수 실패 | {ticker} | 주문 금액으로 살 수 있는 수량이 최소 단위보다 작습니다")
        held = Decimal(self.state.paper_volumes.get(ticker, "0"))
        self.state.paper_krw = krw - order_krw
        self.state.paper_volumes[ticker] = volume_to_str(held + volume)
        return Fill(
            True,
            f"모의 매수 성공 | {ticker} | 주문 {order_krw:,} KRW | 체결가 {format_price(price)} | 수량 {volume_to_str(volume)}",
            entry_price=price,
            volume=volume,
        )

    def sell(self, ticker: str, volume: Decimal, price: Decimal) -> Fill:
        held = Decimal(self.state.paper_volumes.get(ticker, "0"))
        sell_volume = floor_volume(min(volume, held))
        if sell_volume <= 0:
            return Fill(False, f"모의 매도 실패 | {ticker} | 매도할 수량이 없습니다")
        gross = sell_volume * price
        proceeds = int((gross * (Decimal("1") - FEE_RATE)).to_integral_value(rounding=ROUND_DOWN))
        self.state.paper_krw = state_krw(self.state) + proceeds
        remaining = floor_volume(held - sell_volume)
        if remaining <= 0:
            self.state.paper_volumes.pop(ticker, None)
        else:
            self.state.paper_volumes[ticker] = volume_to_str(remaining)
        return Fill(
            True,
            f"모의 매도 성공 | {ticker} | 수량 {volume_to_str(sell_volume)} | 체결가 {format_price(price)} | 정산 {proceeds:,} KRW",
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

    def fetch_market(self, config: Config, now: datetime) -> list[Snapshot]:
        tickers = [config.ticker] if config.ticker else self.market.krw_tickers()
        krw, coins = self._accounts()
        volumes = {ticker: coins.get(ticker, Decimal("0")) for ticker in tickers}
        # 문자열로 넘기면 build_snapshots 가 Decimal 로 다시 읽는다.
        text_volumes = {ticker: format(volume, "f") for ticker, volume in volumes.items()}
        return build_snapshots(self.market, tickers, now, krw, text_volumes)

    def available_krw(self) -> Decimal:
        return self._balance("KRW")

    def _accounts(self) -> tuple[Decimal, dict[str, Decimal]]:
        """전체 잔고를 한 번만 조회한다. 종목마다 잔고 API 를 치지 않는다."""
        rows = self.upbit.get_balances()
        self._sleep(REQUEST_INTERVAL_SEC)
        explain_balance_response(rows)
        krw = Decimal("0")
        coins: dict[str, Decimal] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            currency = str(row.get("currency") or "")
            if currency == "KRW" and row.get("unit_currency") in (None, "KRW"):
                krw = Decimal(str(row.get("balance") or "0"))
            elif row.get("unit_currency") == "KRW" and currency:
                coins[f"KRW-{currency}"] = Decimal(str(row.get("balance") or "0"))
        return krw, coins

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

def apply_fill(state: BotState, ticker: str, decision: Decision, fill: Fill, day: str) -> None:
    if decision.action == "buy" and fill.success and fill.entry_price is not None and fill.volume is not None:
        state.positions[ticker] = Position(
            trading_day=day,
            entry_price=format(fill.entry_price, "f"),
            volume=volume_to_str(fill.volume),
        )
        state.last_buy_days[ticker] = day
        return
    position = state.positions.get(ticker)
    if decision.action == "sell" and fill.success and position is not None:
        sold = fill.volume if fill.volume is not None else Decimal(position.volume)
        remaining = floor_volume(Decimal(position.volume) - sold)
        # 부분 체결이면 남은 수량만 다음 루프에서 다시 판다.
        if remaining <= 0:
            state.positions.pop(ticker, None)
        else:
            state.positions[ticker] = Position(
                trading_day=position.trading_day,
                entry_price=position.entry_price,
                volume=volume_to_str(remaining),
            )


def run_once(config: Config, exchange, state: BotState, now: datetime, log_status: bool) -> str:
    """원화 마켓 시세를 보고 매도 후 돌파 종목을 매수한다.

    반환값은 ``hold``, ``filled``, ``rejected`` 중 하나다.
    조회 실패는 호출한 쪽으로 올려 보내고, 그 쪽에서 프로그램을 유지한다.
    """
    snapshots = exchange.fetch_market(config, now)
    day = trading_day_key(now)
    paired = [(snap, evaluate(config, snap, state)) for snap in snapshots]
    if log_status or config.once:
        _log_scan(paired, state, now)
        _save_quotes(config.state_path, paired)

    filled = 0
    rejected = 0
    for snap, decision in paired:
        if decision.action != "sell":
            continue
        outcome = _execute(config, exchange, state, snap, decision, day, now)
        filled += outcome == "filled"
        rejected += outcome == "rejected"

    # 돌파 폭이 큰 종목부터 산다. 매도 뒤 잔고와 보유 종목 수로 다시 판단한다.
    candidates = []
    for snap, decision in paired:
        if decision.target_price is None or decision.target_price <= 0 or snap.price < decision.target_price:
            continue
        if snap.ticker in state.positions or state.last_buy_days.get(snap.ticker) == day:
            continue
        candidates.append((snap.price / decision.target_price, snap))
    candidates.sort(key=lambda item: item[0], reverse=True)

    for _strength, snap in candidates:
        krw = _available_krw(exchange, state)
        refreshed = _with_balance(snap, krw, Decimal(state.paper_volumes.get(snap.ticker, "0")))
        decision = evaluate(config, refreshed, state)
        if decision.action != "buy":
            continue
        outcome = _execute(config, exchange, state, refreshed, decision, day, now)
        filled += outcome == "filled"
        rejected += outcome == "rejected"

    if filled:
        return "filled"
    if rejected:
        return "rejected"
    return "hold"


def _available_krw(exchange, state: BotState) -> Decimal:
    available = getattr(exchange, "available_krw", None)
    if callable(available):
        return Decimal(available())
    return Decimal(state_krw(state))


def _with_balance(snap: Snapshot, krw: Decimal, coin: Decimal) -> Snapshot:
    return Snapshot(
        now=snap.now,
        price=snap.price,
        today_open=snap.today_open,
        prev_high=snap.prev_high,
        prev_low=snap.prev_low,
        candle_day=snap.candle_day,
        krw_balance=krw,
        coin_volume=coin,
        ticker=snap.ticker,
    )


def _execute(config: Config, exchange, state: BotState, snap: Snapshot, decision: Decision, day: str, now: datetime) -> str:
    log_balances(snap, snap.ticker, now)
    target_text = format_price(decision.target_price) if decision.target_price is not None else "-"
    if decision.action == "buy":
        log(
            f"매수 시도 | {snap.ticker} | {decision.reason} | 현재가 {format_price(snap.price)} | "
            f"목표가 {target_text} | 주문금액 {decision.order_krw:,} KRW",
            now,
        )
        fill = exchange.buy(snap.ticker, int(decision.order_krw or 0), snap.price)
    else:
        pnl = _format_rate(decision.pnl_rate) if decision.pnl_rate is not None else "-"
        # 계좌에 실제로 있는 수량만 판다. 봇이 사지 않은 코인은 여기 포함되지 않는다.
        volume = floor_volume(min(decision.sell_volume or Decimal("0"), snap.coin_volume))
        if volume <= 0:
            log(f"매도할 잔고가 없어 포지션 기록을 정리합니다 | {snap.ticker}", now)
            state.positions.pop(snap.ticker, None)
            save_state(config.state_path, state)
            return "filled"
        log(
            f"매도 시도 | {snap.ticker} | {decision.reason} | 현재가 {format_price(snap.price)} | "
            f"수익률 {pnl} | 수량 {volume_to_str(volume)}",
            now,
        )
        fill = exchange.sell(snap.ticker, volume, snap.price)

    if fill.success:
        log(f"주문 성공 | {fill.message}", now)
        apply_fill(state, snap.ticker, decision, fill, day)
        outcome = "filled"
    else:
        log(f"주문 실패 | {fill.message}", now)
        outcome = "rejected"
    save_state(config.state_path, state)
    return outcome


def _log_scan(paired: list[tuple[Snapshot, Decision]], state: BotState, now: datetime) -> None:
    breakouts = []
    for snap, decision in paired:
        if decision.target_price is None or decision.target_price <= 0 or snap.price < decision.target_price:
            continue
        if snap.ticker in state.positions:
            continue
        breakouts.append((snap.price / decision.target_price, snap, decision))
    breakouts.sort(key=lambda item: item[0], reverse=True)
    krw = paired[0][0].krw_balance if paired else Decimal("0")
    log(
        f"시세 포착 | 조회 {len(paired)}개 | 돌파 {len(breakouts)}개 | "
        f"보유 {state.open_position_count()}개 | 원화 {format_krw(krw)}",
        now,
    )
    for _strength, snap, decision in breakouts[:8]:
        assert decision.target_price is not None
        log(
            f"돌파 | {snap.ticker} | 현재가 {format_price(snap.price)} | "
            f"목표가 {format_price(decision.target_price)} | 초과 {_format_rate(snap.price / decision.target_price - 1)}",
            now,
        )
    by_ticker = {snap.ticker: snap for snap, _decision in paired}
    for ticker, position in state.positions.items():
        snap = by_ticker.get(ticker)
        if snap is None or Decimal(position.entry_price) <= 0:
            log(f"보유 | {ticker} | 수량 {position.volume}", now)
            continue
        pnl = price_change_rate(snap.price, Decimal(position.entry_price))
        log(
            f"보유 | {ticker} | 현재가 {format_price(snap.price)} | 평단 {position.entry_price} | "
            f"평가손익 {_format_rate(pnl)} | 수량 {position.volume}",
            now,
        )


def _save_quotes(state_path: Path, paired: list[tuple[Snapshot, Decision]]) -> None:
    rows = []
    for snap, decision in paired:
        rows.append(
            {
                "ticker": snap.ticker,
                "price": format(snap.price, "f"),
                "target": None if decision.target_price is None else format(decision.target_price, "f"),
                "action": decision.action,
            }
        )
    rows.sort(key=lambda row: row["ticker"])
    path = quote_path_for(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


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
    state = load_state(config.state_path, config.paper_krw)
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
        except IpNotAllowed as exc:
            # 같은 오류로 다시 치면 업비트가 미등록 IP 알림을 계속 보낸다.
            log(str(exc), now)
            return 2
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
    universe = config.ticker or "원화 마켓 전체"
    per_order = order_budget_ratio(config) * Decimal("100")
    print(f"종목     : {universe}", flush=True)
    print(
        f"매개변수 : K={config.k} | 익절={_format_rate(config.take_profit)} | "
        f"손절=-{config.stop_loss * Decimal('100'):.2f}% | "
        f"종목당 {per_order:.2f}% | 최대 {position_slots(config)}종목",
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
    parser.add_argument(
        "--ticker",
        default=os.environ.get("UPBIT_TICKER", "ALL"),
        help="ALL 이면 원화 마켓 전체, KRW-BTC 처럼 주면 그 종목만",
    )
    args = parser.parse_args(argv)

    if args.live and args.paper:
        parser.error("--live 와 --paper 는 함께 쓸 수 없습니다.")
    if args.live:
        live = True
    elif args.paper:
        live = False
    else:
        live = _env_flag_is_live()

    ticker = _parse_ticker(args.ticker)
    loop_default = "1" if ticker else "2"
    label = ticker or "KRW-ALL"
    config = Config(
        ticker=ticker,
        k=_env_decimal("UPBIT_K", "0.5"),
        take_profit=_env_decimal("UPBIT_TAKE_PROFIT", "0.03"),
        stop_loss=_env_decimal("UPBIT_STOP_LOSS", "0.02"),
        invest_ratio=_env_decimal("UPBIT_INVEST_RATIO", "0.5"),
        max_positions=_env_int("UPBIT_MAX_POSITIONS", "10"),
        min_order_krw=_env_int("UPBIT_MIN_ORDER_KRW", "5000"),
        no_buy_minutes=_env_int("UPBIT_NO_BUY_MINUTES", "5"),
        loop_seconds=float(_env_decimal("UPBIT_LOOP_SECONDS", loop_default)),
        paper_krw=_env_int("UPBIT_PAPER_KRW", "1000000"),
        live=live,
        once=args.once,
        state_path=Path("state") / f"{'live' if live else 'paper'}_{label}.json",
    )
    _validate_config(config)
    return config


def _parse_ticker(raw: str) -> Optional[str]:
    ticker = raw.strip().upper()
    if ticker in {"ALL", "*"}:
        return None
    parts = ticker.split("-")
    if len(parts) != 2 or parts[0] != "KRW" or not parts[1]:
        raise SystemExit("종목은 ALL 또는 KRW-BTC 같은 원화 마켓 코드여야 합니다.")
    return ticker


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
    if config.max_positions < 1:
        raise SystemExit("UPBIT_MAX_POSITIONS 는 1 이상이어야 합니다.")


def main(argv: Optional[list[str]] = None) -> int:
    config = parse_args(argv)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
