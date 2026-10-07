"""변동성 돌파 판단, 모의 체결, 주문 결과 해석을 네트워크 없이 검증한다."""

import json
import os
import sys
import unittest
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trader
from trader import (
    ACCESS_KEY,
    BotState,
    Config,
    FEE_RATE,
    Fill,
    KST,
    MarketData,
    Position,
    REQUEST_INTERVAL_SEC,
    SECRET_KEY,
    Snapshot,
    calc_order_krw,
    calc_target_price,
    evaluate,
    is_order_success,
    load_state,
    minutes_until_session_close,
    parse_args,
    run,
    run_once,
    save_state,
    trading_day_key,
    volume_to_str,
)


ENV_KEYS = [
    "UPBIT_ACCESS_KEY",
    "UPBIT_SECRET_KEY",
    "UPBIT_DRY_RUN",
    "UPBIT_TICKER",
    "UPBIT_K",
    "UPBIT_TAKE_PROFIT",
    "UPBIT_STOP_LOSS",
    "UPBIT_INVEST_RATIO",
    "UPBIT_PAPER_KRW",
    "UPBIT_LOOP_SECONDS",
    "UPBIT_NO_BUY_MINUTES",
    "UPBIT_MIN_ORDER_KRW",
    "UPBIT_MAX_POSITIONS",
]


def kst(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=KST)


def snapshot(
    now,
    price,
    krw="1000000",
    coin="0",
    today_open="100",
    prev_high="110",
    prev_low="90",
    candle_day=None,
    ticker="KRW-BTC",
):
    return Snapshot(
        now=now,
        price=Decimal(price),
        today_open=None if today_open is None else Decimal(today_open),
        prev_high=None if prev_high is None else Decimal(prev_high),
        prev_low=None if prev_low is None else Decimal(prev_low),
        candle_day=trading_day_key(now) if candle_day is None else candle_day,
        krw_balance=Decimal(krw),
        coin_volume=Decimal(coin),
        ticker=ticker,
    )


class StrategyTest(unittest.TestCase):
    def test_target_price_uses_half_of_previous_range(self):
        # 시가 100, 전일 고가 110, 전일 저가 90, K 0.5 → 목표가 110
        target = calc_target_price(Decimal("100"), Decimal("110"), Decimal("90"), Decimal("0.5"))
        self.assertEqual(target, Decimal("110"))

    def test_trading_day_rolls_at_0900_kst(self):
        self.assertEqual(trading_day_key(kst(2026, 10, 7, 8, 59)), "2026-10-06")
        self.assertEqual(trading_day_key(kst(2026, 10, 7, 9, 0)), "2026-10-07")
        # 시간대가 없으면 한국시간으로 본다.
        self.assertEqual(trading_day_key(datetime(2026, 10, 7, 8, 30)), "2026-10-06")

    def test_minutes_until_close(self):
        self.assertAlmostEqual(minutes_until_session_close(kst(2026, 10, 7, 8, 56)), 4.0)
        self.assertAlmostEqual(minutes_until_session_close(kst(2026, 10, 7, 9, 0)), 24 * 60)

    def test_buy_when_price_reaches_target_with_half_balance(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(Config(ticker="KRW-BTC"), snapshot(now, "110"), BotState())
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.order_krw, 500_000)
        self.assertEqual(decision.target_price, Decimal("110"))

    def test_hold_below_target(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(Config(), snapshot(now, "109.9"), BotState())
        self.assertEqual(decision.action, "hold")

    def test_no_second_buy_same_trading_day(self):
        now = kst(2026, 10, 7, 11, 0)
        state = BotState(last_buy_days={"KRW-BTC": "2026-10-07"})
        decision = evaluate(Config(), snapshot(now, "150"), state)
        self.assertEqual(decision.action, "hold")

    def test_no_buy_in_last_five_minutes(self):
        now = kst(2026, 10, 7, 8, 56)
        # 08:56 의 거래일은 10월 6일 09:00 에 시작한 장이다.
        decision = evaluate(
            Config(),
            snapshot(now, "150", candle_day="2026-10-06"),
            BotState(),
        )
        self.assertEqual(decision.action, "hold")
        self.assertIn("장 마감", decision.reason)

    def test_skip_buy_when_order_would_be_under_5000(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(Config(ticker="KRW-BTC"), snapshot(now, "150", krw="9000"), BotState())
        self.assertEqual(decision.action, "hold")
        self.assertIsNone(decision.order_krw)

    def test_take_profit_at_plus_3_percent(self):
        now = kst(2026, 10, 7, 12, 0)
        state = BotState(positions={"KRW-BTC": Position("2026-10-07", "100000", "0.01")})
        hold = evaluate(Config(), snapshot(now, "102999", coin="0.01"), state)
        self.assertEqual(hold.action, "hold")
        sell = evaluate(Config(), snapshot(now, "103000", coin="0.01"), state)
        self.assertEqual(sell.action, "sell")
        self.assertIn("익절", sell.reason)

    def test_stop_loss_at_minus_2_percent(self):
        now = kst(2026, 10, 7, 12, 0)
        state = BotState(positions={"KRW-BTC": Position("2026-10-07", "100000", "0.01")})
        hold = evaluate(Config(), snapshot(now, "98001", coin="0.01"), state)
        self.assertEqual(hold.action, "hold")
        sell = evaluate(Config(), snapshot(now, "98000", coin="0.01"), state)
        self.assertEqual(sell.action, "sell")
        self.assertIn("손절", sell.reason)

    def test_sell_entire_position_at_next_session_open(self):
        now = kst(2026, 10, 8, 9, 0)
        state = BotState(
            last_buy_days={"KRW-BTC": "2026-10-07"},
            positions={"KRW-BTC": Position("2026-10-07", "100000", "0.01")},
        )
        # 가격이 그대로여도 09:00 에는 판다.
        decision = evaluate(Config(), snapshot(now, "100000", coin="0.01"), state)
        self.assertEqual(decision.action, "sell")
        self.assertEqual(decision.sell_volume, Decimal("0.01000000"))
        self.assertIn("09:00", decision.reason)

    def test_time_exit_beats_a_new_buy_signal(self):
        now = kst(2026, 10, 8, 9, 0)
        state = BotState(positions={"KRW-BTC": Position("2026-10-07", "100", "1")})
        decision = evaluate(Config(), snapshot(now, "200", coin="1"), state)
        self.assertEqual(decision.action, "sell")

    def test_wait_when_today_candle_is_not_ready(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(
            Config(),
            snapshot(now, "150", candle_day="2026-10-06"),
            BotState(),
        )
        self.assertEqual(decision.action, "hold")
        self.assertIsNone(decision.target_price)

    def test_order_krw_never_spends_the_last_won_of_a_full_balance(self):
        self.assertEqual(calc_order_krw(Decimal("100000"), Decimal("1"), 5000), 99999)
        self.assertEqual(calc_order_krw(Decimal("10000.9"), Decimal("0.5"), 5000), 5000)
        self.assertIsNone(calc_order_krw(Decimal("9999"), Decimal("0.5"), 5000))

    def test_volume_string_is_not_scientific_notation(self):
        self.assertEqual(volume_to_str(Decimal("0.000000019")), "0.00000001")
        self.assertNotIn("e", volume_to_str(Decimal("0.00000001")))

    def test_fifty_thousand_won_still_meets_the_minimum_order(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(Config(max_positions=10), snapshot(now, "110", krw="50000"), BotState())
        # 5만 원의 5%는 2,500원으로 최소 주문보다 작다. 50% 한도인 2만 5천 원으로 한 건을 낸다.
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.order_krw, 25_000)

    def test_unregistered_ip_stops_without_retry_text(self):
        with self.assertRaises(trader.IpNotAllowed) as caught:
            trader.explain_balance_response(
                {
                    "error": {
                        "name": "no_authorization_ip",
                        "message": "The request was made from an unregistered IP address. Request IP: 203.0.113.8",
                    }
                }
            )
        self.assertIn("203.0.113.8", str(caught.exception))
        self.assertIn("다시 조회하지 않습니다", str(caught.exception))

    def test_all_market_order_uses_one_slot_of_the_fifty_percent(self):
        now = kst(2026, 10, 7, 10, 0)
        decision = evaluate(Config(max_positions=10), snapshot(now, "110"), BotState())
        # 1,000,000 * 50% / 10종목 = 50,000
        self.assertEqual(decision.action, "buy")
        self.assertEqual(decision.order_krw, 50_000)

    def test_full_book_blocks_a_new_breakout(self):
        now = kst(2026, 10, 7, 10, 0)
        state = BotState(positions={"KRW-ETH": Position("2026-10-07", "100", "1")})
        decision = evaluate(Config(max_positions=1), snapshot(now, "150"), state)
        self.assertEqual(decision.action, "hold")
        self.assertIn("최대 보유", decision.reason)


class OrderResultTest(unittest.TestCase):
    def test_success_requires_uuid_and_no_error(self):
        self.assertTrue(is_order_success({"uuid": "abc", "state": "done"}))
        self.assertFalse(is_order_success({"error": {"message": "잔고 부족"}}))
        self.assertFalse(is_order_success(None))
        self.assertFalse(is_order_success({"uuid": "abc", "error": {"message": "x"}}))


class PaperExecutionTest(unittest.TestCase):
    def test_paper_buy_and_sell_update_virtual_balance(self):
        state = BotState(paper_krw=1_000_000)
        exchange = trader.PaperExchange(state, market=None, initial_krw=1_000_000)
        bought = exchange.buy("KRW-BTC", 500_000, Decimal("100"))
        self.assertTrue(bought.success)
        self.assertEqual(state.paper_krw, 500_000)
        expected_volume = (Decimal(500_000) * (Decimal("1") - FEE_RATE) / Decimal("100"))
        expected_volume = expected_volume.quantize(Decimal("0.00000001"))
        self.assertEqual(Decimal(state.paper_volumes["KRW-BTC"]), expected_volume)

        sold = exchange.sell("KRW-BTC", expected_volume, Decimal("103"))
        self.assertTrue(sold.success)
        self.assertNotIn("KRW-BTC", state.paper_volumes)
        gross = expected_volume * Decimal("103")
        proceeds = int((gross * (Decimal("1") - FEE_RATE)).to_integral_value(rounding=ROUND_DOWN))
        self.assertEqual(state.paper_krw, 500_000 + proceeds)

    def test_run_once_buys_then_does_not_buy_again(self):
        now = kst(2026, 10, 7, 10, 0)
        later = kst(2026, 10, 7, 10, 5)
        state = BotState(paper_krw=1_000_000)
        path = Path(self.id().replace(".", "_") + ".json")
        # 테스트가 작업 디렉터리에 파일을 남기지 않게 임시 경로를 쓴다.
        path = Path(os.environ.get("TMPDIR", "/tmp")) / path.name
        config = Config(state_path=path, once=True, ticker="KRW-BTC")
        snaps = [
            snapshot(now, "110"),
            snapshot(later, "111", krw="500000", coin="1"),
        ]

        class Scripted:
            def __init__(self):
                self.buys = []

            def fetch_market(self, config, moment):
                del config, moment
                return [snaps.pop(0)]

            def buy(self, ticker, order_krw, price):
                del ticker
                self.buys.append(order_krw)
                volume = (Decimal(order_krw) * (Decimal("1") - FEE_RATE) / price).quantize(Decimal("0.00000001"))
                return Fill(True, "모의 매수 성공", entry_price=price, volume=volume)

            def sell(self, ticker, volume, price):
                del ticker, volume, price
                raise AssertionError("같은 거래일에는 바로 팔지 않아야 합니다")

        exchange = Scripted()
        try:
            with mock.patch("sys.stdout"):
                first = run_once(config, exchange, state, now, log_status=True)
                second = run_once(config, exchange, state, later, log_status=True)
            self.assertEqual(first, "filled")
            self.assertEqual(exchange.buys, [500_000])
            self.assertEqual(second, "hold")
            self.assertEqual(state.last_buy_days["KRW-BTC"], "2026-10-07")
            self.assertIn("KRW-BTC", state.positions)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["positions"]["KRW-BTC"]["trading_day"], "2026-10-07")
        finally:
            path.unlink(missing_ok=True)

    def test_failed_order_does_not_open_a_position(self):
        now = kst(2026, 10, 7, 10, 0)
        state = BotState(paper_krw=1_000_000)
        path = Path("/tmp/upbit_trader_fail.json")
        config = Config(state_path=path, ticker="KRW-BTC")

        class Reject:
            def fetch_market(self, config, moment):
                del config, moment
                return [snapshot(now, "110")]

            def buy(self, ticker, order_krw, price):
                del ticker, order_krw, price
                return Fill(False, "잔고 부족")

        try:
            with mock.patch("sys.stdout"):
                outcome = run_once(config, Reject(), state, now, log_status=False)
            self.assertEqual(outcome, "rejected")
            self.assertEqual(state.positions, {})
            self.assertEqual(state.last_buy_days, {})
        finally:
            path.unlink(missing_ok=True)

    def test_trade_log_includes_balance_and_result(self):
        now = kst(2026, 10, 7, 10, 0)
        state = BotState(paper_krw=1_000_000)
        path = Path("/tmp/upbit_trader_log.json")
        config = Config(state_path=path, ticker="KRW-BTC")

        class Scripted:
            def fetch_market(self, config, moment):
                del config, moment
                return [snapshot(now, "110")]

            def buy(self, ticker, order_krw, price):
                del ticker, order_krw
                return Fill(True, "모의 매수 성공", entry_price=price, volume=Decimal("1"))

        try:
            with mock.patch("sys.stdout") as stdout:
                run_once(config, Scripted(), state, now, log_status=False)
            text = stdout.write.call_args_list
            printed = "".join(call.args[0] for call in text)
            self.assertIn("잔고 조회", printed)
            self.assertIn("주문 성공", printed)
            self.assertIn("매수 시도", printed)
            self.assertIn("KRW-BTC", printed)
        finally:
            path.unlink(missing_ok=True)

    def test_scan_buys_the_stronger_breakout_when_only_one_slot_is_open(self):
        now = kst(2026, 10, 7, 10, 0)
        state = BotState(paper_krw=1_000_000)
        path = Path("/tmp/upbit_trader_multi.json")
        config = Config(state_path=path, max_positions=1, ticker=None)
        eth = snapshot(now, "150", ticker="KRW-ETH")
        btc = snapshot(now, "120", ticker="KRW-BTC")

        class Book:
            def __init__(self):
                self.buys = []

            def fetch_market(self, config, moment):
                del config, moment
                return [btc, eth]

            def available_krw(self):
                return Decimal(state.paper_krw or 0)

            def buy(self, ticker, order_krw, price):
                self.buys.append(ticker)
                state.paper_krw = int(state.paper_krw or 0) - order_krw
                return Fill(True, f"모의 매수 성공 | {ticker}", entry_price=price, volume=Decimal("1"))

            def sell(self, ticker, volume, price):
                del ticker, volume, price
                raise AssertionError("매도 조건이 아닙니다")

        book = Book()
        try:
            with mock.patch("sys.stdout"):
                outcome = run_once(config, book, state, now, log_status=True)
            self.assertEqual(outcome, "filled")
            self.assertEqual(book.buys, ["KRW-ETH"])
            self.assertIn("KRW-ETH", state.positions)
            self.assertNotIn("KRW-BTC", state.positions)
        finally:
            path.unlink(missing_ok=True)
            trader.quote_path_for(path).unlink(missing_ok=True)


class LoopSafetyTest(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in ENV_KEYS}
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_parse_defaults_match_requested_strategy(self):
        config = parse_args([])
        self.assertFalse(config.live)
        self.assertIsNone(config.ticker)
        self.assertEqual(config.max_positions, 10)
        self.assertEqual(config.k, Decimal("0.5"))
        self.assertEqual(config.take_profit, Decimal("0.03"))
        self.assertEqual(config.stop_loss, Decimal("0.02"))
        self.assertEqual(config.invest_ratio, Decimal("0.5"))

    def test_live_flag_does_not_put_keys_in_config(self):
        config = parse_args(["--live", "--once"])
        self.assertTrue(config.live)
        self.assertTrue(str(config.state_path).startswith("state/live_"))
        # 키는 환경 변수에서만 읽고, 설정 객체에는 담지 않는다.
        self.assertNotIn("secret", as_public_dict(config))

    def test_live_without_keys_stops_before_any_order(self):
        if ACCESS_KEY or SECRET_KEY:
            self.skipTest("이 환경에는 이미 업비트 키가 있다")
        config = Config(live=True, state_path=Path("/tmp/ignored.json"))
        with self.assertRaises(SystemExit):
            trader.build_exchange(config, BotState())

    def test_invalid_ticker_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--ticker", "BTC-ETH"])

    def test_once_survives_a_market_data_error(self):
        path = Path("/tmp/upbit_trader_boom.json")
        config = Config(state_path=path, once=True, live=False)

        class Boom:
            def fetch_market(self, config, moment):
                del config, moment
                raise RuntimeError("network down")

        try:
            with mock.patch("sys.stdout"):
                code = run(config, now_fn=lambda: kst(2026, 10, 7, 10, 0), exchange_builder=lambda state: Boom())
            self.assertEqual(code, 1)
        finally:
            path.unlink(missing_ok=True)

    def test_market_data_pauses_after_each_request(self):
        index = [pd.Timestamp("2026-10-06 09:00:00"), pd.Timestamp("2026-10-07 09:00:00")]
        frame = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [120.0, 106.0],
                "low": [80.0, 104.0],
                "close": [110.0, 105.0],
                "volume": [1.0, 1.0],
                "value": [1.0, 1.0],
            },
            index=index,
        )
        pauses = []
        with mock.patch("trader.pyupbit.get_ohlcv", return_value=frame) as ohlcv, mock.patch(
            "trader.pyupbit.get_current_price", return_value=105.0
        ):
            market = MarketData(sleep=pauses.append)
            market.daily_candles("KRW-BTC")
            market.daily_candles("KRW-BTC")
            price = market.current_price("KRW-BTC")
        self.assertEqual(ohlcv.call_count, 1)
        self.assertEqual(price, Decimal("105.0"))
        self.assertEqual(pauses, [REQUEST_INTERVAL_SEC, REQUEST_INTERVAL_SEC])

    def test_load_levels_skips_a_market_without_candles(self):
        index = [pd.Timestamp("2026-10-06 09:00:00"), pd.Timestamp("2026-10-07 09:00:00")]
        frame = pd.DataFrame(
            {
                "open": [100.0, 105.0],
                "high": [120.0, 106.0],
                "low": [80.0, 104.0],
                "close": [110.0, 105.0],
                "volume": [1.0, 1.0],
                "value": [1.0, 1.0],
            },
            index=index,
        )

        def fake_ohlcv(ticker="KRW-BTC", interval="day", count=2, to=None, period=0.1):
            del interval, count, to, period
            if ticker == "KRW-BAD":
                return None
            return frame

        market = MarketData(sleep=lambda _seconds: None)
        with mock.patch("trader.pyupbit.get_ohlcv", side_effect=fake_ohlcv):
            levels = market.load_levels(["KRW-BTC", "KRW-ETH", "KRW-BAD"])
        self.assertEqual(set(levels), {"KRW-BTC", "KRW-ETH"})
        self.assertEqual(levels["KRW-ETH"].today_open, Decimal("105"))
        self.assertEqual(levels["KRW-ETH"].candle_day, "2026-10-07")

    def test_state_round_trip(self):
        path = Path("/tmp/upbit_trader_state.json")
        state = BotState(
            last_buy_days={"KRW-BTC": "2026-10-07"},
            paper_krw=500000,
            paper_volumes={"KRW-BTC": "0.01000000"},
            positions={"KRW-BTC": Position("2026-10-07", "100", "0.01000000")},
        )
        try:
            save_state(path, state)
            loaded = load_state(path, 1)
            self.assertEqual(loaded.positions["KRW-BTC"].entry_price, "100")
            self.assertEqual(loaded.paper_krw, 500000)
            self.assertEqual(loaded.paper_volumes["KRW-BTC"], "0.01000000")
        finally:
            path.unlink(missing_ok=True)


class LiveOrderParseTest(unittest.TestCase):
    def test_buy_uses_executed_average_after_wait_state(self):
        class Delayed:
            def __init__(self):
                self.reads = 0

            def buy_market_order(self, ticker, price):
                del ticker, price
                return {"uuid": "u1", "state": "wait", "executed_volume": "0", "executed_funds": "0"}

            def get_order(self, uuid):
                self.reads += 1
                if self.reads < 2:
                    return {"uuid": uuid, "state": "wait", "executed_volume": "0", "executed_funds": "0"}
                return {
                    "uuid": uuid,
                    "state": "done",
                    "executed_volume": "0.01",
                    "executed_funds": "500000",
                    "paid_fee": "250",
                }

        exchange = trader.LiveExchange(Delayed(), market=None, sleep=lambda seconds: None)
        fill = exchange.buy("KRW-BTC", 500_000, Decimal("50000000"))
        self.assertTrue(fill.success)
        self.assertEqual(fill.entry_price, Decimal("50000000"))
        self.assertEqual(fill.volume, Decimal("0.01000000"))

    def test_cancelled_buy_is_a_failure(self):
        class Cancelled:
            def buy_market_order(self, ticker, price):
                del ticker, price
                return {"uuid": "u2", "state": "cancel", "executed_volume": "0", "executed_funds": "0"}

            def get_order(self, uuid):
                raise AssertionError("취소된 주문은 다시 조회할 필요가 없습니다")

        exchange = trader.LiveExchange(Cancelled(), market=None, sleep=lambda seconds: None)
        fill = exchange.buy("KRW-BTC", 500_000, Decimal("100"))
        self.assertFalse(fill.success)

    def test_none_response_is_a_failure(self):
        class Down:
            def buy_market_order(self, ticker, price):
                del ticker, price
                return None

        exchange = trader.LiveExchange(Down(), market=None, sleep=lambda seconds: None)
        fill = exchange.buy("KRW-BTC", 500_000, Decimal("100"))
        self.assertFalse(fill.success)
        self.assertIn("매수 실패", fill.message)


def as_public_dict(config: Config) -> dict:
    return dict(config.__dict__)


if __name__ == "__main__":
    unittest.main()
