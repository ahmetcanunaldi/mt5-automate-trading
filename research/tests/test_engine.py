import numpy as np
import pandas as pd
import pytest

from research.calendar_news import blackout_masks, utc_to_server
from research.engine import Costs, Guards, prepare_exec, run
from research.metrics import summarize

ZERO = Costs(commission_per_lot=0.0, slippage_pts=0.0, min_spread_pts=0.0)


def make_bars(prices, start="2025-03-04 10:00", spread=0.0):
    """prices: list of (o,h,l,c) M1 bars."""
    idx = pd.date_range(start, periods=len(prices), freq="1min")
    df = pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=idx)
    df["spread"] = spread
    return df


def sig(t, d=1, sl=5.0, tp=6.0, hold=600, be=0.0):
    return pd.DataFrame({"dir": [d], "sl": [sl], "tp": [tp], "hold_min": [hold], "be": [be]}, index=[pd.Timestamp(t)])


def test_lot_sizing_respects_half_percent_risk():
    bars = make_bars([(100, 100.5, 99.5, 100)] * 5 + [(100, 100, 94, 94)])
    res = run(prepare_exec(bars, 1, costs=ZERO), sig("2025-03-04 10:00", sl=5.0), Guards(), ZERO)
    t = res.trades.iloc[0]
    assert t.lots == pytest.approx(0.10)          # $50 / ($5 * 100)
    assert t.reason == "SL"
    assert t.pnl == pytest.approx(-50.0)
    assert t.pnl >= -0.005 * 10_000 - 1e-9


def test_lot_rounds_down_and_includes_commission():
    bars = make_bars([(100, 100.5, 99.5, 100)] * 3)
    c = Costs(commission_per_lot=7.0, slippage_pts=0.0, min_spread_pts=0.0)
    res = run(prepare_exec(bars, 1, costs=c), sig("2025-03-04 10:00", sl=7.0), Guards(), c)
    # 50 / (700 + 7) = 0.0707 -> 0.07
    assert res.trades.iloc[0].lots == pytest.approx(0.07)
    assert res.trades.iloc[0].risk_usd <= 50.0


def test_sl_wins_when_both_hit_same_bar():
    bars = make_bars([(100, 100, 100, 100), (100, 107, 94, 100)])
    res = run(prepare_exec(bars, 1, costs=ZERO), sig("2025-03-04 10:00"), Guards(), ZERO)
    assert res.trades.iloc[0].reason == "SL"


def test_short_uses_ask_for_exits():
    # short TP at 94 (bid entry 100); ask = bid + 1 so bid must reach 93 for the TP to fill
    bars = make_bars([(100, 100, 100, 100), (100, 100, 93.5, 94), (94, 94, 93, 93)], spread=100)
    res = run(prepare_exec(bars, 1, costs=ZERO), sig("2025-03-04 10:00", d=-1, sl=5, tp=6), Guards(), ZERO)
    t = res.trades.iloc[0]
    assert t.reason == "TP" and t.exit_time == pd.Timestamp("2025-03-04 10:02")


def test_single_position_no_hedge():
    bars = make_bars([(100, 100.2, 99.8, 100)] * 10)
    s = pd.concat([sig("2025-03-04 10:00", d=1), sig("2025-03-04 10:02", d=-1)])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(), ZERO)
    assert len(res.trades) == 1   # the opposite signal is ignored while a position is open


def test_eod_flatten_and_no_late_entries():
    bars = make_bars([(100, 100.2, 99.8, 100)] * 30, start="2025-03-04 23:30")
    s = pd.concat([sig("2025-03-04 23:31")])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(), ZERO)
    assert len(res.trades) == 0   # after 22:00 no entries
    bars = make_bars([(100, 100.2, 99.8, 100)] * 200, start="2025-03-04 21:00")
    res = run(prepare_exec(bars, 1, costs=ZERO), sig("2025-03-04 21:30", hold=10_000), Guards(), ZERO)
    t = res.trades.iloc[0]
    assert t.reason == "EOD" and t.exit_time <= pd.Timestamp("2025-03-05 00:00")


def test_daily_soft_stop_blocks_new_entries():
    # risk is 0.5% of the CURRENT balance (lots round down): -50, -45 x4 -> after 5 losers day P&L = -2.3%
    # which is beyond the 2% soft stop, so the 6th signal must be blocked
    rows = []
    for _ in range(6):
        rows += [(100, 100, 100, 100), (100, 100, 94, 94), (100, 100, 100, 100)]
    bars = make_bars(rows)
    times = [bars.index[3 * k] for k in range(6)]
    s = pd.concat([sig(t) for t in times])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(max_trades_day=10), ZERO)
    assert len(res.trades) == 5
    assert summarize(res)["max_daily_dd_pct"] < 3.0


def test_news_masks():
    news = pd.DatetimeIndex([pd.Timestamp("2025-03-04 15:30")])
    idx = pd.date_range("2025-03-04 14:00", periods=180, freq="1min")
    block, flat = blackout_masks(idx, 1, news, 30, 30, 10)
    s = pd.Series(block, idx)
    assert not s["2025-03-04 15:00"] and s["2025-03-04 15:01"] and s["2025-03-04 15:59"] and not s["2025-03-04 16:00"]
    f = pd.Series(flat, idx)
    first = f[f].index[0]
    assert first == pd.Timestamp("2025-03-04 15:19")   # closing at 15:20 leaves 10 min


def test_calendar_timezone():
    # NFP 2026-01-09 13:30 UTC -> server UTC+2 = 15:30 ; 2026-07-02 12:30 UTC -> server UTC+3 = 15:30
    t = pd.DatetimeIndex(["2026-01-09 13:30", "2026-07-02 12:30"])
    assert list(utc_to_server(t)) == [pd.Timestamp("2026-01-09 15:30"), pd.Timestamp("2026-07-02 15:30")]


def test_trailing_stop_locks_profit():
    # long from 100, trail $3: high 110 -> stop 107; next bar drops to 105 -> exit 107
    bars = make_bars([(100, 100, 100, 100), (100, 110, 100, 109), (109, 109, 105, 105)])
    s = sig("2025-03-04 10:00", sl=5, tp=50)
    s["trail"] = 3.0
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(), ZERO)
    t = res.trades.iloc[0]
    assert t.reason == "SL" and t.exit == pytest.approx(107.0) and t.pnl > 0


def test_multi_position_same_direction_only():
    bars = make_bars([(100, 100.2, 99.8, 100)] * 10)
    s = pd.concat([sig("2025-03-04 10:00", d=1), sig("2025-03-04 10:02", d=1), sig("2025-03-04 10:04", d=-1)])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(max_positions=3, max_open_risk_pct=1.5), ZERO)
    assert list(res.trades.dir) == [1, 1]          # the short is rejected while longs are open (no hedge)


def test_open_risk_cap():
    bars = make_bars([(100, 100.2, 99.8, 100)] * 10)
    s = pd.concat([sig(f"2025-03-04 10:0{k}", d=1) for k in range(4)])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(max_positions=4, max_open_risk_pct=1.0), ZERO)
    assert len(res.trades) == 2 and res.trades.risk_usd.sum() <= 100.0 + 1e-9


def test_daily_hard_guard_multi():
    # two longs of 0.5% each, then a crash: guard must cap the day loss at 3% max (here SLs hit first at -1%)
    bars = make_bars([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 80, 80)])
    s = pd.concat([sig("2025-03-04 10:00", d=1), sig("2025-03-04 10:01", d=1)])
    res = run(prepare_exec(bars, 1, costs=ZERO), s, Guards(max_positions=2), ZERO)
    assert res.trades.pnl.sum() >= -300.0 - 1e-6
    assert summarize(res)["max_daily_dd_pct"] <= 3.0 + 1e-6
