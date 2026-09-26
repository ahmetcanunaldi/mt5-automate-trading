import pandas as pd
import pytest

from research.engine import Guards, prepare_exec
from research.engine_limit import run_orders
from research.tests.test_engine import ZERO, make_bars


def order(act, exp, d, e, sl, tp, hold=600, group=0, sow=False):
    return pd.DataFrame({"act_time": [pd.Timestamp(act)], "exp_time": [pd.Timestamp(exp)], "dir": [d], "entry": [e],
                         "sl": [sl], "tp": [tp], "hold_min": [hold], "group": [group], "stop_on_win": [sow]})


def test_limit_fill_and_tp():
    bars = make_bars([(102, 102, 101, 101), (101, 101, 99.5, 100), (100, 104, 100, 104)])
    r = run_orders(prepare_exec(bars, 1, costs=ZERO), order("2025-03-04 10:00", "2025-03-04 11:00", 1, 100, 98, 103),
                   Guards(), ZERO)
    t = r.trades.iloc[0]
    assert t.entry == pytest.approx(100) and t.reason == "TP" and t.lots == pytest.approx(0.25)


def test_limit_not_filled_expires():
    bars = make_bars([(102, 102, 101, 101)] * 5)
    r = run_orders(prepare_exec(bars, 1, costs=ZERO), order("2025-03-04 10:00", "2025-03-04 10:03", 1, 100, 98, 103),
                   Guards(), ZERO)
    assert len(r.trades) == 0


def test_fill_bar_that_hits_sl_is_a_loss():
    bars = make_bars([(102, 102, 101, 101), (101, 101, 97, 101)])
    r = run_orders(prepare_exec(bars, 1, costs=ZERO), order("2025-03-04 10:00", "2025-03-04 11:00", 1, 100, 98, 103),
                   Guards(), ZERO)
    assert r.trades.iloc[0].reason == "SL"


def test_stop_on_win_cancels_sibling():
    bars = make_bars([(101, 101, 99.9, 100), (100, 104, 100, 104), (104, 104, 104, 104), (104, 104, 99, 100),
                      (100, 104, 100, 104)])
    o = pd.concat([order("2025-03-04 10:00", "2025-03-04 11:00", 1, 100, 98, 103, group=7, sow=True),
                   order("2025-03-04 10:02", "2025-03-04 11:00", 1, 100, 98, 103, group=7, sow=True)])
    r = run_orders(prepare_exec(bars, 1, costs=ZERO), o, Guards(max_positions=2), ZERO)
    assert len(r.trades) == 1
