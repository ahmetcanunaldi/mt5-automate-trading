"""Event-driven single-position backtest engine with FundingPips-style risk guards.

Execution model (conservative):
- Bars are BID prices (MT5 convention). Ask = bid + spread.
- Long: enter at ask (+slippage), exit at bid. Short: enter at bid (-slippage), exit at ask.
- SL/TP are checked on each execution bar; if both are touched in the same bar, SL wins (pessimistic).
  Gaps through SL fill at the bar open.
- Commission is charged per lot round-turn.
- One position at a time, never opposite positions (no hedging), no averaging/grid/martingale.

Guards (internal limits, stricter than FundingPips):
- risk per trade = risk_pct * balance, lot rounded DOWN to 0.01; trade skipped if < min lot.
- daily: day reference = balance at the first bar of the server day (we are flat overnight by design).
  new entries stop once day P&L <= -daily_soft_pct; open position force-closed if
  day P&L incl. floating <= -daily_hard_pct.
- total: risk halves below total_derisk_pct drawdown; trading stops for good at total_stop_pct.
- intraday: no entries after `last_entry_min` (minute of server day), flatten at `flatten_min`,
  Friday flatten at `fri_flatten_min`; news masks from calendar_news.blackout_masks.
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from numba import njit

POINT = 0.01
CONTRACT = 100.0  # oz per lot -> $1 move = $100 per lot


@dataclass
class Costs:
    commission_per_lot: float = 7.0   # USD round turn (conservative for prop firm metals)
    slippage_pts: float = 5.0          # per side, points (0.05$)
    spread_mult: float = 1.0           # stress test multiplier on recorded spread
    min_spread_pts: float = 15.0       # floor on spread


@dataclass
class Guards:
    initial_balance: float = 10_000.0
    risk_pct: float = 0.5
    daily_soft_pct: float = 2.0
    daily_hard_pct: float = 3.0
    total_derisk_pct: float = 6.5
    total_stop_pct: float = 8.0
    max_trades_day: int = 5
    first_entry_min: int = 60 + 5      # 01:05 server (after daily break)
    last_entry_min: int = 22 * 60      # 22:00 server
    flatten_min: int = 23 * 60 + 45    # 23:45 server
    fri_flatten_min: int = 22 * 60 + 30


@njit(cache=True)
def _run(t_min, dow, day_id, o, h, l, c, spr, block, flat,
         s_idx, s_dir, s_sl, s_tp, s_hold, s_be, s_trail,
         init_bal, risk_pct, soft, hard, derisk, tstop, max_td, first_m, last_m, flat_m, fri_m,
         comm, slip):
    n = len(o)
    ns = len(s_idx)
    max_tr = ns + 1
    tr_ei = np.full(max_tr, -1, np.int64); tr_xi = np.full(max_tr, -1, np.int64)
    tr_dir = np.zeros(max_tr, np.int64); tr_ep = np.zeros(max_tr); tr_xp = np.zeros(max_tr)
    tr_lot = np.zeros(max_tr); tr_pnl = np.zeros(max_tr); tr_reason = np.zeros(max_tr, np.int64)
    tr_risk = np.zeros(max_tr)
    eq_close = np.zeros(n)
    ntr = 0
    bal = init_bal
    in_pos = False
    d = 0; ep = 0.0; lot = 0.0; slp = 0.0; tpp = 0.0; ei = 0; deadline = 0; be_px = 0.0; be_done = False
    trail = 0.0
    cur_day = -1; day_ref = bal; day_trades = 0; stopped = False
    si = 0
    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]; day_ref = bal; day_trades = 0
        # ---- entry at bar open ----
        while si < ns and s_idx[si] < i:
            si += 1
        if (not in_pos) and si < ns and s_idx[si] == i and not stopped:
            tm = t_min[i]
            ok = (tm >= first_m) and (tm <= last_m) and (not block[i]) and (day_trades < max_td)
            if dow[i] == 4 and tm >= fri_m - 60:
                ok = False
            day_pnl = bal - day_ref
            if day_pnl <= -soft / 100.0 * day_ref:
                ok = False
            if ok:
                dd_tot = (init_bal - bal) / init_bal * 100.0
                rp = risk_pct if dd_tot < derisk else risk_pct * 0.5
                # the new trade's full risk must fit inside the hard daily limit
                room = hard / 100.0 * day_ref + day_pnl
                risk_usd = min(rp / 100.0 * bal, room)
                dist = s_sl[si]
                lots = np.floor(risk_usd / (dist * 100.0 + comm) / 0.01 + 1e-9) * 0.01
                if lots >= 0.01:
                    d = s_dir[si]
                    sp = spr[i]
                    if d == 1:
                        ep = o[i] + sp + slip
                        slp = ep - dist; tpp = ep + s_tp[si]
                    else:
                        ep = o[i] - slip
                        slp = ep + dist; tpp = ep - s_tp[si]
                    lot = lots; ei = i; deadline = s_hold[si]; in_pos = True
                    be_px = s_be[si]; be_done = False; trail = s_trail[si]
                    day_trades += 1
                    tr_ei[ntr] = i; tr_dir[ntr] = d; tr_ep[ntr] = ep; tr_lot[ntr] = lot
                    tr_risk[ntr] = lots * (dist * 100.0 + comm)
        # ---- manage open position within bar i ----
        if in_pos:
            sp = spr[i]
            xp = 0.0; reason = 0
            if d == 1:
                if o[i] <= slp and i > ei:
                    xp = o[i] - slip; reason = 1
                elif l[i] <= slp:
                    xp = slp - slip; reason = 1
                elif h[i] >= tpp:
                    xp = tpp; reason = 2
            else:
                if o[i] + sp >= slp and i > ei:
                    xp = o[i] + sp + slip; reason = 1
                elif h[i] + sp >= slp:
                    xp = slp + slip; reason = 1
                elif l[i] + sp <= tpp:
                    xp = tpp; reason = 2
            # hard daily guard on worst price of the bar
            if reason == 0:
                worst = (l[i] - ep) if d == 1 else (ep - (h[i] + sp))
                if (bal - day_ref) + worst * lot * CONTRACT - lot * comm <= -hard / 100.0 * day_ref:
                    lim = (-hard / 100.0 * day_ref - (bal - day_ref) + lot * comm) / (lot * CONTRACT)
                    xp = ep + lim if d == 1 else ep - lim
                    reason = 5
            # trailing stop (chandelier on bar extremes; applies from next bar)
            if reason == 0 and trail > 0:
                if d == 1:
                    if h[i] - trail > slp:
                        slp = h[i] - trail
                else:
                    if l[i] + sp + trail < slp:
                        slp = l[i] + sp + trail
            # break-even move (after the bar, applies from next bar)
            if reason == 0 and be_px > 0 and not be_done:
                if (d == 1 and h[i] - ep >= be_px) or (d == -1 and ep - (l[i] + sp) >= be_px):
                    slp = ep; be_done = True
            if reason == 0:
                if i - ei + 1 >= deadline:
                    reason = 3
                elif flat[i]:
                    reason = 4
                elif t_min[i] >= flat_m or (dow[i] == 4 and t_min[i] >= fri_m):
                    reason = 6
                elif i + 1 >= n or day_id[i + 1] != day_id[i]:
                    reason = 6
                if reason != 0:
                    xp = c[i] if d == 1 else c[i] + sp
            if reason != 0:
                pnl = (xp - ep) * d * lot * CONTRACT - lot * comm
                bal += pnl
                tr_xi[ntr] = i; tr_xp[ntr] = xp; tr_pnl[ntr] = pnl; tr_reason[ntr] = reason
                ntr += 1
                in_pos = False
                if (init_bal - bal) / init_bal * 100.0 >= tstop:
                    stopped = True
        # equity at bar close
        if in_pos:
            mark = c[i] if d == 1 else c[i] + spr[i]
            eq_close[i] = bal + (mark - ep) * d * lot * CONTRACT
        else:
            eq_close[i] = bal
    return (tr_ei[:ntr], tr_xi[:ntr], tr_dir[:ntr], tr_ep[:ntr], tr_xp[:ntr], tr_lot[:ntr],
            tr_pnl[:ntr], tr_reason[:ntr], tr_risk[:ntr], eq_close)


REASONS = {1: "SL", 2: "TP", 3: "TIME", 4: "NEWS", 5: "DAILY_GUARD", 6: "EOD"}


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series          # equity at each execution bar close
    daily: pd.DataFrame        # per server day: start, end, min equity
    params: dict = field(default_factory=dict)


def prepare_exec(bars: pd.DataFrame, bar_minutes: int, news_block=None, news_flat=None, costs: Costs = Costs()):
    idx = bars.index
    spr_pts = np.maximum(bars["spread"].to_numpy(float) * costs.spread_mult, costs.min_spread_pts)
    x = {
        "t_min": (idx.hour * 60 + idx.minute).to_numpy(np.int64),
        "dow": idx.dayofweek.to_numpy(np.int64),
        "day_id": idx.values.astype("datetime64[D]").astype(np.int64),
        "o": bars["open"].to_numpy(float), "h": bars["high"].to_numpy(float),
        "l": bars["low"].to_numpy(float), "c": bars["close"].to_numpy(float),
        "spr": spr_pts * POINT,
        "block": np.zeros(len(bars), bool) if news_block is None else news_block,
        "flat": np.zeros(len(bars), bool) if news_flat is None else news_flat,
        "index": idx, "bar_minutes": bar_minutes,
    }
    return x


def run(exec_x: dict, signals: pd.DataFrame, guards: Guards = Guards(), costs: Costs = Costs()) -> Result:
    """signals: DataFrame indexed by decision time (bar CLOSE time of the signal bar) with columns
    dir (+1/-1), sl (USD distance), tp (USD distance), hold_min (max holding minutes), be (USD, 0=off).
    Entry happens at the open of the first execution bar starting at/after the decision time."""
    idx = exec_x["index"]
    sig = signals.sort_index()
    s_idx = np.searchsorted(idx.values.astype("datetime64[ns]"), sig.index.values.astype("datetime64[ns]"),
                            side="left").astype(np.int64)
    keep = s_idx < len(idx)
    sig = sig[keep]; s_idx = s_idx[keep]
    # one signal per execution bar (first wins)
    _, first = np.unique(s_idx, return_index=True)
    sig = sig.iloc[first]; s_idx = s_idx[first]
    bm = exec_x["bar_minutes"]
    s_hold = np.maximum(1, np.ceil(sig["hold_min"].to_numpy(float) / bm)).astype(np.int64)
    be = sig["be"].to_numpy(float) if "be" in sig else np.zeros(len(sig))
    trail = sig["trail"].to_numpy(float) if "trail" in sig else np.zeros(len(sig))
    out = _run(exec_x["t_min"], exec_x["dow"], exec_x["day_id"], exec_x["o"], exec_x["h"], exec_x["l"],
               exec_x["c"], exec_x["spr"], exec_x["block"], exec_x["flat"],
               s_idx, sig["dir"].to_numpy(np.int64), sig["sl"].to_numpy(float), sig["tp"].to_numpy(float),
               s_hold, be, trail,
               guards.initial_balance, guards.risk_pct, guards.daily_soft_pct, guards.daily_hard_pct,
               guards.total_derisk_pct, guards.total_stop_pct, guards.max_trades_day,
               guards.first_entry_min, guards.last_entry_min, guards.flatten_min, guards.fri_flatten_min,
               costs.commission_per_lot, costs.slippage_pts * POINT)
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, eq = out
    trades = pd.DataFrame({
        "entry_time": idx[ei], "exit_time": idx[xi] + pd.Timedelta(minutes=bm) * (rsn >= 3),
        "dir": dr, "entry": ep, "exit": xp, "lots": lot, "pnl": pnl, "risk_usd": risk,
        "reason": [REASONS[r] for r in rsn],
    })
    trades["R"] = trades["pnl"] / trades["risk_usd"]
    equity = pd.Series(eq, index=idx, name="equity")
    day = idx.normalize()
    daily = pd.DataFrame({"end": equity.groupby(day).last(), "min": equity.groupby(day).min()})
    daily["start"] = daily["end"].shift(1).fillna(guards.initial_balance)
    return Result(trades, equity, daily, {"guards": asdict(guards), "costs": asdict(costs)})
