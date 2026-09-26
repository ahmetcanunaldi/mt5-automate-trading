"""Event-driven backtest engine (up to K same-direction positions) with FundingPips-style risk guards.

Execution model (conservative):
- Bars are BID prices (MT5 convention). Ask = bid + spread.
- Long: enter at ask (+slippage), exit at bid. Short: enter at bid (-slippage), exit at ask.
- SL/TP are checked on each execution bar; if both are touched in the same bar, SL wins (pessimistic).
  Gaps through SL fill at the bar open.
- Commission is charged per lot round-turn.
- Up to `max_positions` positions, ALL in the same direction (never opposite positions = no hedging).
  Every position has its own server-side SL sized to risk_pct; no averaging-down/grid/martingale logic.

Guards (internal limits, stricter than FundingPips):
- risk per trade = risk_pct * balance, lot rounded DOWN to 0.01; trade skipped if < min lot.
- total open risk (sum of SL risk of open positions) <= max_open_risk_pct * balance.
- daily: day reference = balance at the first bar of the server day (we are flat overnight by design).
  New entries stop once realized day P&L <= -daily_soft_pct. A new trade's risk plus the open risk must fit
  inside the hard daily room. All positions are force-closed if day P&L incl. worst floating <= -daily_hard_pct.
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
MAXK = 8


@dataclass
class Costs:
    commission_per_lot: float = 7.0   # USD round turn (conservative for prop firm metals)
    slippage_pts: float = 5.0          # per side, points (0.05$)
    spread_mult: float = 1.0           # stress test multiplier on recorded spread
    min_spread_pts: float = 15.0       # floor on spread
    swap_long: float = -79.48          # USD per lot per night (Vantage XAUUSD, swap_mode points; 1 pt = $1/lot)
    swap_short: float = 34.41
    triple_dow: int = 2                # Wednesday rollover charged x3 (weekend)


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
    max_positions: int = 1
    max_open_risk_pct: float = 1.0
    intraday: bool = True              # False = swing: no end-of-day exit, swaps charged at rollover
    weekend_flat: bool = True          # swing only: close everything on the last bar before the weekend


@njit(cache=True)
def _run(t_min, dow, day_id, o, h, l, c, spr, block, flat,
         s_idx, s_dir, s_sl, s_tp, s_hold, s_be, s_trail,
         init_bal, risk_pct, soft, hard, derisk, tstop, max_td, first_m, last_m, flat_m, fri_m,
         comm, slip, max_pos, max_open_risk, intraday, weekend_flat, swap_l, swap_s, triple_dow):
    n = len(o)
    ns = len(s_idx)
    max_tr = ns + 1
    tr_ei = np.full(max_tr, -1, np.int64); tr_xi = np.full(max_tr, -1, np.int64)
    tr_dir = np.zeros(max_tr, np.int64); tr_ep = np.zeros(max_tr); tr_xp = np.zeros(max_tr)
    tr_lot = np.zeros(max_tr); tr_pnl = np.zeros(max_tr); tr_reason = np.zeros(max_tr, np.int64)
    tr_risk = np.zeros(max_tr); tr_sig = np.full(max_tr, -1, np.int64)
    eq_close = np.zeros(n); eq_low = np.zeros(n)
    K = max_pos
    act = np.zeros(K, np.bool_); pd_ = np.zeros(K, np.int64); pep = np.zeros(K); plot = np.zeros(K)
    psl = np.zeros(K); ptp = np.zeros(K); pei = np.zeros(K, np.int64); pdl = np.zeros(K, np.int64)
    pbe = np.zeros(K); pbed = np.zeros(K, np.bool_); ptr = np.zeros(K); prisk = np.zeros(K)
    pid = np.zeros(K, np.int64); pswap = np.zeros(K)
    ntr = 0          # trades opened (index into tr_*)
    bal = init_bal
    cur_day = -1; day_ref = bal; day_trades = 0; stopped = False
    si = 0
    for i in range(n):
        if day_id[i] != cur_day:
            if i > 0 and not intraday:
                nights = 3 if dow[i - 1] == triple_dow else 1
                for k in range(K):
                    if act[k]:
                        sw = (swap_l if pd_[k] == 1 else swap_s) * plot[k] * nights
                        bal += sw; pswap[k] += sw
            cur_day = day_id[i]; day_trades = 0
            day_ref = max(bal, eq_close[i - 1]) if i > 0 else bal
        sp = spr[i]
        # ---- entry at bar open ----
        while si < ns and s_idx[si] < i:
            si += 1
        if si < ns and s_idx[si] == i and not stopped:
            nact = 0; cur_dir = 0; open_risk = 0.0
            for k in range(K):
                if act[k]:
                    nact += 1; cur_dir = pd_[k]; open_risk += prisk[k]
            d = s_dir[si]
            tm = t_min[i]
            ok = (tm >= first_m) and (tm <= last_m) and (not block[i]) and (day_trades < max_td) and (nact < K)
            if nact > 0 and d != cur_dir:
                ok = False                                  # no hedging
            if intraday and dow[i] == 4 and tm >= fri_m - 60:
                ok = False
            day_pnl = bal - day_ref
            if day_pnl <= -soft / 100.0 * day_ref:
                ok = False
            if ok:
                dd_tot = (init_bal - bal) / init_bal * 100.0
                rp = risk_pct if dd_tot < derisk else risk_pct * 0.5
                room = hard / 100.0 * day_ref + day_pnl - open_risk
                risk_usd = min(rp / 100.0 * bal, room, max_open_risk / 100.0 * bal - open_risk)
                dist = s_sl[si]
                lots = np.floor(risk_usd / (dist * 100.0 + comm) / 0.01 + 1e-9) * 0.01
                if lots >= 0.01:
                    k = 0
                    while act[k]:
                        k += 1
                    if d == 1:
                        ep = o[i] + sp + slip
                        psl[k] = ep - dist; ptp[k] = ep + s_tp[si]
                    else:
                        ep = o[i] - slip
                        psl[k] = ep + dist; ptp[k] = ep - s_tp[si]
                    act[k] = True; pd_[k] = d; pep[k] = ep; plot[k] = lots; pei[k] = i; pdl[k] = s_hold[si]
                    pbe[k] = s_be[si]; pbed[k] = False; ptr[k] = s_trail[si]
                    prisk[k] = lots * (dist * 100.0 + comm); pid[k] = ntr; pswap[k] = 0.0
                    day_trades += 1
                    tr_ei[ntr] = i; tr_dir[ntr] = d; tr_ep[ntr] = ep; tr_lot[ntr] = lots
                    tr_risk[ntr] = prisk[k]; tr_sig[ntr] = si
                    ntr += 1
        # ---- hard daily guard on the bar's worst price (all positions share one direction) ----
        any_act = False; sum_lot = 0.0; sum_eplot = 0.0; d_all = 0; worst_pnl = 0.0
        for k in range(K):
            if act[k]:
                any_act = True; d_all = pd_[k]
                sum_lot += plot[k]; sum_eplot += pep[k] * plot[k]
                wp = (l[i] - pep[k]) if pd_[k] == 1 else (pep[k] - (h[i] + sp))
                worst_pnl += wp * plot[k] * CONTRACT - plot[k] * comm
        guard_hit = False
        if any_act and (bal - day_ref) + worst_pnl <= -hard / 100.0 * day_ref:
            guard_hit = True
            # price where total day P&L == -hard (solve linear equation for the common exit price)
            target_open_pnl = -hard / 100.0 * day_ref - (bal - day_ref) + sum_lot * comm
            if d_all == 1:
                px = (target_open_pnl / CONTRACT + sum_eplot) / sum_lot
            else:
                px = (sum_eplot - target_open_pnl / CONTRACT) / sum_lot
        # ---- manage each position ----
        for k in range(K):
            if not act[k]:
                continue
            d = pd_[k]; ep = pep[k]; slp = psl[k]; tpp = ptp[k]
            xp = 0.0; reason = 0
            if guard_hit:
                xp = px; reason = 5
            elif d == 1:
                if o[i] <= slp and i > pei[k]:
                    xp = o[i] - slip; reason = 1
                elif l[i] <= slp:
                    xp = slp - slip; reason = 1
                elif h[i] >= tpp:
                    xp = tpp; reason = 2
            else:
                if o[i] + sp >= slp and i > pei[k]:
                    xp = o[i] + sp + slip; reason = 1
                elif h[i] + sp >= slp:
                    xp = slp + slip; reason = 1
                elif l[i] + sp <= tpp:
                    xp = tpp; reason = 2
            if reason == 0 and ptr[k] > 0:
                if d == 1:
                    if h[i] - ptr[k] > psl[k]:
                        psl[k] = h[i] - ptr[k]
                else:
                    if l[i] + sp + ptr[k] < psl[k]:
                        psl[k] = l[i] + sp + ptr[k]
            if reason == 0 and pbe[k] > 0 and not pbed[k]:
                if (d == 1 and h[i] - ep >= pbe[k]) or (d == -1 and ep - (l[i] + sp) >= pbe[k]):
                    if (d == 1 and ep > psl[k]) or (d == -1 and ep < psl[k]):
                        psl[k] = ep
                    pbed[k] = True
            if reason == 0:
                if i - pei[k] + 1 >= pdl[k]:
                    reason = 3
                elif flat[i]:
                    reason = 4
                elif i + 1 >= n:
                    reason = 6
                elif intraday:
                    if t_min[i] >= flat_m or (dow[i] == 4 and t_min[i] >= fri_m) or day_id[i + 1] != day_id[i]:
                        reason = 6
                elif weekend_flat and dow[i] == 4 and day_id[i + 1] - day_id[i] > 1:
                    reason = 6
                if reason != 0:
                    xp = c[i] if d == 1 else c[i] + sp
            if reason != 0:
                pnl = (xp - ep) * d * plot[k] * CONTRACT - plot[k] * comm
                bal += pnl
                j = pid[k]
                tr_xi[j] = i; tr_xp[j] = xp; tr_pnl[j] = pnl + pswap[k]; tr_reason[j] = reason
                act[k] = False
        if (init_bal - bal) / init_bal * 100.0 >= tstop:
            stopped = True
        # equity at bar close and at the bar's worst price
        fl = 0.0; fw = 0.0
        for k in range(K):
            if act[k]:
                if pd_[k] == 1:
                    fl += (c[i] - pep[k]) * plot[k] * CONTRACT
                    fw += (l[i] - pep[k]) * plot[k] * CONTRACT
                else:
                    fl += (pep[k] - (c[i] + sp)) * plot[k] * CONTRACT
                    fw += (pep[k] - (h[i] + sp)) * plot[k] * CONTRACT
        eq_close[i] = bal + fl
        eq_low[i] = min(bal + fw, bal + fl)
    return (tr_ei[:ntr], tr_xi[:ntr], tr_dir[:ntr], tr_ep[:ntr], tr_xp[:ntr], tr_lot[:ntr],
            tr_pnl[:ntr], tr_reason[:ntr], tr_risk[:ntr], tr_sig[:ntr], eq_close, eq_low)


REASONS = {1: "SL", 2: "TP", 3: "TIME", 4: "NEWS", 5: "DAILY_GUARD", 6: "EOD"}


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series          # equity at each execution bar close
    daily: pd.DataFrame        # per server day: start, end, min (intrabar worst) equity
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
    dir (+1/-1), sl (USD distance), tp (USD distance), hold_min (max holding minutes), be (USD, 0=off),
    trail (USD, 0=off), optional leg. Entry happens at the open of the first execution bar at/after the
    decision time. One entry per execution bar (first signal wins)."""
    assert 1 <= guards.max_positions <= MAXK
    idx = exec_x["index"]
    sig = signals.sort_index(kind="stable")
    s_idx = np.searchsorted(idx.values.astype("datetime64[ns]"), sig.index.values.astype("datetime64[ns]"),
                            side="left").astype(np.int64)
    keep = s_idx < len(idx)
    sig = sig[keep]; s_idx = s_idx[keep]
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
               costs.commission_per_lot, costs.slippage_pts * POINT, guards.max_positions,
               guards.max_open_risk_pct, guards.intraday, guards.weekend_flat, costs.swap_long, costs.swap_short,
               costs.triple_dow)
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi, eq, eql = out
    closed = xi >= 0
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi = (a[closed] for a in (ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi))
    trades = pd.DataFrame({
        "entry_time": idx[ei], "exit_time": idx[xi] + pd.Timedelta(minutes=bm) * (rsn >= 3),
        "dir": dr, "entry": ep, "exit": xp, "lots": lot, "pnl": pnl, "risk_usd": risk,
        "reason": [REASONS[r] for r in rsn],
    })
    if "leg" in sig:
        trades["leg"] = sig["leg"].to_numpy()[sgi]
    trades["R"] = trades["pnl"] / trades["risk_usd"]
    trades = trades.sort_values("exit_time", kind="stable").reset_index(drop=True)
    equity = pd.Series(eq, index=idx, name="equity")
    day = idx.normalize()
    daily = pd.DataFrame({"end": equity.groupby(day).last(),
                          "min": pd.Series(eql, index=idx).groupby(day).min()})
    daily["start"] = daily["end"].shift(1).fillna(guards.initial_balance)
    return Result(trades, equity, daily, {"guards": asdict(guards), "costs": asdict(costs)})
