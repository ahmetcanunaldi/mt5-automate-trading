"""Pending LIMIT-order engine with the same guards as research/engine.py.

Orders carry ABSOLUTE prices: entry (limit), sl, tp. An order becomes active at `act_idx`
(decision time) and expires at `exp_idx`. Fill model (bid bars, ask = bid + spread):
  long  limit fills when ask low (l + spread) <= entry; fill price = min(entry, ask open) (gap = better fill)
  short limit fills when bid high (h) >= entry;      fill price = max(entry, bid open)
On the fill bar the position is checked against SL immediately (pessimistic: a bar that fills and
touches the SL is a loss); TP on the fill bar is only granted if the bar CLOSE is beyond TP.
Guards at fill time: session window, news block, max trades/day, soft daily stop, hard daily room,
total stop, K same-direction positions, open-risk cap. Orders that cannot be filled under the guards
are cancelled. `group` lets a strategy cancel sibling orders (e.g. "done for the day after a win").
"""
import numpy as np
import pandas as pd
from numba import njit

from research.engine import CONTRACT, POINT, REASONS, Costs, Guards, Result
from dataclasses import asdict


@njit(cache=True)
def _run_limit(t_min, dow, day_id, o, h, l, c, spr, block, flat,
               a_idx, x_idx, a_dir, a_ent, a_sl, a_tp, a_hold, a_grp, a_stop_on_win,
               init_bal, risk_pct, soft, hard, derisk, tstop, max_td, first_m, last_m, flat_m, fri_m,
               comm, slip, max_pos, max_open_risk):
    n = len(o); no = len(a_idx)
    tr_ei = np.full(no, -1, np.int64); tr_xi = np.full(no, -1, np.int64)
    tr_dir = np.zeros(no, np.int64); tr_ep = np.zeros(no); tr_xp = np.zeros(no)
    tr_lot = np.zeros(no); tr_pnl = np.zeros(no); tr_reason = np.zeros(no, np.int64)
    tr_risk = np.zeros(no); tr_ord = np.full(no, -1, np.int64)
    eq_close = np.zeros(n); eq_low = np.zeros(n)
    K = max_pos
    act = np.zeros(K, np.bool_); pd_ = np.zeros(K, np.int64); pep = np.zeros(K); plot = np.zeros(K)
    psl = np.zeros(K); ptp = np.zeros(K); pei = np.zeros(K, np.int64); pdl = np.zeros(K, np.int64)
    prisk = np.zeros(K); pid = np.zeros(K, np.int64); pgrp = np.zeros(K, np.int64)
    alive = np.ones(no, np.bool_)
    won_grp = np.zeros(no + 1, np.bool_)   # groups that already produced a winner
    ntr = 0; bal = init_bal
    cur_day = -1; day_ref = bal; day_trades = 0; stopped = False
    first_live = 0
    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]; day_ref = bal; day_trades = 0
        sp = spr[i]
        # ---- pending orders ----
        while first_live < no and (x_idx[first_live] < i or not alive[first_live]):
            first_live += 1
        q = first_live
        while q < no and a_idx[q] <= i:
            if not alive[q] or x_idx[q] < i or a_idx[q] > i:
                q += 1
                continue
            g = a_grp[q]
            if a_stop_on_win[q] and won_grp[g]:
                alive[q] = False; q += 1; continue
            d = a_dir[q]
            touched = (l[i] + sp <= a_ent[q]) if d == 1 else (h[i] >= a_ent[q])
            if not touched:
                q += 1
                continue
            alive[q] = False
            nact = 0; cur_dir = 0; open_risk = 0.0
            for k in range(K):
                if act[k]:
                    nact += 1; cur_dir = pd_[k]; open_risk += prisk[k]
            tm = t_min[i]
            ok = (not stopped) and (tm >= first_m) and (tm <= last_m) and (not block[i]) and (day_trades < max_td) \
                and (nact < K) and not (nact > 0 and d != cur_dir) and not (dow[i] == 4 and tm >= fri_m - 60)
            day_pnl = bal - day_ref
            if day_pnl <= -soft / 100.0 * day_ref:
                ok = False
            if not ok:
                q += 1
                continue
            if d == 1:
                ep = min(a_ent[q], o[i] + sp) + slip
            else:
                ep = max(a_ent[q], o[i]) - slip
            dist = (ep - a_sl[q]) * d
            if dist <= 0.05:
                q += 1
                continue
            dd_tot = (init_bal - bal) / init_bal * 100.0
            rp = risk_pct if dd_tot < derisk else risk_pct * 0.5
            room = hard / 100.0 * day_ref + day_pnl - open_risk
            risk_usd = min(rp / 100.0 * bal, room, max_open_risk / 100.0 * bal - open_risk)
            lots = np.floor(risk_usd / (dist * CONTRACT + comm) / 0.01 + 1e-9) * 0.01
            if lots < 0.01:
                q += 1
                continue
            k = 0
            while act[k]:
                k += 1
            act[k] = True; pd_[k] = d; pep[k] = ep; plot[k] = lots; psl[k] = a_sl[q]; ptp[k] = a_tp[q]
            pei[k] = i; pdl[k] = a_hold[q]; prisk[k] = lots * (dist * CONTRACT + comm); pid[k] = ntr; pgrp[k] = g
            day_trades += 1
            tr_ei[ntr] = i; tr_dir[ntr] = d; tr_ep[ntr] = ep; tr_lot[ntr] = lots; tr_risk[ntr] = prisk[k]
            tr_ord[ntr] = q
            ntr += 1
            q += 1
        # ---- hard daily guard ----
        any_act = False; sum_lot = 0.0; sum_eplot = 0.0; d_all = 0; worst_pnl = 0.0
        for k in range(K):
            if act[k]:
                any_act = True; d_all = pd_[k]; sum_lot += plot[k]; sum_eplot += pep[k] * plot[k]
                wp = (l[i] - pep[k]) if pd_[k] == 1 else (pep[k] - (h[i] + sp))
                worst_pnl += wp * plot[k] * CONTRACT - plot[k] * comm
        guard_hit = False; px = 0.0
        if any_act and (bal - day_ref) + worst_pnl <= -hard / 100.0 * day_ref:
            guard_hit = True
            target_open_pnl = -hard / 100.0 * day_ref - (bal - day_ref) + sum_lot * comm
            if d_all == 1:
                px = (target_open_pnl / CONTRACT + sum_eplot) / sum_lot
            else:
                px = (sum_eplot - target_open_pnl / CONTRACT) / sum_lot
        # ---- manage positions ----
        for k in range(K):
            if not act[k]:
                continue
            d = pd_[k]; ep = pep[k]; xp = 0.0; reason = 0
            fill_bar = pei[k] == i
            if guard_hit:
                xp = px; reason = 5
            elif d == 1:
                if o[i] <= psl[k] and not fill_bar:
                    xp = o[i] - slip; reason = 1
                elif l[i] <= psl[k]:
                    xp = psl[k] - slip; reason = 1
                elif (not fill_bar and h[i] >= ptp[k]) or (fill_bar and c[i] >= ptp[k]):
                    xp = ptp[k]; reason = 2
            else:
                if o[i] + sp >= psl[k] and not fill_bar:
                    xp = o[i] + sp + slip; reason = 1
                elif h[i] + sp >= psl[k]:
                    xp = psl[k] + slip; reason = 1
                elif (not fill_bar and l[i] + sp <= ptp[k]) or (fill_bar and c[i] + sp <= ptp[k]):
                    xp = ptp[k]; reason = 2
            if reason == 0:
                if i - pei[k] + 1 >= pdl[k]:
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
                pnl = (xp - ep) * d * plot[k] * CONTRACT - plot[k] * comm
                bal += pnl
                j = pid[k]
                tr_xi[j] = i; tr_xp[j] = xp; tr_pnl[j] = pnl; tr_reason[j] = reason
                act[k] = False
                if pnl > 0:
                    won_grp[pgrp[k]] = True
        if (init_bal - bal) / init_bal * 100.0 >= tstop:
            stopped = True
        fl = 0.0; fw = 0.0
        for k in range(K):
            if act[k]:
                if pd_[k] == 1:
                    fl += (c[i] - pep[k]) * plot[k] * CONTRACT; fw += (l[i] - pep[k]) * plot[k] * CONTRACT
                else:
                    fl += (pep[k] - (c[i] + sp)) * plot[k] * CONTRACT; fw += (pep[k] - (h[i] + sp)) * plot[k] * CONTRACT
        eq_close[i] = bal + fl
        eq_low[i] = min(bal + fw, bal + fl)
    return (tr_ei[:ntr], tr_xi[:ntr], tr_dir[:ntr], tr_ep[:ntr], tr_xp[:ntr], tr_lot[:ntr], tr_pnl[:ntr],
            tr_reason[:ntr], tr_risk[:ntr], tr_ord[:ntr], eq_close, eq_low)


def run_orders(exec_x: dict, orders: pd.DataFrame, guards: Guards = Guards(), costs: Costs = Costs()) -> Result:
    """orders columns: act_time, exp_time, dir, entry, sl, tp, hold_min, group (int), stop_on_win (bool), [leg]."""
    idx = exec_x["index"]
    iv = idx.values.astype("datetime64[ns]")
    od = orders.sort_values("act_time", kind="stable").reset_index(drop=True)
    a_idx = np.searchsorted(iv, od["act_time"].values.astype("datetime64[ns]"), side="left").astype(np.int64)
    x_idx = np.searchsorted(iv, od["exp_time"].values.astype("datetime64[ns]"), side="left").astype(np.int64) - 1
    bm = exec_x["bar_minutes"]
    hold = np.maximum(1, np.ceil(od["hold_min"].to_numpy(float) / bm)).astype(np.int64)
    grp = od["group"].to_numpy(np.int64) if "group" in od else np.arange(len(od), dtype=np.int64)
    grp = pd.factorize(grp)[0].astype(np.int64)
    sow = od["stop_on_win"].to_numpy(np.bool_) if "stop_on_win" in od else np.zeros(len(od), np.bool_)
    out = _run_limit(exec_x["t_min"], exec_x["dow"], exec_x["day_id"], exec_x["o"], exec_x["h"], exec_x["l"],
                     exec_x["c"], exec_x["spr"], exec_x["block"], exec_x["flat"],
                     a_idx, x_idx, od["dir"].to_numpy(np.int64), od["entry"].to_numpy(float),
                     od["sl"].to_numpy(float), od["tp"].to_numpy(float), hold, grp, sow,
                     guards.initial_balance, guards.risk_pct, guards.daily_soft_pct, guards.daily_hard_pct,
                     guards.total_derisk_pct, guards.total_stop_pct, guards.max_trades_day,
                     guards.first_entry_min, guards.last_entry_min, guards.flatten_min, guards.fri_flatten_min,
                     costs.commission_per_lot, costs.slippage_pts * POINT, guards.max_positions,
                     guards.max_open_risk_pct)
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, oi, eq, eql = out
    closed = xi >= 0
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, oi = (a[closed] for a in (ei, xi, dr, ep, xp, lot, pnl, rsn, risk, oi))
    trades = pd.DataFrame({"entry_time": idx[ei], "exit_time": idx[xi] + pd.Timedelta(minutes=bm) * (rsn >= 3),
                           "dir": dr, "entry": ep, "exit": xp, "lots": lot, "pnl": pnl, "risk_usd": risk,
                           "reason": [REASONS[r] for r in rsn]})
    if "leg" in od:
        trades["leg"] = od["leg"].to_numpy()[oi]
    trades["R"] = trades["pnl"] / trades["risk_usd"]
    trades = trades.sort_values("exit_time", kind="stable").reset_index(drop=True)
    equity = pd.Series(eq, index=idx, name="equity")
    day = idx.normalize()
    daily = pd.DataFrame({"end": equity.groupby(day).last(), "min": pd.Series(eql, index=idx).groupby(day).min()})
    daily["start"] = daily["end"].shift(1).fillna(guards.initial_balance)
    return Result(trades, equity, daily, {"guards": asdict(guards), "costs": asdict(costs)})
