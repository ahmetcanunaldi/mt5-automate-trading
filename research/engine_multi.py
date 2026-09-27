"""Multi-symbol event-driven engine: one account, shared FundingPips-style guards, positions in several symbols.

Same execution model as research/engine.py (bid bars, ask = bid + spread, SL-first, slippage, commission, swaps,
lot step per symbol, risk on initial or current balance, payouts + consistency, day profit cap, news masks with the
5-hour exemption). Differences:
- Bars of all symbols are aligned on the union M1 timeline; a symbol without a bar at t is "closed" (no entry,
  price carried forward, no exits).
- No hedging PER SYMBOL (a symbol's positions share one direction); different symbols are independent.
- Several signals may enter on the same bar (in signal order); per (symbol, bar) only the first signal is kept.
- Weekend flat closes each symbol at its own last bar before the weekend.
- Hard daily guard: when the worst-case day P&L breaches the limit, all positions are closed at the bar's worst
  price (conservative; the single-symbol engine solves the exact limit price).
"""
from dataclasses import asdict

import numpy as np
import pandas as pd
from numba import njit

from research.engine import REASONS, Costs, Guards, Result

MAXK = 16


def align(execs: dict):
    """execs: {sym: exec_x from engine.prepare_exec}. Returns dict of (S, n) arrays on the union timeline."""
    syms = list(execs)
    U = execs[syms[0]]["index"]
    for s in syms[1:]:
        U = U.union(execs[s]["index"])
    n, S = len(U), len(syms)
    out = {k: np.zeros((S, n)) for k in ("o", "h", "l", "c", "spr", "cv")}
    out["live"] = np.zeros((S, n), np.bool_)
    out["block"] = np.ones((S, n), np.bool_)
    out["flat"] = np.zeros((S, n), np.bool_)
    out["wkend"] = np.zeros((S, n), np.bool_)
    for j, s in enumerate(syms):
        x = execs[s]
        pos = U.get_indexer(x["index"])
        live = np.zeros(n, bool); live[pos] = True
        c = pd.Series(np.nan, index=U); c.iloc[pos] = x["c"]; c = c.ffill().bfill().to_numpy()
        for k in ("o", "h", "l"):
            a = np.array(c, copy=True); a[pos] = x[k]; out[k][j] = a
        out["c"][j] = c
        for k in ("spr", "cv"):
            a = pd.Series(np.nan, index=U); a.iloc[pos] = x[k]; out[k][j] = a.ffill().bfill().to_numpy()
        out["live"][j] = live
        b = np.ones(n, bool); b[pos] = x["block"]; out["block"][j] = b
        f = np.zeros(n, bool); f[pos] = x["flat"]; out["flat"][j] = f
        # last live bar of the symbol before a weekend (or before a gap of > 1 calendar day)
        di = x["index"].values.astype("datetime64[D]").astype(np.int64)
        last = np.r_[np.diff(di) > 1, True] & (x["index"].dayofweek.to_numpy() == 4)
        w = np.zeros(n, bool); w[pos[last]] = True; out["wkend"][j] = w
    out["t_min"] = (U.hour * 60 + U.minute).to_numpy(np.int64)
    out["dow"] = U.dayofweek.to_numpy(np.int64)
    out["day_id"] = U.values.astype("datetime64[D]").astype(np.int64)
    out["index"] = U; out["syms"] = syms
    out["sym_index"] = {s: execs[s]["index"] for s in syms}
    out["point"] = np.array([execs[s]["point"] for s in syms])
    out["vmin"] = np.array([execs[s].get("vmin", 0.01) for s in syms])
    out["vstep"] = np.array([execs[s].get("vstep", 0.01) for s in syms])
    return out


@njit(cache=True)
def _run_multi(t_min, dow, day_id, o, h, l, c, spr, cv, live, block, flat, wkend,
               s_idx, s_sym, s_dir, s_sl, s_tp, s_hold, s_trail, s_rm, s_flatm,
               init_bal, risk_pct, soft, hard, derisk, tstop, max_td, first_m, last_m, flat_m, fri_m,
               comm, slip, swap_l, swap_s, triple_dow, vmin, vstep,
               max_pos, max_pos_sym, max_open_risk, weekend_flat, risk_on_init, pay_pct, cons_pct, pcap, exempt_bars, cush):
    S, n = o.shape
    ns = len(s_idx)
    tr_ei = np.full(ns + 1, -1, np.int64); tr_xi = np.full(ns + 1, -1, np.int64)
    tr_dir = np.zeros(ns + 1, np.int64); tr_ep = np.zeros(ns + 1); tr_xp = np.zeros(ns + 1)
    tr_lot = np.zeros(ns + 1); tr_pnl = np.zeros(ns + 1); tr_reason = np.zeros(ns + 1, np.int64)
    tr_risk = np.zeros(ns + 1); tr_sig = np.full(ns + 1, -1, np.int64)
    eq_close = np.zeros(n); eq_low = np.zeros(n); bal_close = np.zeros(n)
    K = max_pos
    act = np.zeros(K, np.bool_); psym = np.zeros(K, np.int64); pd_ = np.zeros(K, np.int64); pep = np.zeros(K)
    plot = np.zeros(K); psl = np.zeros(K); ptp = np.zeros(K); pei = np.zeros(K, np.int64); pdl = np.zeros(K, np.int64)
    ptr = np.zeros(K); prisk = np.zeros(K); pid = np.zeros(K, np.int64); pswap = np.zeros(K)
    pintra = np.zeros(K, np.bool_)
    pay_i = np.full(n, -1, np.int64); pay_amt = np.zeros(n); reach_i = np.full(n, -1, np.int64); npay = 0
    bal = init_bal; day_bal0 = bal; cyc_best = 0.0; reached = -1; paid_now = False; capped = False
    cur_day = -1; day_ref = bal; day_trades = 0; stopped = False; ntr = 0; si = 0
    for i in range(n):
        if day_id[i] != cur_day:
            paid_now = False
            if i > 0 and pay_pct > 0:
                dp = bal - day_bal0
                if dp > cyc_best:
                    cyc_best = dp
                prof = bal - init_bal
                anyopen = False
                for k in range(K):
                    if act[k]:
                        anyopen = True
                if prof >= pay_pct / 100.0 * init_bal:
                    if reached < 0:
                        reached = i
                    if (not anyopen) and cyc_best <= cons_pct / 100.0 * prof:
                        pay_i[npay] = i; pay_amt[npay] = prof; reach_i[npay] = reached; npay += 1
                        bal = init_bal; cyc_best = 0.0; reached = -1; paid_now = True
            if i > 0:
                for k in range(K):
                    if act[k]:
                        j = psym[k]
                        nights = 3 if dow[i - 1] == triple_dow[j] else 1
                        sw = (swap_l[j] if pd_[k] == 1 else swap_s[j]) * plot[k] * nights
                        bal += sw; pswap[k] += sw
            cur_day = day_id[i]; day_trades = 0; day_bal0 = bal; capped = False
            day_ref = max(bal, eq_close[i - 1]) if (i > 0 and not paid_now) else bal
        # ---- entries ----
        while si < ns and s_idx[si] < i:
            si += 1
        while si < ns and s_idx[si] == i:
            j = s_sym[si]; d = s_dir[si]
            if stopped or capped or (not live[j, i]) or block[j, i]:
                si += 1
                continue
            nact = 0; nsym = 0; open_risk = 0.0; conflict = False
            for k in range(K):
                if act[k]:
                    nact += 1; open_risk += prisk[k]
                    if psym[k] == j:
                        nsym += 1
                        if pd_[k] != d:
                            conflict = True
            tm = t_min[i]
            ok = (tm >= first_m) and (tm <= last_m) and (day_trades < max_td) and (nact < K) and (nsym < max_pos_sym) and not conflict
            if s_flatm[si] and dow[i] == 4 and tm >= fri_m - 60:
                ok = False
            day_pnl = bal - day_ref
            if day_pnl <= -soft / 100.0 * day_ref:
                ok = False
            if ok:
                dd_tot = (init_bal - bal) / init_bal * 100.0
                if cush > 0:
                    rp = risk_pct * min(1.0, max(0.1, (tstop - dd_tot) / (tstop - cush)))
                else:
                    rp = risk_pct if dd_tot < derisk else risk_pct * 0.5
                room = hard / 100.0 * day_ref + day_pnl - open_risk
                base = init_bal if risk_on_init else bal
                risk_usd = min(rp * min(s_rm[si], 1.0) / 100.0 * base, room, max_open_risk / 100.0 * base - open_risk)
                dist = s_sl[si]
                lots = np.floor(risk_usd / (dist * cv[j, i] + comm[j]) / vstep[j] + 1e-9) * vstep[j]
                if lots >= vmin[j] - 1e-9:
                    k = 0
                    while act[k]:
                        k += 1
                    sp = spr[j, i]
                    if d == 1:
                        ep = o[j, i] + sp + slip[j]
                        psl[k] = ep - dist; ptp[k] = ep + s_tp[si]
                    else:
                        ep = o[j, i] - slip[j]
                        psl[k] = ep + dist; ptp[k] = ep - s_tp[si]
                    act[k] = True; psym[k] = j; pd_[k] = d; pep[k] = ep; plot[k] = lots; pei[k] = i
                    pdl[k] = s_hold[si]; ptr[k] = s_trail[si]; prisk[k] = lots * (dist * cv[j, i] + comm[j])
                    pid[k] = ntr; pswap[k] = 0.0; pintra[k] = s_flatm[si]
                    day_trades += 1
                    tr_ei[ntr] = i; tr_dir[ntr] = d; tr_ep[ntr] = ep; tr_lot[ntr] = lots
                    tr_risk[ntr] = prisk[k]; tr_sig[ntr] = si
                    ntr += 1
            si += 1
        # ---- daily hard guard on worst prices ----
        any_act = False; worst_pnl = 0.0; fl_c = 0.0
        for k in range(K):
            if act[k]:
                any_act = True
                j = psym[k]; sp = spr[j, i]
                if pd_[k] == 1:
                    worst_pnl += (l[j, i] - pep[k]) * plot[k] * cv[j, i] - plot[k] * comm[j]
                    fl_c += (c[j, i] - pep[k]) * plot[k] * cv[j, i]
                else:
                    worst_pnl += (pep[k] - (h[j, i] + sp)) * plot[k] * cv[j, i] - plot[k] * comm[j]
                    fl_c += (pep[k] - (c[j, i] + sp)) * plot[k] * cv[j, i]
        guard_hit = any_act and (bal - day_ref) + worst_pnl <= -hard / 100.0 * day_ref
        cap_hit = False
        if pcap > 0 and any_act and (bal - day_ref) + fl_c >= pcap / 100.0 * day_ref:
            cap_hit = True; capped = True
        # ---- manage positions ----
        for k in range(K):
            if not act[k]:
                continue
            j = psym[k]; d = pd_[k]; ep = pep[k]; sp = spr[j, i]
            xp = 0.0; reason = 0
            if guard_hit:
                xp = l[j, i] if d == 1 else h[j, i] + sp; reason = 5
            elif cap_hit:
                xp = c[j, i] if d == 1 else c[j, i] + sp; reason = 7
            elif live[j, i]:
                slp = psl[k]; tpp = ptp[k]
                if d == 1:
                    if o[j, i] <= slp and i > pei[k]:
                        xp = o[j, i] - slip[j]; reason = 1
                    elif l[j, i] <= slp:
                        xp = slp - slip[j]; reason = 1
                    elif h[j, i] >= tpp:
                        xp = tpp; reason = 2
                else:
                    if o[j, i] + sp >= slp and i > pei[k]:
                        xp = o[j, i] + sp + slip[j]; reason = 1
                    elif h[j, i] + sp >= slp:
                        xp = slp + slip[j]; reason = 1
                    elif l[j, i] + sp <= tpp:
                        xp = tpp; reason = 2
                if reason == 0 and ptr[k] > 0:
                    if d == 1:
                        if h[j, i] - ptr[k] > psl[k]:
                            psl[k] = h[j, i] - ptr[k]
                    else:
                        if l[j, i] + sp + ptr[k] < psl[k]:
                            psl[k] = l[j, i] + sp + ptr[k]
                if reason == 0:
                    if i - pei[k] + 1 >= pdl[k]:
                        reason = 3
                    elif flat[j, i] and (exempt_bars <= 0 or i - pei[k] < exempt_bars):
                        reason = 4
                    elif i + 1 >= n:
                        reason = 6
                    elif pintra[k] and (t_min[i] >= flat_m or (dow[i] == 4 and t_min[i] >= fri_m) or day_id[i + 1] != day_id[i]):
                        reason = 6
                    elif weekend_flat and wkend[j, i]:
                        reason = 6
                    if reason != 0:
                        xp = c[j, i] if d == 1 else c[j, i] + sp
            elif i + 1 >= n:
                reason = 6; xp = c[j, i] if d == 1 else c[j, i] + sp
            if reason != 0:
                pnl = (xp - ep) * d * plot[k] * cv[j, i] - plot[k] * comm[j]
                bal += pnl
                q = pid[k]
                tr_xi[q] = i; tr_xp[q] = xp; tr_pnl[q] = pnl + pswap[k]; tr_reason[q] = reason
                act[k] = False
        if (init_bal - bal) / init_bal * 100.0 >= tstop:
            stopped = True
        fl = 0.0; fw = 0.0
        for k in range(K):
            if act[k]:
                j = psym[k]; sp = spr[j, i]
                if pd_[k] == 1:
                    fl += (c[j, i] - pep[k]) * plot[k] * cv[j, i]; fw += (l[j, i] - pep[k]) * plot[k] * cv[j, i]
                else:
                    fl += (pep[k] - (c[j, i] + sp)) * plot[k] * cv[j, i]; fw += (pep[k] - (h[j, i] + sp)) * plot[k] * cv[j, i]
        eq_close[i] = bal + fl; bal_close[i] = bal; eq_low[i] = min(bal + fw, bal + fl)
    return (tr_ei[:ntr], tr_xi[:ntr], tr_dir[:ntr], tr_ep[:ntr], tr_xp[:ntr], tr_lot[:ntr], tr_pnl[:ntr],
            tr_reason[:ntr], tr_risk[:ntr], tr_sig[:ntr], eq_close, eq_low, bal_close, pay_i[:npay], pay_amt[:npay],
            reach_i[:npay])


def run(ax: dict, signals: pd.DataFrame, costs: dict, guards: Guards = Guards(), max_pos_sym: int = 6,
        intraday_legs=()) -> Result:
    """ax: output of align(); signals: DataFrame with column `sym` (symbol name) + the usual columns
    (dir, sl, tp, hold_min, trail, risk_mult, leg). costs: {sym: engine.Costs}. Legs listed in intraday_legs are
    flattened at guards.flatten_min / Friday fri_flatten_min; all others follow their hold and weekend flat."""
    assert 1 <= guards.max_positions <= MAXK
    syms = ax["syms"]; U = ax["index"]
    sig = signals[signals["sym"].isin(syms)].sort_index(kind="stable")
    # each signal enters at the first bar of ITS symbol at/after the decision time
    s_idx = np.full(len(sig), len(U), np.int64)
    for s in syms:
        m = (sig["sym"] == s).to_numpy()
        si_ = ax["sym_index"][s]
        p = np.searchsorted(si_.values.astype("datetime64[ns]"), sig.index.values[m].astype("datetime64[ns]"), side="left")
        ok = p < len(si_)
        u = np.full(m.sum(), len(U), np.int64)
        u[ok] = U.get_indexer(si_[p[ok]])
        s_idx[m] = u
    keep = s_idx < len(U)
    sig = sig[keep]; s_idx = s_idx[keep]
    s_sym = sig["sym"].map({s: j for j, s in enumerate(syms)}).to_numpy(np.int64)
    key = pd.Series(s_idx * 64 + s_sym)
    first = ~key.duplicated().to_numpy()
    sig = sig[first]; s_idx = s_idx[first]; s_sym = s_sym[first]
    o = np.lexsort((np.arange(len(s_idx)), s_idx))
    sig = sig.iloc[o]; s_idx = s_idx[o]; s_sym = s_sym[o]
    s_hold = np.maximum(1, np.ceil(sig["hold_min"].to_numpy(float))).astype(np.int64)
    trail = sig["trail"].fillna(0.0).to_numpy(float) if "trail" in sig else np.zeros(len(sig))
    rm = sig["risk_mult"].fillna(1.0).to_numpy(float) if "risk_mult" in sig else np.ones(len(sig))
    flatm = sig["leg"].isin(intraday_legs).to_numpy() if "leg" in sig else np.zeros(len(sig), bool)
    C = [costs[s] for s in syms]
    arr = lambda f: np.array([f(c_) for c_ in C], dtype=float)  # noqa: E731
    out = _run_multi(ax["t_min"], ax["dow"], ax["day_id"], ax["o"], ax["h"], ax["l"], ax["c"], ax["spr"], ax["cv"],
                     ax["live"], ax["block"], ax["flat"], ax["wkend"],
                     s_idx, s_sym, sig["dir"].to_numpy(np.int64), sig["sl"].to_numpy(float), sig["tp"].to_numpy(float),
                     s_hold, trail, rm, flatm,
                     guards.initial_balance, guards.risk_pct, guards.daily_soft_pct, guards.daily_hard_pct,
                     guards.total_derisk_pct, guards.total_stop_pct, guards.max_trades_day,
                     guards.first_entry_min, guards.last_entry_min, guards.flatten_min, guards.fri_flatten_min,
                     arr(lambda c_: c_.commission_per_lot), arr(lambda c_: c_.slippage_pts) * ax["point"],
                     arr(lambda c_: c_.swap_long), arr(lambda c_: c_.swap_short),
                     np.array([c_.triple_dow for c_ in C], np.int64), ax["vmin"], ax["vstep"],
                     guards.max_positions, max_pos_sym, guards.max_open_risk_pct, guards.weekend_flat,
                     guards.risk_on_initial, guards.payout_pct, guards.consistency_pct, guards.day_profit_cap_pct,
                     int(guards.news_exempt_min), guards.cushion_start_pct)
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi, eq, eql, balc, pay_i, pay_amt, reach_i = out
    closed = xi >= 0
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi = (a[closed] for a in (ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi))
    trades = pd.DataFrame({"entry_time": U[ei], "exit_time": U[xi] + pd.Timedelta(minutes=1) * (rsn >= 3),
                           "sym": sig["sym"].to_numpy()[sgi], "dir": dr, "entry": ep, "exit": xp, "lots": lot,
                           "pnl": pnl, "risk_usd": risk, "reason": [REASONS[r] for r in rsn]})
    if "leg" in sig:
        trades["leg"] = sig["leg"].to_numpy()[sgi]
    trades["R"] = trades["pnl"] / trades["risk_usd"]
    trades = trades.sort_values("exit_time", kind="stable").reset_index(drop=True)
    equity = pd.Series(eq, index=U, name="equity")
    day = U.normalize()
    daily = pd.DataFrame({"end": equity.groupby(day).last(), "min": pd.Series(eql, index=U).groupby(day).min()})
    daily["bal_end"] = pd.Series(balc, index=U).groupby(day).last()
    daily["start"] = np.maximum(daily["end"].shift(1), daily["bal_end"].shift(1)).fillna(guards.initial_balance)
    daily["start_eq"] = daily["end"].shift(1).fillna(guards.initial_balance)
    payouts = pd.DataFrame({"time": U[pay_i], "amount": pay_amt, "reached": U[reach_i]})
    return Result(trades, equity, daily, {"guards": asdict(guards), "costs": {s: asdict(costs[s]) for s in syms},
                                          "payouts": payouts})
