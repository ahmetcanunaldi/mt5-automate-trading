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
MAXK = 12


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
    risk_on_initial: bool = False      # size from the initial balance (funded account with payouts back to initial)
    payout_pct: float = 0.0            # >0: withdraw profit when cycle profit >= payout_pct % (balance back to initial)
    consistency_pct: float = 100.0     # payout only if the best day <= consistency_pct % of the cycle profit
    day_profit_cap_pct: float = 0.0    # >0: when day P&L (incl. floating) >= cap, close all and stop for the day
    news_reentry: bool = False         # re-open a position closed by the news flatten once the blackout ends
    cap_dynamic: bool = False          # day cap = max(day_profit_cap_pct, cons/(100-cons) * cycle profit so far)
    news_exempt_min: int = 0           # >0: positions older than this (minutes) at the news flatten bar are kept
                                       # (FundingPips: trades opened >= 5 h before a news event are exempt)


@njit(cache=True)
def _run(t_min, dow, day_id, o, h, l, c, spr, cv, block, flat,
         s_idx, s_dir, s_sl, s_tp, s_hold, s_be, s_trail, s_rm,
         init_bal, risk_pct, soft, hard, derisk, tstop, max_td, first_m, last_m, flat_m, fri_m,
         comm, slip, max_pos, max_open_risk, intraday, weekend_flat, swap_l, swap_s, triple_dow, risk_on_init, pay_pct, cons_pct, pcap, reent, capdyn, vmin, vstep, exempt_bars):
    n = len(o)
    ns = len(s_idx)
    max_tr = (4 * ns + 1) if reent else (ns + 1)
    tr_ei = np.full(max_tr, -1, np.int64); tr_xi = np.full(max_tr, -1, np.int64)
    tr_dir = np.zeros(max_tr, np.int64); tr_ep = np.zeros(max_tr); tr_xp = np.zeros(max_tr)
    tr_lot = np.zeros(max_tr); tr_pnl = np.zeros(max_tr); tr_reason = np.zeros(max_tr, np.int64)
    tr_risk = np.zeros(max_tr); tr_sig = np.full(max_tr, -1, np.int64)
    eq_close = np.zeros(n); eq_low = np.zeros(n); bal_close = np.zeros(n)
    K = max_pos
    act = np.zeros(K, np.bool_); pd_ = np.zeros(K, np.int64); pep = np.zeros(K); plot = np.zeros(K)
    psl = np.zeros(K); ptp = np.zeros(K); pei = np.zeros(K, np.int64); pdl = np.zeros(K, np.int64)
    pbe = np.zeros(K); pbed = np.zeros(K, np.bool_); ptr = np.zeros(K); prisk = np.zeros(K)
    pid = np.zeros(K, np.int64); pswap = np.zeros(K)
    pdist = np.zeros(K); ptpd = np.zeros(K); prm = np.ones(K); psig = np.zeros(K, np.int64)
    # re-entry queue (positions closed by the news flatten)
    q_act = np.zeros(K, np.bool_); q_dir = np.zeros(K, np.int64); q_dist = np.zeros(K); q_tpd = np.zeros(K)
    q_exp = np.zeros(K, np.int64); q_be = np.zeros(K); q_tr = np.zeros(K); q_rm = np.ones(K)
    q_sig = np.zeros(K, np.int64); q_day = np.zeros(K, np.int64)
    pay_i = np.full(n, -1, np.int64); pay_amt = np.zeros(n); reach_i = np.full(n, -1, np.int64); npay = 0
    day_bal0 = init_bal; cyc_best = 0.0; reached = -1; paid_now = False; capped = False
    ntr = 0          # trades opened (index into tr_*)
    bal = init_bal
    cur_day = -1; day_ref = bal; day_trades = 0; stopped = False
    si = 0
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
            if i > 0 and not intraday:
                nights = 3 if dow[i - 1] == triple_dow else 1
                for k in range(K):
                    if act[k]:
                        sw = (swap_l if pd_[k] == 1 else swap_s) * plot[k] * nights
                        bal += sw; pswap[k] += sw
            cur_day = day_id[i]; day_trades = 0; day_bal0 = bal; capped = False
            day_ref = max(bal, eq_close[i - 1]) if (i > 0 and not paid_now) else bal
        sp = spr[i]
        # ---- entry at bar open: pending news re-entries first, then the bar's signal ----
        while si < ns and s_idx[si] < i:
            si += 1
        for cand in range(K + 1):
            if cand < K:
                if not q_act[cand]:
                    continue
                if i >= q_exp[cand] or (intraday and day_id[i] != q_day[cand]) or day_id[i] - q_day[cand] > 3:
                    q_act[cand] = False
                    continue
                if block[i] or flat[i]:
                    continue
                q_act[cand] = False                       # one attempt once the blackout is over
                d = q_dir[cand]; dist = q_dist[cand]; tpd = q_tpd[cand]; hold = q_exp[cand] - i
                be_ = q_be[cand]; tr_ = q_tr[cand]; rm_ = q_rm[cand]; sg = q_sig[cand]
            else:
                if not (si < ns and s_idx[si] == i):
                    continue
                d = s_dir[si]; dist = s_sl[si]; tpd = s_tp[si]; hold = s_hold[si]
                be_ = s_be[si]; tr_ = s_trail[si]; rm_ = s_rm[si]; sg = si
            if stopped or capped:
                continue
            nact = 0; cur_dir = 0; open_risk = 0.0
            for k in range(K):
                if act[k]:
                    nact += 1; cur_dir = pd_[k]; open_risk += prisk[k]
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
                base = init_bal if risk_on_init else bal
                risk_usd = min(rp * min(rm_, 1.0) / 100.0 * base, room, max_open_risk / 100.0 * base - open_risk)
                lots = np.floor(risk_usd / (dist * cv[i] + comm) / vstep + 1e-9) * vstep
                if lots >= vmin - 1e-9:
                    k = 0
                    while act[k]:
                        k += 1
                    if d == 1:
                        ep = o[i] + sp + slip
                        psl[k] = ep - dist; ptp[k] = ep + tpd
                    else:
                        ep = o[i] - slip
                        psl[k] = ep + dist; ptp[k] = ep - tpd
                    act[k] = True; pd_[k] = d; pep[k] = ep; plot[k] = lots; pei[k] = i; pdl[k] = hold
                    pbe[k] = be_; pbed[k] = False; ptr[k] = tr_
                    prisk[k] = lots * (dist * cv[i] + comm); pid[k] = ntr; pswap[k] = 0.0
                    pdist[k] = dist; ptpd[k] = tpd; prm[k] = rm_; psig[k] = sg
                    day_trades += 1
                    tr_ei[ntr] = i; tr_dir[ntr] = d; tr_ep[ntr] = ep; tr_lot[ntr] = lots
                    tr_risk[ntr] = prisk[k]; tr_sig[ntr] = sg
                    ntr += 1
        # ---- hard daily guard on the bar's worst price (all positions share one direction) ----
        any_act = False; sum_lot = 0.0; sum_eplot = 0.0; d_all = 0; worst_pnl = 0.0
        for k in range(K):
            if act[k]:
                any_act = True; d_all = pd_[k]
                sum_lot += plot[k]; sum_eplot += pep[k] * plot[k]
                wp = (l[i] - pep[k]) if pd_[k] == 1 else (pep[k] - (h[i] + sp))
                worst_pnl += wp * plot[k] * cv[i] - plot[k] * comm
        guard_hit = False; cap_hit = False
        if pcap > 0 and any_act:
            fl_c = 0.0
            for k in range(K):
                if act[k]:
                    fl_c += ((c[i] - pep[k]) if pd_[k] == 1 else (pep[k] - (c[i] + sp))) * plot[k] * cv[i]
            cap_usd = pcap / 100.0 * day_ref
            if capdyn:
                cap_usd = max(cap_usd, cons_pct / (100.0 - cons_pct) * (day_bal0 - init_bal))
            if (bal - day_ref) + fl_c >= cap_usd:
                cap_hit = True; capped = True
        if any_act and (bal - day_ref) + worst_pnl <= -hard / 100.0 * day_ref:
            guard_hit = True
            # price where total day P&L == -hard (solve linear equation for the common exit price)
            target_open_pnl = -hard / 100.0 * day_ref - (bal - day_ref) + sum_lot * comm
            if d_all == 1:
                px = (target_open_pnl / cv[i] + sum_eplot) / sum_lot
            else:
                px = (sum_eplot - target_open_pnl / cv[i]) / sum_lot
        # ---- manage each position ----
        for k in range(K):
            if not act[k]:
                continue
            d = pd_[k]; ep = pep[k]; slp = psl[k]; tpp = ptp[k]
            xp = 0.0; reason = 0
            if guard_hit:
                xp = px; reason = 5
            elif cap_hit:
                xp = c[i] if d == 1 else c[i] + sp; reason = 7
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
                elif flat[i] and (exempt_bars <= 0 or i - pei[k] < exempt_bars):
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
                pnl = (xp - ep) * d * plot[k] * cv[i] - plot[k] * comm
                bal += pnl
                j = pid[k]
                tr_xi[j] = i; tr_xp[j] = xp; tr_pnl[j] = pnl + pswap[k]; tr_reason[j] = reason
                act[k] = False
                if reent and reason == 4 and pei[k] + pdl[k] > i + 1 and ntr + K < max_tr:
                    q = 0
                    while q < K and q_act[q]:
                        q += 1
                    if q < K:
                        q_act[q] = True; q_dir[q] = d; q_dist[q] = pdist[k]; q_tpd[q] = ptpd[k]
                        q_exp[q] = pei[k] + pdl[k]; q_be[q] = pbe[k]; q_tr[q] = ptr[k]; q_rm[q] = prm[k]
                        q_sig[q] = psig[k]; q_day[q] = day_id[i]
        if (init_bal - bal) / init_bal * 100.0 >= tstop:
            stopped = True
        # equity at bar close and at the bar's worst price
        fl = 0.0; fw = 0.0
        for k in range(K):
            if act[k]:
                if pd_[k] == 1:
                    fl += (c[i] - pep[k]) * plot[k] * cv[i]
                    fw += (l[i] - pep[k]) * plot[k] * cv[i]
                else:
                    fl += (pep[k] - (c[i] + sp)) * plot[k] * cv[i]
                    fw += (pep[k] - (h[i] + sp)) * plot[k] * cv[i]
        eq_close[i] = bal + fl
        bal_close[i] = bal
        eq_low[i] = min(bal + fw, bal + fl)
    return (tr_ei[:ntr], tr_xi[:ntr], tr_dir[:ntr], tr_ep[:ntr], tr_xp[:ntr], tr_lot[:ntr],
            tr_pnl[:ntr], tr_reason[:ntr], tr_risk[:ntr], tr_sig[:ntr], eq_close, eq_low, bal_close,
            pay_i[:npay], pay_amt[:npay], reach_i[:npay])


REASONS = {1: "SL", 2: "TP", 3: "TIME", 4: "NEWS", 5: "DAILY_GUARD", 6: "EOD", 7: "PROFIT_CAP"}


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series          # equity at each execution bar close
    daily: pd.DataFrame        # per server day: start, end, min (intrabar worst) equity
    params: dict = field(default_factory=dict)


def prepare_exec(bars: pd.DataFrame, bar_minutes: int, news_block=None, news_flat=None, costs: Costs = Costs(),
                 point: float = POINT, contract: float = CONTRACT, quote: str = "USD", fx=None,
                 vmin: float = 0.01, vstep: float = 0.01):
    """point / contract / quote describe the symbol (default XAUUSD). quote="JPY": P&L converted to USD at the
    bar's close (USD per 1.0 price unit per lot = contract / close)."""
    idx = bars.index
    spr_pts = np.maximum(bars["spread"].to_numpy(float) * costs.spread_mult, costs.min_spread_pts)
    if quote == "USD":
        cv = np.full(len(bars), contract)
    elif fx is not None:                       # e.g. EUR-quoted index: USD per EUR (EURUSD close, as-of)
        cv = contract * fx.reindex(idx, method="ffill").bfill().to_numpy(float)
    else:                                      # USD base, quote currency = price (USDJPY)
        cv = contract / bars["close"].to_numpy(float)
    x = {
        "t_min": (idx.hour * 60 + idx.minute).to_numpy(np.int64),
        "dow": idx.dayofweek.to_numpy(np.int64),
        "day_id": idx.values.astype("datetime64[D]").astype(np.int64),
        "o": bars["open"].to_numpy(float), "h": bars["high"].to_numpy(float),
        "l": bars["low"].to_numpy(float), "c": bars["close"].to_numpy(float),
        "spr": spr_pts * point, "cv": cv, "point": point, "vmin": vmin, "vstep": vstep,
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
    rm = sig["risk_mult"].fillna(1.0).to_numpy(float) if "risk_mult" in sig else np.ones(len(sig))
    out = _run(exec_x["t_min"], exec_x["dow"], exec_x["day_id"], exec_x["o"], exec_x["h"], exec_x["l"],
               exec_x["c"], exec_x["spr"], exec_x["cv"], exec_x["block"], exec_x["flat"],
               s_idx, sig["dir"].to_numpy(np.int64), sig["sl"].to_numpy(float), sig["tp"].to_numpy(float),
               s_hold, be, trail, rm,
               guards.initial_balance, guards.risk_pct, guards.daily_soft_pct, guards.daily_hard_pct,
               guards.total_derisk_pct, guards.total_stop_pct, guards.max_trades_day,
               guards.first_entry_min, guards.last_entry_min, guards.flatten_min, guards.fri_flatten_min,
               costs.commission_per_lot, costs.slippage_pts * exec_x.get("point", POINT), guards.max_positions,
               guards.max_open_risk_pct, guards.intraday, guards.weekend_flat, costs.swap_long, costs.swap_short,
               costs.triple_dow, guards.risk_on_initial, guards.payout_pct, guards.consistency_pct,
               guards.day_profit_cap_pct, guards.news_reentry, guards.cap_dynamic,
               exec_x.get("vmin", 0.01), exec_x.get("vstep", 0.01),
               int(guards.news_exempt_min // exec_x["bar_minutes"]))
    ei, xi, dr, ep, xp, lot, pnl, rsn, risk, sgi, eq, eql, balc, pay_i, pay_amt, reach_i = out
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
    daily["bal_end"] = pd.Series(balc, index=idx).groupby(day).last()
    # FundingPips daily reference: max(balance, equity) at the start of the server day
    daily["start"] = np.maximum(daily["end"].shift(1), daily["bal_end"].shift(1)).fillna(guards.initial_balance)
    daily["start_eq"] = daily["end"].shift(1).fillna(guards.initial_balance)   # for returns / Sharpe
    payouts = pd.DataFrame({"time": idx[pay_i], "amount": pay_amt, "reached": idx[reach_i]})
    return Result(trades, equity, daily, {"guards": asdict(guards), "costs": asdict(costs), "payouts": payouts})
