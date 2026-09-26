"""EXP-002: statistical edge discovery on XAUUSD (H1 2010-2026, M15 2022-2026)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab  # noqa: E402

pd.set_option("display.width", 200)
h1 = lab.load("H1").loc["2010":]
r = np.log(h1["close"]).diff() * 1e4  # bps
df = pd.DataFrame({"r": r, "h": h1.index.hour, "y": h1.index.year, "dow": h1.index.dayofweek}).dropna()

print("=== A. hour-of-day drift (bps), 2010-2026, t-stat, share of years with same sign ===")
g = df.groupby("h")["r"]
tab = pd.DataFrame({"mean": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count())})
yr = df.groupby(["h", "y"])["r"].mean().unstack()
tab["same_sign_years"] = (np.sign(yr).eq(np.sign(tab["mean"]), axis=0)).mean(axis=1)
tab["mean_2022+"] = df[df.y >= 2022].groupby("h")["r"].mean()
tab["t_2022+"] = df[df.y >= 2022].groupby("h")["r"].apply(lambda x: x.mean() / x.std() * np.sqrt(len(x)))
print(tab.round(2).to_string())

print("\n=== B. session-to-session predictability (sign agreement / corr) 2015-2026 ===")
d = h1.loc["2015":].copy()
d["date"] = d.index.normalize()
def seg(a, b):
    x = d[(d.index.hour >= a) & (d.index.hour < b)]
    return (np.log(x.groupby("date")["close"].last()) - np.log(x.groupby("date")["open"].first())) * 1e4
asia, lon, ny1, ny2 = seg(1, 10), seg(10, 15), seg(15, 19), seg(19, 23)
S = pd.DataFrame({"asia": asia, "london": lon, "ny_am": ny1, "ny_pm": ny2}).dropna()
print(S.corr().round(3).to_string())
for a, b in [("asia", "london"), ("london", "ny_am"), ("asia", "ny_am"), ("ny_am", "ny_pm")]:
    x = S[S[a].abs() > S[a].abs().quantile(0.7)]
    print(f"{a}->{b} (top30% |{a}|): P(same sign)={np.mean(np.sign(x[a]) == np.sign(x[b])):.3f} n={len(x)} "
          f"mean {b}*sign({a})={np.mean(np.sign(x[a]) * x[b]):.1f}bps")

print("\n=== C. post-news drift (M15, 2022-07..): move in [0,30m] vs [30m,150m] after USD high-impact ===")
m15 = lab.load("M15")
news = calendar_news.load_news_server_times()
news = news[(news >= m15.index[0]) & (news <= m15.index[-1])]
news = pd.DatetimeIndex(sorted(set(news.floor("15min"))))
c = m15["close"]
def px(t):
    i = c.index.searchsorted(t)
    return c.iloc[i - 1] if 0 < i <= len(c) else np.nan
rows = []
for t in news:
    p0, p30, p150 = px(t), px(t + pd.Timedelta("30min")), px(t + pd.Timedelta("150min"))
    rows.append((t, p30 - p0, p150 - p30))
N = pd.DataFrame(rows, columns=["t", "m0_30", "m30_150"]).dropna()
for q in (0, 0.5, 0.75):
    x = N[N.m0_30.abs() >= N.m0_30.abs().quantile(q)]
    print(f"|move0-30| >= q{q}: n={len(x)} P(continuation)={np.mean(np.sign(x.m0_30)==np.sign(x.m30_150)):.3f} "
          f"mean cont $={np.mean(np.sign(x.m0_30)*x.m30_150):.2f} median |m0_30|=${x.m0_30.abs().median():.1f}")

print("\n=== D. day-of-week (bps/day) 2010-2026 and 2022+ ===")
dd = df.groupby([df.index.normalize(), "dow"])["r"].sum().reset_index()
print(dd.groupby("dow")["r"].agg(["mean", "count"]).round(1).T.to_string())

print("\n=== E. M15 return autocorrelation by session (2022-07..), lag1 ===")
m = m15.copy(); m["r"] = np.log(m.close).diff(); m["r1"] = m["r"].shift(1); m["h"] = m.index.hour
print(m.groupby("h").apply(lambda x: x["r"].corr(x["r1"])).round(3).to_dict())
