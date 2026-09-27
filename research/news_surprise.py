"""EXP-060: news surprise direction. Pull actual/forecast for US tier-1 releases from the MT5 calendar (MCP),
define the USD-implied gold direction of the surprise (USD-positive surprise -> gold down), then at T+30 min:
  agree   : the 0->30 min gold move has the surprise-implied sign  -> trade continuation
  disagree: the move is against the surprise                        -> trade reversal (towards the surprise sign)
Measure T+30 -> T+30+h returns (bps) and per-year consistency, M1 2019-2026."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research import calendar_news, lab  # noqa: E402

# +1: higher-than-forecast is USD-POSITIVE (gold-negative); -1: higher is USD-negative
EVENTS = {"nonfarm-payrolls": 1, "consumer-price-index-mm": 1, "consumer-price-index-ex-food-energy-mm": 1,
          "core-pce-price-index-mm": 1, "retail-sales-mm": 1, "ism-manufacturing-pmi": 1, "ism-non-manufacturing-pmi": 1,
          "producer-price-index-mm": 1, "average-hourly-earnings-mm": 1, "gross-domestic-product-qq": 1,
          "unemployment-rate": -1, "initial-jobless-claims": -1, "fed-interest-rate-decision": 1,
          "jolts-job-openings": 1, "adp-nonfarm-employment-change": 1}
CACHE = lab.DATA / "calendar_values.csv"


def pull():
    from mcp_client import MT5MCP
    c = MT5MCP()
    ev = c.call("economic_calendar_list_events_by_country", country_code="US")["events"]
    rows = []
    for e in ev:
        if e["event_code"] not in EVENTS:
            continue
        v = c.call("economic_calendar_list_values", datetime_from="2018-12-01T00:00:00",
                   datetime_to="2026-09-26T00:00:00", event_id=e["id"], limit=5000)
        for x in v.get("values", []):
            rows.append({"time_cal": x["time"], "code": e["event_code"], "actual": x.get("actual_value"),
                         "forecast": x.get("forecast_value"), "prev": x.get("prev_value")})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv(CACHE) if CACHE.exists() else pull()
    df = df.dropna(subset=["actual", "forecast"])
    df["t"] = calendar_news.utc_to_server(pd.DatetimeIndex(pd.to_datetime(df.time_cal) - pd.Timedelta(hours=3)))
    df["gold_dir"] = -np.sign(df.actual - df.forecast) * df.code.map(EVENTS)     # surprise-implied gold direction
    df = df[df.gold_dir != 0]
    # one row per release time (combine simultaneous releases by majority)
    agg = df.groupby("t").gold_dir.sum().apply(np.sign)
    agg = agg[agg != 0]
    m1 = pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet").loc["2019":]
    c = m1["close"]
    T = agg.index
    p0 = c.asof(T - pd.Timedelta(minutes=1)).to_numpy(); p30 = c.asof(T + pd.Timedelta(minutes=30)).to_numpy()
    r30 = np.log(p30 / p0) * 1e4
    agree = np.sign(r30) == agg.to_numpy()
    print(f"releases with a surprise: {len(agg)}; initial reaction agrees with surprise: {agree.mean() * 100:.0f} %")
    for h in (30, 90, 240):
        ph = c.asof(T + pd.Timedelta(minutes=30 + h)).to_numpy()
        rh = pd.Series(np.log(ph / p30) * 1e4, index=T)
        for name, msk, sign in (("agree -> continue", agree, np.sign(r30)), ("disagree -> towards surprise", ~agree, agg.to_numpy()),
                                ("disagree -> continue reaction", ~agree, np.sign(r30)),
                                ("all -> towards surprise", np.ones(len(T), bool), agg.to_numpy())):
            v = (rh * sign)[msk].replace([np.inf, -np.inf], np.nan).dropna()
            big = v[np.abs(r30[msk][: len(v)]) > 0]  # placeholder to keep shapes aligned
            yr = v.groupby(v.index.year).mean()
            print(f"h+{h:>3}m {name:<32} n {len(v):>4} {v.mean():+6.2f} bps t {v.mean() / v.std() * np.sqrt(len(v)):+.2f} "
                  f"yrs+ {(yr > 0).sum()}/{len(yr)}")
