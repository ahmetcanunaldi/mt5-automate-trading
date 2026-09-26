"""Refresh the USD high-impact news list for the EA.

1. pulls calendar values from the MT5 economic calendar via MCP -> data/calendar_usd_high.csv
2. converts fixed-UTC+3 calendar times to trade-server time (UTC+2/+3, US DST)
3. writes Common/Files/xau_news_server.csv ("YYYY.MM.DD HH:MI" per line) via MCP
Run weekly for live trading (the EA refuses to start without the file).
"""
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mcp_client import MT5MCP, ROOT  # noqa: E402
from research.calendar_news import load_news_server_times  # noqa: E402

COMMON = pathlib.Path(r"C:\Users\ahmet\AppData\Roaming\MetaQuotes\Terminal\Common\Files")


def pull(c, start="2017-12-01", end="2027-12-31"):
    ev = c.call("economic_calendar_list_events_by_country", country_code="US")["events"]
    rows = []
    for e in ev:
        if e["importance"] != "High importance":
            continue
        v = c.call("economic_calendar_list_values", datetime_from=start + "T00:00:00", datetime_to=end + "T00:00:00",
                   event_id=e["id"], limit=5000)
        for x in v.get("values", []) if isinstance(v, dict) else []:
            rows.append((x["time"], e["event_code"], e["id"]))
    rows = sorted(set(rows))
    with open(ROOT / "data" / "calendar_usd_high.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_server", "event_code", "event_id"])
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    c = MT5MCP()
    if "--no-pull" not in sys.argv:
        print("calendar rows:", pull(c))
    news = load_news_server_times()
    text = "\n".join(t.strftime("%Y.%m.%d %H:%M") for t in news) + "\n"
    c.call("write_file", path=str(COMMON / "xau_news_server.csv"), content=text, overwrite=True)
    print("news events written:", len(news), news[0], "->", news[-1])
