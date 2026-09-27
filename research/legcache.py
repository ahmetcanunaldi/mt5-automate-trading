"""Cache of the book's leg signals (combo8.build is slow: ~5-10 min). Rebuild with --rebuild after changing a leg."""
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from research import lab  # noqa: E402

CACHE = lab.DATA / "legs_best.pkl"


def load(rebuild=False):
    if rebuild or not CACHE.exists():
        from research.combo8 import build
        m1, legs = build()
        CACHE.write_bytes(pickle.dumps(legs))
        return m1, legs
    return pd.read_parquet(lab.DATA / "XAUUSD_M1_2018.parquet"), pickle.loads(CACHE.read_bytes())


if __name__ == "__main__":
    load(rebuild="--rebuild" in sys.argv)
