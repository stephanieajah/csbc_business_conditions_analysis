"""Ingest a new quarterly CSBC file and refresh the national trend chart.

Usage:
    python src/ingest.py "data/raw/Data CSBC-Q3 2024.csv"
    python src/ingest.py            # rebuild from everything in data/raw/
"""
import sys
import glob
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "combined.csv"
OUTPUT = ROOT / "outputs" / "pipeline_national_trend.png"

REQUIRED = ["GEO", "Business_characteristics", "Business_information",
            "Expected_change", "VALUE", "Quarter"]

DIRECTION_ALIASES = {
    "increase": "increase",
    "stay about the same": "stay about the same",
    "stay the same": "stay about the same",
    "decrease": "decrease",
}
METRIC_ALIASES = {
    "employment": "Employment", "sales": "Sales",
    "profitability": "Profitability",
    "investment": "Investment", "capital investment": "Investment",
}


def read_quarter(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing column(s) {missing}")

    df["direction"] = df["Expected_change"].str.strip().str.lower().map(DIRECTION_ALIASES)
    df["metric"] = df["Business_information"].str.strip().str.lower().map(METRIC_ALIASES)

    for col, src in [("direction", "Expected_change"), ("metric", "Business_information")]:
        bad = sorted(df.loc[df[col].isna(), src].unique())
        if bad:
            raise ValueError(f"{Path(path).name}: unrecognised {src} value(s) {bad}")

    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df[REQUIRED + ["direction", "metric"]]


def build_combined(new_file=None):
    if new_file:
        frames = [read_quarter(PROCESSED)] if PROCESSED.exists() else []
        frames.append(read_quarter(new_file))
    else:
        frames = [read_quarter(p) for p in sorted(glob.glob(str(RAW / "*.csv")))]

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        ["GEO", "Business_characteristics", "metric", "direction", "Quarter"], keep="last"
    )
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED, index=False)
    return combined


def to_cells(long_df):
    cells = (
        long_df.pivot_table(
            index=["GEO", "Business_characteristics", "metric", "Quarter"],
            columns="direction", values="VALUE", aggfunc="first", dropna=False,
        ).reset_index()
        .rename(columns={"increase": "pct_increase",
                         "stay about the same": "pct_same",
                         "decrease": "pct_decrease"})
    )
    parts = cells[["pct_increase", "pct_same", "pct_decrease"]]
    cells["base"] = parts.sum(axis=1, skipna=True)
    cells["n_missing"] = parts.isna().sum(axis=1)
    return cells


def qkey(q):
    n, y = q.split(" ")
    return (int(y), int(n[1]))


def national_trend(cells):
    total = [c for c in cells["Business_characteristics"].unique()
             if c.startswith("North American")][0]
    nat = cells[(cells["GEO"] == "Canada")
                & (cells["Business_characteristics"] == total)].copy()

    order = sorted(nat["Quarter"].unique(), key=qkey)

    # Reconstruct a missing 'decrease' in ANY quarter, not only the newest one:
    # a quarter stops being "latest" as soon as the next file arrives.
    complete_base = nat[nat["n_missing"] == 0].groupby("metric")["base"].mean()
    fix = nat["pct_decrease"].isna() & nat["metric"].map(complete_base).notna()
    imputed = []
    if fix.any():
        nat.loc[fix, "pct_decrease"] = (
            nat.loc[fix, "metric"].map(complete_base)
            - nat.loc[fix, "pct_increase"] - nat.loc[fix, "pct_same"]
        )
        imputed = list(zip(nat.loc[fix, "Quarter"], nat.loc[fix, "metric"]))
        for q in sorted(nat.loc[fix, "Quarter"].unique(), key=qkey):
            n = int((fix & nat["Quarter"].eq(q)).sum())
            print(f"[info] reconstructed {n} missing 'decrease' value(s) in {q}")

    nat["base"] = nat[["pct_increase", "pct_same", "pct_decrease"]].sum(axis=1)
    nat["net_balance"] = 100 * (nat["pct_increase"] - nat["pct_decrease"]) / nat["base"]

    wide = nat.pivot(index="Quarter", columns="metric", values="net_balance").reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5))
    for m in wide.columns:
        line, = ax.plot(wide.index, wide[m], marker="o", label=m)
        est = [q for q in wide.index if (q, m) in imputed]
        if est:
            ax.plot(est, wide.loc[est, m], marker="o", color=line.get_color(),
                    markerfacecolor="white", markersize=9, ls="none")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Net balance (% increase minus % decrease)")
    ax.set_title("Canadian business expectations, next 3 months\nAll industries, national")
    ax.legend()
    ax.grid(alpha=0.3)
    if imputed:
        ax.text(0.0, -0.14,
                "Hollow markers indicate a reconstructed 'decrease' value, filled from "
                "that metric's average base across quarters where all three responses "
                "are present.",
                transform=ax.transAxes, fontsize=8, color="#555", va="top")
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    return wide


if __name__ == "__main__":
    new_file = sys.argv[1] if len(sys.argv) > 1 else None
    combined = build_combined(new_file)
    print(f"[info] combined dataset: {len(combined)} rows, "
          f"{combined['Quarter'].nunique()} quarters -> {PROCESSED}")
    wide = national_trend(to_cells(combined))
    print(wide.round(1))
    print(f"[info] chart written to {OUTPUT}")
