"""Tests for the cleaning and metric logic in src/ingest.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ingest


def make_file(tmp_path, rows, name="test.csv"):
    """Write a small CSV in the same shape as a real quarterly file."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


BASE_ROW = {
    "GEO": "Canada",
    "Business_characteristics": "Retail trade",
    "Business_information": "Sales",
    "Expected_change": "increase",
    "VALUE": 20.0,
    "Quarter": "Q1 2024",
}


def row(**overrides):
    r = dict(BASE_ROW)
    r.update(overrides)
    return r


def test_drifted_labels_are_normalised(tmp_path):
    """The Q2 2024 wording should map onto the same categories as other quarters."""
    path = make_file(tmp_path, [
        row(Expected_change="stay the same"),
        row(Business_information="Capital Investment"),
    ])
    out = ingest.read_quarter(path)
    assert out["direction"].tolist() == ["stay about the same", "increase"]
    assert out["metric"].tolist() == ["Sales", "Investment"]


def test_case_and_whitespace_are_tolerated(tmp_path):
    path = make_file(tmp_path, [row(Expected_change="  INCREASE ")])
    out = ingest.read_quarter(path)
    assert out["direction"].iloc[0] == "increase"


def test_unknown_category_raises(tmp_path):
    """A new response option must stop the run rather than be silently dropped."""
    path = make_file(tmp_path, [row(Expected_change="might increase")])
    with pytest.raises(ValueError, match="unrecognised"):
        ingest.read_quarter(path)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"GEO": "Canada", "VALUE": 1.0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        ingest.read_quarter(path)


def test_quarter_sort_is_chronological():
    """Q4 2023 must come before Q1 2024, which alphabetical ordering would get wrong."""
    quarters = ["Q1 2024", "Q3 2023", "Q2 2024", "Q4 2023"]
    assert sorted(quarters, key=ingest.qkey) == [
        "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024",
    ]


def test_base_and_missing_count_are_computed():
    """base is the group's own total, which is below 100 when a response is absent."""
    long_df = pd.DataFrame([
        row(Expected_change="increase", VALUE=20.0, direction="increase", metric="Sales"),
        row(Expected_change="stay about the same", VALUE=60.0,
            direction="stay about the same", metric="Sales"),
        row(Expected_change="decrease", VALUE=None, direction="decrease", metric="Sales"),
    ])
    cells = ingest.to_cells(long_df)
    assert cells["base"].iloc[0] == pytest.approx(80.0)
    assert cells["n_missing"].iloc[0] == 1


def test_reweighting_uses_the_group_base_not_100():
    """A group totalling 96 must be scored out of 96, not out of 100."""
    cells = pd.DataFrame([{
        "GEO": "Canada",
        "Business_characteristics": "North American Industry Classification System (NAICS), all industries",
        "metric": "Investment",
        "Quarter": "Q1 2024",
        "pct_increase": 47.5,
        "pct_same": 45.6,
        "pct_decrease": 2.9,
        "base": 96.0,
        "n_missing": 0,
    }])
    wide = ingest.national_trend(cells)
    assert wide.loc["Q1 2024", "Investment"] == pytest.approx(46.458, abs=0.01)


def test_missing_decrease_is_reconstructed_in_any_quarter():
    """A quarter that is no longer the newest must still get its gap filled."""
    total = "North American Industry Classification System (NAICS), all industries"
    cells = pd.DataFrame([
        {"GEO": "Canada", "Business_characteristics": total, "metric": "Sales",
         "Quarter": "Q1 2024", "pct_increase": 18.0, "pct_same": 62.0,
         "pct_decrease": 20.0, "base": 100.0, "n_missing": 0},
        {"GEO": "Canada", "Business_characteristics": total, "metric": "Sales",
         "Quarter": "Q2 2024", "pct_increase": 20.0, "pct_same": 60.0,
         "pct_decrease": None, "base": 80.0, "n_missing": 1},
        {"GEO": "Canada", "Business_characteristics": total, "metric": "Sales",
         "Quarter": "Q3 2024", "pct_increase": 22.0, "pct_same": 58.0,
         "pct_decrease": 20.0, "base": 100.0, "n_missing": 0},
    ])
    wide = ingest.national_trend(cells)
    # Q2 is not the newest quarter here, but its gap should still be filled:
    # decrease = 100 - 20 - 60 = 20, so net balance = 20 - 20 = 0.
    assert wide.loc["Q2 2024", "Sales"] == pytest.approx(0.0, abs=0.01)
    assert not pd.isna(wide.loc["Q2 2024", "Sales"])