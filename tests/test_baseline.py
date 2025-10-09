from pathlib import Path
from src.baseline import recommend_seat, load_seats, score_rows

def test_recommendation_on_sample():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "sample_seats.csv"
    assert csv_path.exists(), f"missing sample csv: {csv_path}"
    assert recommend_seat(str(csv_path)) == 3  # 期望最佳为 3

def test_scoring_monotonic():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "sample_seats.csv"
    df = load_seats(str(csv_path))
    scored = score_rows(df)
    assert "is_occupied" in scored.columns
    assert (scored.loc[scored["is_occupied"] == 1, "score"] == 0).all()
