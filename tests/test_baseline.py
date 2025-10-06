from pathlib import Path
from src.baseline import recommend_seat

def test_recommendation_on_sample():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "sample_seats.csv"
    assert recommend_seat(str(csv_path)) == 3