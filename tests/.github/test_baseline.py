import os
from src.baseline import recommend_seat

def test_recommendation_on_sample():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_seats.csv")
    assert recommend_seat(csv_path) == 3
