from src.baseline import recommend_seat, load_seats, score_rows

def test_recommendation_on_sample():
    seat_id = recommend_seat("data/sample_seats.csv")
    assert isinstance(seat_id, int)
    assert seat_id == 3  # 课程里 sample_seats.csv 的最佳座位就是3

def test_scoring_monotonic():
    df = load_seats("data/sample_seats.csv")
    scored = score_rows(df)
    assert scored["score"].max() >= scored["score"].min()
