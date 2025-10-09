import pandas as pd

def load_seats(path: str) -> pd.DataFrame:
    """Load seat CSV data."""
    return pd.read_csv(path)

def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add a score column based on power (x2) + window (x1), only unoccupied seats."""
    for col in ["is_near_power", "is_window", "is_occupied"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    # 只对未占用座位打分
    df = df[df["is_occupied"] == 0].copy()
    df["score"] = (2 * df["is_near_power"]) + (1 * df["is_window"])
    return df

def recommend_seat(path: str) -> int:
    """Return the best seat_id based on the rule scoring."""
    df = load_seats(path)
    scored = score_rows(df)
    ranked = scored.sort_values(
        ["score", "is_near_power", "is_window"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return int(ranked.iloc[0]["seat_id"])
