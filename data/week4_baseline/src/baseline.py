import pandas as pd

def load_seats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Rule: near power = 2 points, window = 1 point (only for unoccupied seats)
    df = df.copy()
    for col in ["is_near_power", "is_window", "is_occupied"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df["score"] = (2 * df["is_near_power"] + 1 * df["is_window"]) * (1 - df["is_occupied"])
    return df

def recommend_seat(path: str):
    df = load_seats(path)
    scored = score_rows(df)
    ranked = scored.sort_values(
        ["score", "is_near_power", "is_window", "seat_id"],
        ascending=[False, False, False, True]
    )
    return int(ranked.iloc[0]["seat_id"])
