import pandas as pd

NEAR_POWER_WEIGHT = 2
WINDOW_WEIGHT = 1

def load_seats(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "is_occupied" not in df.columns:
        df["is_occupied"] = 0
    for col in ["is_near_power", "is_window", "is_occupied"]:
        if col not in df.columns:
            df[col] = 0
        # 统一成 0/1
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"true": "1", "false": "0"})
            .astype(float)
            .astype(int)
        )
    return df

def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_columns(df)
    df["score"] = (NEAR_POWER_WEIGHT * df["is_near_power"] + WINDOW_WEIGHT * df["is_window"]) * (1 - df["is_occupied"])
    return df

def recommend_seat(path: str) -> int:
    df = load_seats(path)
    scored = score_rows(df)
    ranked = scored.sort_values(
        ["score", "is_near_power", "is_window", "seat_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return int(ranked.iloc[0]["seat_id"])
