import pandas as pd
from typing import Optional

def load_seats(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"seat_id","is_near_power","is_window","occupied"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    available = df[df["occupied"] == 0].copy()
    available["is_near_power"] = available["is_near_power"].astype(int)
    available["is_window"] = available["is_window"].astype(int)
    available["score"] = 2*available["is_near_power"] + 1*available["is_window"]
    return available

def recommend_seat(csv_path: str) -> Optional[int]:
    df = load_seats(csv_path)
    scored = score_rows(df)
    if scored.empty:
        return None
    best = scored.sort_values(
        ["score","is_near_power","is_window","seat_id"],
        ascending=[False,False,False,True]
    ).head(1)
    return int(best["seat_id"].values[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rule-based seat recommendation baseline")
    parser.add_argument("--csv", required=True, help="Path to seats CSV")
    args = parser.parse_args()
    rec = recommend_seat(args.csv)
    if rec is None:
        print("No available seats.")
    else:
        print(f"Baseline recommended seat: {rec}")
