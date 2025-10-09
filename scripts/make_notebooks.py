from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb_dir = root / "notebooks"
nb_dir.mkdir(parents=True, exist_ok=True)

md = """# SmartSeat Data/ML Baseline (Week 4)

We demonstrate a minimal, runnable baseline for SmartSeat’s seat recommendation.

- **Data**: `data/sample_seats.csv`  
- **Logic**: prefer near-power (×2) + window (×1); only unoccupied seats  
- **Output**: recommended seat_id (expected: `3` for baseline)
"""

code1 = """from src.baseline import recommend_seat
print("Recommended seat:", recommend_seat("data/sample_seats.csv"))"""

code2 = """from src.baseline import load_seats, score_rows
df = load_seats("data/sample_seats.csv")
scored = score_rows(df).sort_values(
    ["score","is_near_power","is_window","seat_id"],
    ascending=[False,False,False,True]
)
scored"""

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(md),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
]
(nb_dir / "baseline_model.ipynb").write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote notebooks/baseline_model.ipynb")
