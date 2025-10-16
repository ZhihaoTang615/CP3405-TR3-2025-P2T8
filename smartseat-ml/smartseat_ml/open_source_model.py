from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 绝对路径：定位到 external/Occupancy_model/datatraining.txt
DATA = Path(__file__).resolve().parents[1] / "external" / "Occupancy_model" / "datatraining.txt"

def train_and_eval_open_source() -> dict:
    """
    用开源 Occupancy 数据训练一个随机森林分类器。
    返回 accuracy/precision/recall 作为指标。
    """
    if not DATA.exists():
        raise FileNotFoundError(f"Data file not found: {DATA}")
    df = pd.read_csv(DATA)

    needed = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = df[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
    y = df["Occupancy"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    return {
        "accuracy": float(accuracy_score(yte, yhat)),
        "precision": float(precision_score(yte, yhat, zero_division=0)),
        "recall": float(recall_score(yte, yhat, zero_division=0)),
    }
print(train_and_eval_open_source())
