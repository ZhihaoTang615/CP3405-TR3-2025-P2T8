# smartseat_ml/open_source_model.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 基于当前文件，回到 smartseat-ml，再进 external/Occupancy_model/datatraining.txt
DATA = Path(__file__).resolve().parents[1] / "external" / "Occupancy_model" / "datatraining.txt"

def train_and_eval_open_source():
    df = pd.read_csv(DATA)
    X = df[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
    y = df["Occupancy"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, yhat)),
        "precision": float(precision_score(yte, yhat)),
        "recall": float(recall_score(yte, yhat)),
    }

