# smartseat-ml/smartseat_ml/baseline.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path

def train_and_evaluate_baseline():
    # 使用 Occupancy 数据集
    data_path = Path(__file__).resolve().parents[1] / "external" / "Occupancy_model" / "datatraining.txt"
    df = pd.read_csv(data_path)
    X = df[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
    y = df["Occupancy"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return {"accuracy": float(acc)}
