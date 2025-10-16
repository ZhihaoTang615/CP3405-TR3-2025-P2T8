from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "external" / "Occupancy_model" / "datatraining.txt"

def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    return df[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]].copy()

def rule_score(row) -> float:
    """
    一个简单可解释的规则打分：
    - 光照越大越可能有人（权重0.5）
    - CO2 越高越可能有人（权重0.3）
    - 温度略有影响（权重0.2）
    返回 0~1 之间的分数
    """
    # 归一化（按经验范围大致缩放，避免依赖训练）
    light = min(row["Light"] / 1000.0, 1.0)          # 常见 0~1000+
    co2   = min(row["CO2"] / 2000.0, 1.0)            # 常见 400~2000
    temp  = min(max((row["Temperature"] - 18) / 10, 0.0), 1.0)  # 18~28 度
    score = 0.5 * light + 0.3 * co2 + 0.2 * temp
    return float(score)

def predict_label(score: float, threshold: float = 0.5) -> int:
    return 1 if score >= threshold else 0

def evaluate_rules(threshold: float = 0.5) -> dict:
    """
    用规则打分对 Occupancy 做二分类，返回 accuracy/precision/recall。
    """
    df = _load()
    y_true = df["Occupancy"].astype(int).tolist()
    y_pred = [predict_label(rule_score(r), threshold) for _, r in df.iterrows()]

    tp = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==1)
    tn = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==0)
    fp = sum(1 for t,p in zip(y_true, y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true, y_pred) if t==1 and p==0)

    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec)}

