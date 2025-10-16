# smartseat-ml/tests/test_open_source_model.py
import sys, pathlib
# 把 smartseat-ml 加到 Python 路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def test_open_source_model():
    from smartseat_ml.open_source_model import train_and_eval_open_source
    m = train_and_eval_open_source()
    print("模型评估结果:", m)
    assert "accuracy" in m
    assert 0.0 <= m["accuracy"] <= 1.0
