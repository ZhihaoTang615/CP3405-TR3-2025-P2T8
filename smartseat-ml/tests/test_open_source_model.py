import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def test_open_source_model():
    from smartseat_ml.open_source_model import train_and_eval_open_source
    m = train_and_eval_open_source()
    print("Open-source model metrics:", m)
    assert "accuracy" in m and 0.0 <= m["accuracy"] <= 1.0

