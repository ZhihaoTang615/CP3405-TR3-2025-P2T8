import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def test_rule_baseline():
    from smartseat_ml.baseline import evaluate_rules
    m = evaluate_rules()
    print("Rule baseline metrics:", m)
    assert "accuracy" in m and 0.0 <= m["accuracy"] <= 1.0

