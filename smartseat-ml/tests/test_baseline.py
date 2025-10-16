# smartseat-ml/tests/test_baseline.py
import sys, pathlib
import pytest

# 让 Python 能正确导入 smartseat_ml 包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def test_baseline_model_train_and_eval():
    """
    测试 baseline 模型是否能成功训练并输出准确率。
    """
    from smartseat_ml.baseline import train_and_evaluate_baseline

    # 执行函数
    results = train_and_evaluate_baseline()

    # 打印结果（CI 日志可见）
    print("Baseline test metrics:", results)

    # 检查返回类型与合理范围
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert 0.0 <= results["accuracy"] <= 1.0


def test_model_reproducibility():
    """
    检查重复运行结果是否稳定（容差范围内）
    """
    from smartseat_ml.baseline import train_and_evaluate_baseline

    r1 = train_and_evaluate_baseline()
    r2 = train_and_evaluate_baseline()

    diff = abs(r1["accuracy"] - r2["accuracy"])
    print(f"Accuracy difference between runs: {diff}")
    assert diff < 0.05, "模型结果波动过大"
