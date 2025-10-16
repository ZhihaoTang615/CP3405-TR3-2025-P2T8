import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_eval_open_source():
    """
    改写自 aqibsaeed/Occupancy-Detection
    用传感器数据预测房间是否被占用。
    """
    # 读取数据文件（注意路径要对）
    df = pd.read_csv("smartseat-ml/external/Occupancy_model/datatraining.txt")

    # 选择特征和标签
    X = df[["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]]
    y = df["Occupancy"]

    # 分训练集 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用随机森林模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率、精确率、召回率
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    }
