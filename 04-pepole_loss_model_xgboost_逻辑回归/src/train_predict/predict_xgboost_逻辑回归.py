# ====================== 1. 导入需要的库 ======================
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ====================== 2. 加载测试数据 & 预处理 ======================
# 读取测试集数据
df = pd.read_csv("C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/data/test2.csv")

# 分离特征和真实标签
X_test = df.drop("Attrition", axis=1)
y_true = df["Attrition"]

# 加载训练时保存的 标准化器 和 编码工具
scaler = joblib.load("C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/model/scaler.pkl")

# 与训练代码一致的预处理
drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']
X_test = X_test.drop(columns=drop_cols, errors='ignore')

# 特征映射
X_test['Gender'] = X_test['Gender'].map({'Male': 1, 'Female': 0})
X_test['OverTime'] = X_test['OverTime'].map({'Yes': 1, 'No': 0})
travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
X_test['BusinessTravel'] = X_test['BusinessTravel'].map(travel_map)

# 独热编码
X_test = pd.get_dummies(X_test, columns=[
    'Department', 'EducationField', 'JobRole', 'MaritalStatus'
], drop_first=True)

# 标准化
X_test = scaler.transform(X_test)

# ====================== 3. 加载两个训练好的模型 ======================
models = {
    "逻辑回归": joblib.load("C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/model/lr_model.pkl"),
    "XGBoost": joblib.load("C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/model/xgb_model.pkl")
}

# 存储结果
results = []
predictions = {}

# ====================== 4. 预测 + 评估 ======================
for name, model in models.items():
    print(f"======================================")
    print(f"正在预测：{name}")

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 保存预测值
    predictions[name] = y_pred

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)

    # 输出
    print(f"准确率(Accuracy): {acc:.4f}")
    print(f"精确率(Precision): {precision:.4f}")
    print(f"召回率(Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC值: {auc:.4f}")
    print(f"混淆矩阵:\n{cm}")

    # 保存到结果表
    results.append({
        "模型": name,
        "准确率": round(acc, 4),
        "精确率": round(precision, 4),
        "召回率": round(recall, 4),
        "F1分数": round(f1, 4),
        "AUC": round(auc, 4)
    })

# ====================== 5. 测试集最终对比表 ======================
result_df = pd.DataFrame(results)
print("\n======================================")
print("人才流失预测 - 测试集模型对比结果")
print(result_df)

# ====================== 6. 最优模型 ======================
best_model = result_df.loc[result_df["AUC"].idxmax()]
print("\n✅ 测试集最优模型：")
print(best_model)

# ====================== 7. 输出预测值 ======================
print("\n======================================")
print("📌 测试集前10条预测结果（0=在职  1=离职）")
pred_result = pd.DataFrame({
    "真实值": y_true[:10].values,
    "XGBoost预测": predictions["XGBoost"][:10],
    "逻辑回归预测": predictions["逻辑回归"][:10]
})
print(pred_result)