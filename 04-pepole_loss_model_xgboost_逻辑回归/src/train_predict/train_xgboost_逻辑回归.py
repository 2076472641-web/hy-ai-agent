# ====================== 1. 导入需要的库 ======================
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 2. 基础配置 & 数据预处理 ======================
FILE_PATH = "C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/data/train.csv"

# 1.数据预处理
def data_preprocess(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    # 删除无用列:01_EmployeeNumber(员工序列编码,纯数字)/02_Over18(是否成年,全部都是'Y'成年数值,一模一样)
    # /03_StandardHours(标准工作时长,全部都是80,一模一样)
    df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
    # 缺失值、重复值处理
    df = df.dropna(how='any').drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# 2.EDA数据探索可视化
def eda(data_df):
    print("开始EDA数据探索分析...")
    df = data_df.copy()

    plt.figure(figsize=(20, 15))

    # 年龄分布
    plt.subplot(2, 2, 1)
    plt.hist(df['Age'], bins=20, rwidth=0.8)
    plt.title("年龄分布")
    plt.xlabel("年龄")
    plt.ylabel("人数")
    plt.grid()

    # 月收入分布
    plt.subplot(2, 2, 2)
    plt.hist(df['MonthlyIncome'], bins=30, rwidth=0.8)
    plt.title("月收入分布")
    plt.xlabel("月收入")
    plt.ylabel("人数")
    plt.grid()

    # 离职在职人数
    plt.subplot(2, 2, 3)
    attr_count = df['Attrition'].value_counts()
    plt.bar(['在职', '离职'], attr_count.values)
    plt.title("离职 vs 在职人数")
    plt.ylabel("人数")
    plt.grid()

    # 加班与离职率关系
    plt.subplot(2, 2, 4)
    overtime_attr = df.groupby('OverTime')['Attrition'].mean()
    plt.bar(['不加班', '加班'], overtime_attr.values)
    plt.title("加班与离职率关系")
    plt.ylabel("平均离职率")
    plt.grid()

    plt.tight_layout()
    img_path = "C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/img"
    os.makedirs(img_path, exist_ok=True)
    plt.savefig(os.path.join(img_path, "eda.png"))
    plt.close()
    print("EDA数据探索完成，已保存图片到img文件夹")

# 3.特征工程
def feature_engineering(data_df):
    print("开始特征工程...")
    df = data_df.copy()
    # 类别特征数值映射 分为0和1 方便电脑识别
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}) # 性别:男:1/女:0
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0}) # 是否加班:yes:1/no:0
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df['BusinessTravel'] = df['BusinessTravel'].map(travel_map) # 出差:不出差:0 很少出差:1 经常出差:2
    # 独热编码:把不能计算的文字类别（部门、教育领域、职位、婚姻状况）变成 机器能看懂的 0/1 数字列
    df = pd.get_dummies(df, columns=[
        'Department', 'EducationField', 'JobRole', 'MaritalStatus'
    ], drop_first=True)
    print("特征工程完成")
    return df
# 模型预测结果对比绘图
def plot_result(y_test, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:80], label='真实值', linewidth=2, marker='o')
    plt.plot(y_pred[:80], label='预测值', linewidth=2, marker='x')
    plt.title(title)
    plt.xlabel("样本序号")
    plt.ylabel("是否离职（0=在职，1=离职）")
    plt.legend()
    plt.grid(True)

    img_path = "C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/img"
    os.makedirs(img_path, exist_ok=True)
    plt.savefig(os.path.join(img_path, f"{title}.png"))
    plt.close()

# ====================== 3. 加载数据 + 处理 ======================
if __name__ == '__main__':
    # 数据预处理
    raw_df = data_preprocess(FILE_PATH)
    # EDA可视化
    eda(raw_df)
    # 特征工程
    df = feature_engineering(raw_df)

    # 分割特征和标签
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    # 保存标准化器
    model_dir = "C:/python_code_pro/demon1/talent_loss_model_xgboost_逻辑回归/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    # ====================== 4. 定义两个模型 ======================
    models = {
        "逻辑回归": LogisticRegression(max_iter=2000, random_state=7),
        "XGBoost": XGBClassifier(max_depth=3, n_estimators=200, random_state=7, eval_metric="logloss")
    }

    # 存储结果
    results = []

    # ====================== 5. 训练 + 评估 两个模型 ======================
    for name, model in models.items():
        print(f"======================================")
        print(f"正在训练：{name}")

        # 训练
        model.fit(X_train, y_train)
        # 保存模型
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        # 输出结果
        print(f"准确率(Accuracy): {acc:.4f}")
        print(f"精确率(Precision): {precision:.4f}")
        print(f"召回率(Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC值: {auc:.4f}")
        print(f"混淆矩阵:\n{cm}")

        # 绘制预测对比图
        plot_result(y_test, y_pred, f"{name} 预测结果对比")

        # 保存到总表
        results.append({
            "模型": name,
            "准确率": round(acc, 4),
            "精确率": round(precision, 4),
            "召回率": round(recall, 4),
            "F1分数": round(f1, 4),
            "AUC": round(auc, 4)
        })

    # ====================== 6. 最终模型对比表 ======================
    result_df = pd.DataFrame(results)
    print("\n======================================")
    print("人才流失预测模型 最终对比结果")
    print(result_df)

    # ====================== 7. 自动选出最优模型 ======================
    best_model = result_df.loc[result_df["AUC"].idxmax()]
    print("\n✅ 人才流失预测的最优模型是：")
    print(best_model)