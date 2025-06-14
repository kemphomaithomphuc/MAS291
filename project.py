import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Đọc và xử lý dữ liệu
# -------------------------------
df = pd.read_csv("survey_lung_cancer.csv")

label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M:1, F:0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

target = 'LUNG_CANCER'
numeric_cols = [col for col in df.columns if col != target]

# -------------------------------
# 2. Hồi quy đơn biến để chọn biến mạnh
# -------------------------------
p_values_list = []

for col in numeric_cols:
    try:
        X = df[[col]].dropna()
        X = sm.add_constant(X)
        y = df.loc[X.index, target]
        model = sm.Logit(y, X).fit(disp=0)
        p_value = model.pvalues[col]
        print(f"{col}: P-value = {p_value:.4f}")
        if p_value < 0.01:
            p_values_list.append((col, p_value))
    except Exception as e:
        print(f"{col}: Error - {e}")

# Top 4 biến có p-value thấp nhất
top4_vars = sorted(p_values_list, key=lambda x: x[1])[:4]
top4_features = [var for var, _ in top4_vars]

print("\n✅ Top 4 Predictors (p < 0.01):")
for var, pval in top4_vars:
    print(f"{var}: P-value = {pval:.4e}")

# -------------------------------
# 3. Vẽ biểu đồ phân phối nhãn
# -------------------------------
counts = df[target].value_counts().sort_index()
percentages = counts / counts.sum() * 100

plt.figure(figsize=(6, 4))
bars = plt.bar(['Không mắc', 'Mắc ung thư phổi'], percentages, color=['green', 'red'])

for bar, percent in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{percent:.1f}%", ha='center')

plt.title("Phân phối tỷ lệ mắc ung thư phổi")
plt.ylabel("Tỷ lệ (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Huấn luyện mô hình logistic đa biến
# -------------------------------
X = df[top4_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# 5. Đánh giá mô hình
# -------------------------------
print("\n🎯 Đánh giá mô hình logistic đa biến:")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
print("🔎 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()