import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np

# Cài đặt hiển thị
sns.set(style="whitegrid")
plt.rcParams['axes.titlepad'] = 15

# 1. Đọc dữ liệu
file_path = "survey_lung_cancer.csv"
df = pd.read_csv(file_path)

# 2. Mã hóa biến phân loại
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])

# 3. Tính p-value bằng hồi quy logistic đơn biến
p_values = []
for col in df.columns:
    if col == 'LUNG_CANCER':
        continue
    try:
        X = sm.add_constant(df[[col]])
        y = df['LUNG_CANCER']
        model = sm.Logit(y, X).fit(disp=0)
        p_values.append((col, model.pvalues[col]))
    except Exception as e:
        print(f"❌ Bỏ qua biến '{col}' do lỗi: {e}")

# 4. Chọn 4 biến có p-value nhỏ nhất
top_features = sorted(p_values, key=lambda x: x[1])[:4]
top_vars = [f[0] for f in top_features]

print("🎯 Top 4 biến liên quan nhất đến ung thư phổi:")
for var, p in top_features:
    print(f"- {var} (p-value: {p:.5f})")

# 5. Hồi quy logistic đa biến với 4 biến
X_multi = sm.add_constant(df[top_vars])
y_multi = df['LUNG_CANCER']
multi_model = sm.Logit(y_multi, X_multi).fit()
print("\n📊 Kết quả hồi quy logistic đa biến (Top 4 biến):")
print(multi_model.summary())

# 6. Tính xác suất dự đoán với 4 biến
df['PREDICTED_PROB_TOP4'] = multi_model.predict(X_multi)

# 7. Hồi quy logistic với TOÀN BỘ biến
X_all = df.drop(columns=['LUNG_CANCER', 'PREDICTED_PROB_TOP4'])
X_all = sm.add_constant(X_all)
y_all = df['LUNG_CANCER']
full_model = sm.Logit(y_all, X_all).fit_regularized()
print("\n🧠 Kết quả hồi quy logistic đa biến với TẤT CẢ các biến:")
print(full_model.summary())
df['PREDICTED_PROB_ALL'] = full_model.predict(X_all)

# 8. Chuẩn bị dữ liệu vẽ biểu đồ
df_plot = df.copy()
df_plot['GENDER'] = df_plot['GENDER'].map({1: 'M', 0: 'F'})
df_plot['LUNG_CANCER'] = df_plot['LUNG_CANCER'].map({1: 'YES', 0: 'NO'})

# 9. Vẽ biểu đồ phân tích theo giới tính
for gender in ['M', 'F']:
    gender_df = df_plot[df_plot['GENDER'] == gender]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_vars):
        grouped = pd.crosstab(index=gender_df[feature], columns=gender_df['LUNG_CANCER'], normalize='index') * 100
        grouped = grouped.reset_index()
        melted = pd.melt(grouped, id_vars=[feature], var_name='LUNG_CANCER', value_name='PERCENTAGE')

        ax = axes[i]
        chart = sns.barplot(
            data=melted,
            x=feature,
            y='PERCENTAGE',
            hue='LUNG_CANCER',
            palette={'YES': 'red', 'NO': 'green'},
            ax=ax,
            errorbar=None
        )

        ax.set_title(f'{gender} - {feature}', fontsize=14, pad=12)
        ax.set_xlabel(feature, labelpad=8)
        ax.set_ylabel('Percentage (%)', labelpad=8)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(title='Lung Cancer', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        for container in chart.containers:
            chart.bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9, padding=2)

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f'Phân bố nguy cơ ung thư phổi theo {gender}', fontsize=16)
    plt.show()

# 10. Phân bố xác suất dự đoán từ mô hình toàn phần
plt.figure(figsize=(10, 5))
sns.histplot(df['PREDICTED_PROB_ALL'], bins=20, kde=True, color='darkorange')
plt.title('Phân bố xác suất dự đoán (toàn bộ biến)', fontsize=16)
plt.xlabel('Xác suất (0 → 1)', fontsize=12)
plt.ylabel('Số người', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 11. Biểu đồ sigmoid với từng biến Top 4
y = df['LUNG_CANCER']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, feature in enumerate(top_vars):
    X = df[[feature]]
    model = LogisticRegression()
    model.fit(X, y)

    x_range = np.linspace(X[feature].min(), X[feature].max(), 300).reshape(-1, 1)
    beta1 = model.coef_[0][0]
    beta0 = model.intercept_[0]

    print(f"\n{feature}: f(x) = 1 / (1 + exp(-({beta1:.4f} * x + {beta0:.4f})))")

    y_prob = 1 / (1 + np.exp(-(beta1 * x_range + beta0)))

    axs[i].scatter(X, y, color='black', alpha=0.3, label='Actual data')
    axs[i].plot(x_range, y_prob, color='red', linewidth=2, label='Sigmoid Curve')
    axs[i].set_title(f"Probability of Lung Cancer by {feature}", fontsize=14)
    axs[i].set_xlabel(feature, fontsize=12)
    axs[i].set_ylabel("Probability (%)", fontsize=12)
    axs[i].legend()
    axs[i].grid(alpha=0.3)

plt.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.4, top=0.92)
plt.show()

# 12. Xuất dữ liệu
final_df = df[top_vars + ['LUNG_CANCER', 'PREDICTED_PROB_TOP4', 'PREDICTED_PROB_ALL']]
final_df.to_csv("cleaning_data.csv", index=False)
print(f"\n✅ Đã lưu dữ liệu gồm {top_vars}, 'LUNG_CANCER', 'PREDICTED_PROB_TOP4', 'PREDICTED_PROB_ALL' vào cleaning_data.csv")

# 13. Nhập thông tin bệnh nhân để dự đoán
print("\n🧪 Nhập thông tin bệnh nhân để dự đoán nguy cơ ung thư phổi")
patient_data_top4 = {}
for feature in top_vars:
    val = input(f"Nhập giá trị cho '{feature}' (0 hoặc 1 hoặc số cụ thể nếu là số): ")
    try:
        patient_data_top4[feature] = float(val)
    except:
        print(f"⚠️ Giá trị không hợp lệ cho '{feature}', mặc định 0")
        patient_data_top4[feature] = 0.0

input_df_top4 = pd.DataFrame([patient_data_top4])
input_df_top4 = sm.add_constant(input_df_top4, has_constant='add')  # ✅ Fix shape
prob_top4 = multi_model.predict(input_df_top4)[0]
print(f"📈 Xác suất bị ung thư phổi (Top 4 biến): {prob_top4:.4f} ({prob_top4*100:.2f}%)")

use_full = input("\n❓ Bạn có muốn dự đoán bằng mô hình TẤT CẢ biến? (y/n): ").strip().lower()
if use_full == 'y':
    patient_data_full = {}
    for col in df.columns:
        if col in ['LUNG_CANCER', 'PREDICTED_PROB_TOP4', 'PREDICTED_PROB_ALL']:
            continue
        val = input(f"Nhập giá trị cho '{col}' (0 hoặc 1 hoặc số cụ thể): ")
        try:
            patient_data_full[col] = float(val)
        except:
            print(f"⚠️ Giá trị không hợp lệ cho '{col}', mặc định 0")
            patient_data_full[col] = 0.0

    input_df_full = pd.DataFrame([patient_data_full])
    input_df_full = sm.add_constant(input_df_full, has_constant='add')  # ✅ Fix shape
    prob_full = full_model.predict(input_df_full)[0]
    print(f"🧠 Xác suất bị ung thư phổi (Toàn bộ biến): {prob_full:.4f} ({prob_full*100:.2f}%)")
