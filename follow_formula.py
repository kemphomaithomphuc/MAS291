# survey_lung_cancer.py

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Cài đặt hiển thị đồ thị
sns.set(style="whitegrid")
plt.rcParams['axes.titlepad'] = 15

# 1. Đọc dữ liệu
file_path = "survey_lung_cancer.csv"
df = pd.read_csv(file_path)

# 2. Mã hóa biến phân loại
label_enc = LabelEncoder()
if df['GENDER'].dtype == 'object':
    df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M=1, F=0

if df['LUNG_CANCER'].dtype == 'object':
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# 3. Tính p-value bằng hồi quy logistic đơn biến
p_values = []
for col in df.columns:
    if col == 'LUNG_CANCER':
        continue
    try:
        X = sm.add_constant(df[[col]])  # Thêm hệ số chệch β0
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

# 5. Chuẩn bị dữ liệu vẽ biểu đồ
df_plot = df.copy()
df_plot['GENDER'] = df_plot['GENDER'].map({1: 'M', 0: 'F'})
df_plot['LUNG_CANCER'] = df_plot['LUNG_CANCER'].map({1: 'YES', 0: 'NO'})

# 6. Vẽ biểu đồ phân tích theo giới tính
for gender in ['M', 'F']:
    gender_df = df_plot[df_plot['GENDER'] == gender]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_vars):
        # Tính phần trăm
        grouped = pd.crosstab(
            index=gender_df[feature],
            columns=gender_df['LUNG_CANCER'],
            normalize='index'
        ) * 100
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
            ci=None
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

# 7. Xuất dữ liệu đã chọn thành file
cleaned_df = df[top_vars + ['LUNG_CANCER']]
cleaned_df.to_csv("cleaning_data.csv", index=False)
print(f"\n✅ Đã lưu dữ liệu gồm các biến: {top_vars} và 'LUNG_CANCER' vào cleaning_data.csv")
