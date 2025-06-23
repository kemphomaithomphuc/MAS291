import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np

# Cài đặt hiển thị
sns.set_theme(style="whitegrid")
plt.rcParams['axes.titlepad'] = 15

# Đọc dữ liệu
file_path = "survey_lung_cancer.csv"
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.upper()
    print("Read data successfully.")
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path and try again.")
    raise SystemExit

# Mã hóa biến phân loại
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M:1, F:0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# Hồi quy logistic đơn biến để lấy p-value
p_values = []
target = 'LUNG_CANCER'
features = [col for col in df.columns if col != target]

for col in features:
    try:
        X = sm.add_constant(df[[col]])
        y = df[target]
        model = sm.Logit(y, X).fit(disp=0)
        p = model.pvalues[col]
        p_values.append((col, p))
    except:
        continue

# In toàn bộ p-value
print("\nP-values của tất cả các biến:")
for var, p in sorted(p_values, key=lambda x: x[1]):
    print(f"{var:15}: {p:.4f}")

# Lấy 4 biến có p-value nhỏ nhất
top_features = sorted(p_values, key=lambda x: x[1])[:4]
top_vars = [f[0] for f in top_features]

print("\n4 biến có ý nghĩa thống kê cao nhất:")
for var in top_vars:
    print(f"- {var}")

# Chuẩn bị dữ liệu để hiển thị
df_plot = df.copy()
df_plot['GENDER'] = df_plot['GENDER'].map({1: 'M', 0: 'F'})
df_plot['LUNG_CANCER'] = df_plot['LUNG_CANCER'].map({1: 'YES', 0: 'NO'})

# Biểu đồ phân bố phần trăm theo giới tính
for gender in ['M', 'F']:
    gender_df = df_plot[df_plot['GENDER'] == gender]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_vars):
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
    plt.show()

# Biểu đồ xác suất hồi quy logistic theo từng biến
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
    
    # In ra phương trình sigmoid
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