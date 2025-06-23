import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# 1. Read the dataset
file_path = "survey_lung_cancer.csv"
try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.upper()
    print("Read data successfully.")
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path and try again.")
    exit()

# Encode categorical variables
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M:1, F:0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# Run logistic regression for each feature and collect p-values
p_values_list = []
target = 'LUNG_CANCER'
features = [col for col in df.columns if col != target]

for col in features:
    try:
        X = df[[col]].dropna()
        X = sm.add_constant(X)
        y = df.loc[X.index, target]
        model = sm.Logit(y, X).fit(disp=0)
        p_value = model.pvalues[col]
        print(f"{col}: P-value = {p_value:.11f}")
        if p_value < 0.01:
            p_values_list.append((col, p_value))
    except Exception as e:
        print(f"{col}: Error - {e}")

# Top 4 variables with the lowest p-values
top4_vars = sorted(p_values_list, key=lambda x: x[1])[:4]

print("\nTop 4 Predictors (p < 0.01):")
for var, pval in top4_vars:
    print(f"{var}: P-value = {pval:.4e}")

# Plot the percentage distribution using a bar chart with labels
plt.figure(figsize=(6, 4))
cancer_counts = df['LUNG_CANCER'].value_counts(normalize=True) * 100
labels = ['No Lung Cancer', 'Lung Cancer']
colors = ['green', 'red']  # 0: No, 1: Yes

# Map index 0 → 'No Lung Cancer', 1 → 'Lung Cancer'
cancer_counts.index = labels

sns.barplot(x=cancer_counts.index, y=cancer_counts.values, palette=colors, legend=False) # newUpdate

# Add percentage labels on top of each bar
for i, value in enumerate(cancer_counts.values):
    plt.text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=11)

plt.ylabel('Percentage')
plt.title('Percentage Distribution of Lung Cancer Risk')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Curve
selected_features = ['ALCOHOL CONSUMING', 'ALLERGY', 'COUGHING', 'WHEEZING']
y = df['LUNG_CANCER']  # khai báo rõ biến mục tiêu

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, feature in enumerate(selected_features):
    X = df[[feature]]
    model = LogisticRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min().iloc[0], X.max().iloc[0], 300).reshape(-1, 1) #newUpdate
    y_prob = model.predict_proba(pd.DataFrame(x_range, columns=[feature]))[:, 1] #newUpdate


    axs[i].scatter(X, y, color='black', alpha=0.3, label='Actual data')
    axs[i].plot(x_range, y_prob, color='red', linewidth=2, label='Logistic Regression Curve')
    axs[i].set_title(f"Probability of Lung Cancer by {feature}", fontsize=14)
    axs[i].set_xlabel(feature, fontsize=12)
    axs[i].set_ylabel("Probability (%)", fontsize=12)
    axs[i].legend()
    axs[i].grid(alpha=0.3)

# fix layout
plt.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.4, top=0.92)
plt.show()