import pandas as pd  # Generate data from CSV
import statsmodels.api as sm  # Build logistic model
from sklearn.preprocessing import LabelEncoder  # Encode categorical variables
import matplotlib.pyplot as plt

# Read Data
df = pd.read_csv("survey_lung_cancer.csv")

# Encode categorical variables
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M:1, F:0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

# Target variable
target = 'LUNG_CANCER'
# Select independent variables
numeric_cols = [col for col in df.columns if col != target]

# Store statistically significant variables
p_values_list = []

# Univariate logistic regression
for col in numeric_cols:
    try:
        X = df[[col]].dropna()  # Remove rows with missing values in current column
        X = sm.add_constant(X)  # Add intercept term
        y = df.loc[X.index, target]  # Align target values with current X index
        
        model = sm.Logit(y, X).fit(disp=0)
        p_value = model.pvalues[col]

        print(f"{col}: P-value = {p_value:.4f}")
        
        if p_value < 0.01:
            p_values_list.append((col, p_value))

    except Exception as e:
        print(f"{col}: Error - {e}")

# Top 4 predictors with lowest p-values
top4_vars = sorted(p_values_list, key=lambda x: x[1])[:4]

print("\n✅ Top 4 Predictors (p < 0.01):")
for var, pval in top4_vars:
    print(f"{var}: P-value = {pval:.4e}")

# Plot target distribution
counts = df['LUNG_CANCER'].value_counts().sort_index()
percentages = counts / counts.sum() * 100

# Plotting
plt.figure(figsize=(6, 4))
bars = plt.bar(['Không mắc', 'Mắc ung thư phổi'], percentages, color=['green', 'red'])

# Add percentage labels
for bar, percent in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{percent:.1f}%", ha='center')

plt.title("Phân phối tỷ lệ mắc ung thư phổi")
plt.ylabel("Tỷ lệ (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
