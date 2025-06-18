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
from sklearn.feature_selection import SelectFromModel

# 1. Read the dataset
file_path = "survey_lung_cancer.csv"
try:
    df = pd.read_csv(file_path)
    print("Read data successfully.")
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path and try again.")

# 2. Encode categorical variables
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])         # M=1, F=0
df['LUNG_CANCER'] = label_enc.fit_transform(df['LUNG_CANCER'])  # YES=1, NO=0

# 3. Prepare features and target
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 6. Select top 4 features
selector = SelectFromModel(log_reg, max_features=4, prefit=True)
top_features = X.columns[selector.get_support()].tolist()

# 7. P-values using statsmodels
X_with_const = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit()

# 8. Print p-values
print("=== P-values for Each Feature ===")
print(result.pvalues.sort_values())

# 9. Print top 4 features with p-values
print("\nTop 4 features most associated with Lung Cancer risk:")
for feature in top_features:
    print(f"- {feature} (p-value: {result.pvalues[feature]:.10f})")

# 10. Generate chart showing lung cancer risk distribution by gender
# Decode gender and lung cancer for display
gender_map = {1: 'M', 0: 'F'}
cancer_map = {1: 'YES', 0: 'NO'}
df_plot = df.copy()
df_plot['GENDER'] = df_plot['GENDER'].map(gender_map)
df_plot['LUNG_CANCER'] = df_plot['LUNG_CANCER'].map(cancer_map)

# Create crosstab and normalize by row (gender)
crosstab = pd.crosstab(df_plot['GENDER'], df_plot['LUNG_CANCER'], normalize='index') * 100

# Define custom color mapping: 'YES' → red, 'NO' → blue
color_map = ['blue' if col == 'NO' else 'red' for col in crosstab.columns]

# Plot with custom colors
crosstab.plot(kind='bar', stacked=True, color=color_map)
plt.title('Percentage Distribution of Lung Cancer Risk by Gender')
plt.ylabel('Percentage (%)')
plt.xlabel('Gender')
plt.legend(title='Lung Cancer Risk')
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
