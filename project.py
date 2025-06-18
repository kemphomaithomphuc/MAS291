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
import statsmodels.api as sm
# 1. Read the dataset
file_path = "survey_lung_cancer.csv"
try:
    df = pd.read_csv(file_path)
    print("Read data successfully.")
except FileNotFoundError:
    print(f"File {file_path} not found. Please check the path and try again.")

# 2. Take top 4 features associated with Lung Cancer risk
# 2a. Encode categorical variables
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])         # M=1, F=0
df['LUNG_CANCER'] = label_enc.fit_transform(df['LUNG_CANCER'])  # YES=1, NO=0

# 2b. Take the independent features and target variable
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# 2c. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 2d. Train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 6. Select top 4 features
selector = SelectFromModel(log_reg, max_features=4, prefit=True)
top_features = X.columns[selector.get_support()].tolist()


#Show descriptive statistics for all features
#print("=== Summary Statistics for All Features ===")
#print(df.describe(include='all'))

# Focused statistics on top 4 selected features + target
#top_features_with_target = top_features + ['LUNG_CANCER']
#print("\n=== Summary Statistics for Top 4 Features + Target ===")
#print(df[top_features_with_target].describe())

# Add constant term for intercept
X_with_const = sm.add_constant(X)

# Fit logistic regression model using statsmodels
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit()

# Show p-values for each feature
print("=== P-values for Each Feature ===")
print(result.pvalues.sort_values())

# 7. Print top features
print("Top 4 features most associated with Lung Cancer risk:")
for feature in top_features:
    print(f"- {feature} (p-value: {result.pvalues[feature]:.10f})")
