# lung_cancer_logreg_and_charts.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ ─── Load the data ────────────────────────────────────────────────
CSV_FILE = "survey_lung_cancer.csv"          # adjust if your file is elsewhere
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print("File not found")
    exit()

# 2️⃣ ─── Clean up ────────────────────────────────────────────────────
df.columns = df.columns.str.strip()          # trim stray spaces in headers
df = df.drop_duplicates()

# 3️⃣ ─── Encode categorical targets & features ───────────────────────
le_gender = LabelEncoder()
df["GENDER"]      = le_gender.fit_transform(df["GENDER"])      # M/F → 1/0
le_lc = LabelEncoder()
df["LUNG_CANCER"] = le_lc.fit_transform(df["LUNG_CANCER"])     # YES/NO → 1/0

# 4️⃣ ─── Train / test split & model fit ──────────────────────────────
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ ─── Evaluation ──────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print("Classif. report\n", classification_report(y_test, y_pred))

# 6️⃣ ─── Generate & save feature-risk charts ─────────────────────────
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=col, y="LUNG_CANCER", data=df, ci=None)
    plt.title(f"{col} vs. Lung-Cancer Risk (mean of label)")
    plt.ylabel("Mean of LUNG_CANCER (1 = cancer)")
    plt.tight_layout()
    fname = f"{OUT_DIR}/{col.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Chart saved: {fname}")
# Extract coefficients and match them with feature names
coefficients = model.coef_[0]
features = X.columns

# Create a DataFrame for easier viewing
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'AbsCoefficient': np.abs(coefficients)
})

# Sort by absolute value of coefficients
top_features = coef_df.sort_values(by='AbsCoefficient', ascending=False).head(4)
print("Top 4 most influential features on lung cancer risk:\n", top_features)

"""
Run it from a terminal:

    python lung_cancer_logreg_and_charts.py

The console will show accuracy &-report, and PNG files will appear
in the ./charts directory, one per attribute.
"""
