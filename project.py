import pandas as pd #Genererate data from excel 
import statsmodels.api as sm #Build logistic model
from sklearn.preprocessing import LabelEncoder #Encode 

#Read Data
df = pd.read_csv("survey_lung_cancer.csv")

#Encode
label_enc = LabelEncoder()
df['GENDER'] = label_enc.fit_transform(df['GENDER'])  # M:1, F:0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})


target = 'LUNG_CANCER'
numeric_cols = [col for col in df.columns if col != target]

# Save values which have meaning in statistics-
p_values_list = []

# Univariate logistic regression
for col in numeric_cols:
    try:
        X = df[[col]].dropna()
        X = sm.add_constant(X)
        y = df[target].loc[X.index]
        
        model = sm.Logit(y, X).fit(disp=0)
        p_value = model.pvalues[col]

        print(f"{col}: P-value = {p_value:.4f}")
        
        if p_value < 0.01:
            p_values_list.append((col, p_value))

    except Exception as e:
        print(f"{col}: Lỗi - {e}")

# Take top 4
top4_vars = sorted(p_values_list, key=lambda x: x[1])[:4]

print("\n✅ Top 4:")
for var, pval in top4_vars:
    print(f"{var}: P-value = {pval:.4e}")
