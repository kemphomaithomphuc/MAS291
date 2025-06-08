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
