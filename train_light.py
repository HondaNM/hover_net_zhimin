import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data_path = 'filtered_features.csv'
data = pd.read_csv(data_path)

# Prepare the dataset
X = data.drop('Extracted_Label', axis=1)
y = data['Extracted_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LightGBM data containers
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the model with default configurations
params = {
    'objective': 'binary'
}
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)

# Save the model
model_path = '/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/result/lightGBM/lightgbm_model.pkl'
joblib.dump(bst, model_path)

# Alternatively, for a custom plot and to get feature importance in a DataFrame:
importances = bst.feature_importance(importance_type='split')
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

# Print feature importances
print(feature_importance_df)

# Plotting the top 20 features
plt.figure(figsize=(30, 15))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(20))
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig('top20_feature_importance_lightgbm.png', dpi=300)  # Save the figure
plt.show()

# Print accuracy for reference
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")
