import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import dump, load

# Save the model to a file
model_path = '/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/result/xgboost/xgboost_model.joblib'
# Load the dataset from the CSV file
data_path = 'filtered_features.csv'  # Update this to your CSV file path
data = pd.read_csv(data_path)

# Separate features and label
X = data.drop('Extracted_Label', axis=1)
y = data['Extracted_Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier with default parameters
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
xgb_clf.fit(X_train, y_train)
dump(xgb_clf, model_path)

# Make predictions
y_pred = xgb_clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Extract feature importances
feature_importances = xgb_clf.feature_importances_
features = X.columns
importances = list(zip(features, feature_importances))
importances.sort(key=lambda x: x[1], reverse=True)

# Print feature importances
for feature, importance in importances:
    print(f"{feature}: {importance}")



# Create a DataFrame for easier plotting
importance_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Features'][:20], importance_df['Importance'][:20])  # Top 10 features
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()  # To have the most important feature on top
plt.savefig('/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/result/xgboost/feature_importance_plot.png', dpi=300)  # Saves the figure to the file 'feature_importance_plot.png' with a resolution of 300 DPI
plt.show()