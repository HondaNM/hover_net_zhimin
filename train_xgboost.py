import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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