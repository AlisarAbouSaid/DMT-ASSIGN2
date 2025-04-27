import matplotlib
matplotlib.use('TkAgg')  # Change to TkAgg backend

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the preprocessed data from the pickle file
with open('train_preprocessed.pkl', 'rb') as file:
    processed_data, labels = pickle.load(file)

# Check if labels are available (for supervised training)
if labels is None:
    print("No labels found. Exiting.")
    exit(1)

# Now processed_data contains the features and labels contains the target
# Assuming your target is multi-column (e.g., 'click_bool' and 'booking_bool'),
# but let's simplify and use just one for now: 'click_bool'
y = labels['click_bool']  # Change this based on your label preference (e.g., 'click_bool' or 'booking_bool')
X = processed_data

# Ensure no datetime columns are left
X = X.select_dtypes(exclude=['datetime64[ns]'])  # Drop datetime columns if they exist

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',  # Change to 'multi:softmax' for multi-class classification
    eval_metric='logloss',        # You can change to other metrics if needed
    random_state=42
)

# Train the XGBoost model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Confusion matrix for classification tasks
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot feature importance
xgb.plot_importance(model)
plt.show()  # This should now work with the TkAgg backend
