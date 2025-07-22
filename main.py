import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load your CSV file (make sure it's in the same folder or give full path)
df = pd.read_csv("employee_data.csv")

# Features (input) and Target (output)
X = df.drop('Salary', axis=1)
y = df['Salary']

# Identify column types
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include='int64').columns.tolist()

# Preprocessing: Encode categorical data
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Full Pipeline: Preprocessing + Random Forest
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate performance
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model trained successfully ✅\nRMSE: {rmse:.2f}")

# Save the model
joblib.dump(pipeline, "employee_salary_model.pkl")
print("Model saved as 'employee_salary_model.pkl'")
