import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data_path = r"C:\Users\DELL\Desktop\Housing.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(data_path)

# Define columns
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Separate features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessors on training data
imputer_num = SimpleImputer(strategy='mean')
scaler = StandardScaler()
imputer_cat = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Apply preprocessors to training data
X_train_num = imputer_num.fit_transform(X_train[numerical_cols])
X_train_num = scaler.fit_transform(X_train_num)
X_train_cat = imputer_cat.fit_transform(X_train[categorical_cols])
X_train_cat = encoder.fit_transform(X_train_cat)

# Combine preprocessed numerical and categorical features
X_train_preprocessed = np.hstack((X_train_num, X_train_cat))

# Apply preprocessors to test data
X_test_num = imputer_num.transform(X_test[numerical_cols])
X_test_num = scaler.transform(X_test_num)
X_test_cat = imputer_cat.transform(X_test[categorical_cols])
X_test_cat = encoder.transform(X_test_cat)

# Combine preprocessed numerical and categorical features for test data
X_test_preprocessed = np.hstack((X_test_num, X_test_cat))

# Save the preprocessors
with open('imputer_num.pkl', 'wb') as f:
    pickle.dump(imputer_num, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('imputer_cat.pkl', 'wb') as f:
    pickle.dump(imputer_cat, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Train and evaluate Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_preprocessed, y_train)
y_pred_lr = lin_reg.predict(X_test_preprocessed)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print(f'Linear Regression RMSE: {rmse_lr}')

# Hyperparameter tuning for Ridge Regression
ridge = Ridge()
param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train_preprocessed, y_train)
best_ridge = grid_search_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_preprocessed)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f'Ridge Regression RMSE: {rmse_ridge}')

# Hyperparameter tuning for Lasso Regression
lasso = Lasso()
param_grid_lasso = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train_preprocessed, y_train)
best_lasso = grid_search_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(X_test_preprocessed)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f'Lasso Regression RMSE: {rmse_lasso}')

# Hyperparameter tuning for Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_preprocessed, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_preprocessed)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Random Forest RMSE: {rmse_rf}')

# Hyperparameter tuning for Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
param_grid_gb = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train_preprocessed, y_train)
best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test_preprocessed)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
print(f'Gradient Boosting RMSE: {rmse_gb}')

# Train and evaluate ANN model
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

ann.compile(optimizer='adam', loss='mse')
ann.fit(X_train_preprocessed, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

y_pred_ann = ann.predict(X_test_preprocessed)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
print(f'ANN RMSE: {rmse_ann}')

# Compare RMSE of all models
print(f'Linear Regression RMSE: {rmse_lr}')
print(f'Ridge Regression RMSE: {rmse_ridge}')
print(f'Lasso Regression RMSE: {rmse_lasso}')
print(f'Random Forest RMSE: {rmse_rf}')
print(f'Gradient Boosting RMSE: {rmse_gb}')
print(f'ANN RMSE: {rmse_ann}')

# Choose the best model based on RMSE
best_model = min(
    [(lin_reg, rmse_lr), (best_ridge, rmse_ridge), (best_lasso, rmse_lasso), (best_rf, rmse_rf), (best_gb, rmse_gb), (ann, rmse_ann)],
    key=lambda x: x[1]
)[0]

# Save the best model
with open('best_model.pkl', 'wb') as file:
    if isinstance(best_model, Sequential):
        # Save TensorFlow model separately as pickle does not support saving Keras models directly
        best_model.save('best_model_tf.h5')
        pickle.dump('best_model_tf.h5', file)
    else:
        pickle.dump(best_model, file)

print("Best model saved to best_model.pkl")
