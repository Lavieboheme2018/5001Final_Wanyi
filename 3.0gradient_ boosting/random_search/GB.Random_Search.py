import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint as sp_randint

# Load the CSV file
file_path = 'house_van3.0.csv'
df_house = pd.read_csv(file_path)

# Selecting relevant features for comparison and model
features = ['Total floor area', 'Age', 'Lot Size']
target = 'Price'

# Create comparison plots
plt.figure(figsize=(15, 10))
for i, column in enumerate(features, 1):
    plt.subplot(2, 2, i)
    plt.scatter(df_house[column], df_house[target])
    plt.xlabel(column)
    plt.ylabel('Price')
    if column != 'List Date':
        fit = np.polyfit(df_house[column], df_house[target], 1)
        fit_fn = np.poly1d(fit)
        plt.plot(df_house[column], fit_fn(df_house[column]), 'r--')
    plt.title(f'Price vs {column}')

# Preprocessing: Handle missing values if any
df_house = df_house.dropna(subset=features + [target])

# Splitting the dataset into training and testing sets
X = df_house[features]
y = df_house[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Search Cross-Validation
param_dist = {
    "n_estimators": sp_randint(100, 500),
    "max_depth": sp_randint(3, 10),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "learning_rate": [0.01, 0.05, 0.1, 0.2]
}
random_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_distributions=param_dist,
                                   n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Use the best parameters for the Gradient Boosting model
best_gb_model = random_search.best_estimator_
y_pred = best_gb_model.predict(X_test)

# Evaluating the optimized model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Outputting the evaluation metrics
print("Optimized Gradient Boosting Model Evaluation:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2 Score:", r2)

# Visualization of actual vs predicted values
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
