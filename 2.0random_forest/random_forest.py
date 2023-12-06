import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the CSV file
file_path = 'house_van2.0.csv'
df_house = pd.read_csv(file_path)

# Convert 'List Date' to datetime if necessary
df_house['List Date'] = pd.to_datetime(df_house['List Date'], errors='coerce')

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

# Creating and training the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_pred = rf_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Outputting the evaluation metrics
print("Random Forest Model Evaluation:")
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
