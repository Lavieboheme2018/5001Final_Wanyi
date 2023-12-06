import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, model_selection, metrics, preprocessing

# Load the CSV file
file_path_modified = 'house_van.csv'
df_house = pd.read_csv(file_path_modified)

# Convert 'List Date' to datetime
df_house['List Date'] = pd.to_datetime(df_house['List Date'])
print(df_house.shape)

# Create comparison plots
plt.figure(figsize=(15, 10))

# List of columns to compare with 'Price'
compare_columns = ['List Date', 'Days on market', 'Total floor area', 'Year Built', 'Age', 'Lot Size']

# Plotting each comparison in a subplot
for i, column in enumerate(compare_columns, 1):
    plt.subplot(3, 2, i)
    plt.scatter(df_house[column], df_house['Price'])
    plt.xlabel(column)
    plt.ylabel('Price')
    if column != 'List Date':
        fit = np.polyfit(df_house[column], df_house['Price'], 1)
        fit_x = np.poly1d(fit)
        plt.plot(df_house[column], fit_x(df_house[column]), 'r--')
    plt.title(f'Price vs {column}')

# Adjust layout
plt.tight_layout()

# Display plots
plt.show()
plt.close()

# building models
df_x = df_house[['Total floor area', 'Age', 'Lot Size']]
df_y = df_house['Price']
x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x, df_y, test_size=0.3)

# model training
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)

# Model evaluation
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
print("R^2 Score:", metrics.r2_score(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

# Visualization of actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
