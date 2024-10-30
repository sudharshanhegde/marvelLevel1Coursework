import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (replace this with actual data if available)
data = {
    'SquareFeet': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 5, 20, 15, 10],
    'Bathrooms': [2, 3, 2, 4, 3],
    'Price': [300000, 400000, 350000, 500000, 450000]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['SquareFeet', 'Bedrooms', 'Age', 'Bathrooms']]
y = df['Price']

# Split the dataset into training and testing sets with a larger training portion to handle small data size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Example of predicting for a new house with feature names to avoid the feature name warning
new_house = pd.DataFrame([[2500, 4, 5, 3]], columns=['SquareFeet', 'Bedrooms', 'Age', 'Bathrooms'])
predicted_price = model.predict(new_house)
print("Predicted Price for New House:", predicted_price[0])
