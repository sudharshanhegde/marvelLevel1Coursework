import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the Linear Regression Class
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Adding bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1 for bias
        # Calculating the optimal parameters (theta) using the Normal Equation
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X):
        return self.intercept_ + X.dot(self.coef_)

# Step 2: Generate Sample Data
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + Gaussian noise

# Step 3: Train the Linear Regression model from scratch
model = LinearRegression()
model.fit(X, y)

# Step 4: Make Predictions
X_new = np.array([[0], [2]])  # New input for predictions
y_predict = model.predict(X_new)

# Step 5: Plot the Results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='Linear Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression from Scratch')
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression as SKLinearRegression

# Step 6: Using Scikit-Learn for Linear Regression
sk_model = SKLinearRegression()
sk_model.fit(X, y)

# Step 7: Make Predictions with Scikit-Learn
y_sk_predict = sk_model.predict(X_new)

# Step 8: Plot the Results with Scikit-Learn's Predictions
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='From Scratch Prediction')
plt.plot(X_new, y_sk_predict, color='green', linestyle='--', label='Scikit-Learn Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Comparison')
plt.legend()
plt.show()

# Step 9: Print Coefficients
print("Manual Coefficients:", model.coef_)
print("Manual Intercept:", model.intercept_)
print("Scikit-Learn Coefficients:", sk_model.coef_)
print("Scikit-Learn Intercept:", sk_model.intercept_)
