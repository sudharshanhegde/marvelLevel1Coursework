import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the Logistic Regression Class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.coef_) + self.intercept_
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# Step 2: Generate Sample Data (Binary Classification)
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Class 1 if sum of features > 1, else Class 0

# Step 3: Train the Logistic Regression model from scratch
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Step 4: Make Predictions
y_pred = model.predict(X)

# Step 5: Visualize the Results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Logistic Regression from Scratch')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import accuracy_score

# Step 6: Using Scikit-Learn for Logistic Regression
sk_model = SKLogisticRegression()
sk_model.fit(X, y)

# Step 7: Make Predictions with Scikit-Learn
y_sk_pred = sk_model.predict(X)

# Step 8: Calculate Accuracy
accuracy_manual = accuracy_score(y, y_pred)
accuracy_sklearn = accuracy_score(y, y_sk_pred)

print("Manual Logistic Regression Accuracy:", accuracy_manual)
print("Scikit-Learn Logistic Regression Accuracy:", accuracy_sklearn)

# Step 9: Visualize Scikit-Learn's Predictions
plt.scatter(X[:, 0], X[:, 1], c=y_sk_pred, cmap='viridis', edgecolor='k', s=50, alpha=0.5)
plt.title('Scikit-Learn Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
