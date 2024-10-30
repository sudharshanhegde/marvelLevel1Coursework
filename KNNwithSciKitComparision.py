import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to evaluate the KNN model
def evaluate_knn_model(X_train, X_test, y_train, y_test, n_neighbors=5):
    # Custom KNN Classifier
    class KNN:
        def __init__(self, n_neighbors):
            self.n_neighbors = n_neighbors
            
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            
        def predict(self, X):
            predictions = []
            for x in X:
                # Calculate distances from x to all points in X_train
                distances = np.linalg.norm(self.X_train - x, axis=1)
                # Get the indices of the n nearest neighbors
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
                # Get the most common class
                nearest_classes = self.y_train[nearest_indices]
                predictions.append(np.bincount(nearest_classes).argmax())
            return np.array(predictions)

    # Instantiate the custom KNN model
    custom_knn = KNN(n_neighbors)
    custom_knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred_custom = custom_knn.predict(X_test)
    
    # Instantiate the built-in KNN model
    sk_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    sk_knn.fit(X_train, y_train)
    y_pred_sk = sk_knn.predict(X_test)
    
    # Evaluation
    print(f"Custom KNN Results:")
    print(confusion_matrix(y_test, y_pred_custom))
    print(classification_report(y_test, y_pred_custom))
    print("Custom KNN Accuracy:", accuracy_score(y_test, y_pred_custom))
    
    print(f"\nScikit-Learn KNN Results:")
    print(confusion_matrix(y_test, y_pred_sk))
    print(classification_report(y_test, y_pred_sk))
    print("Scikit-Learn KNN Accuracy:", accuracy_score(y_test, y_pred_sk))

# --- Iris Dataset ---
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
print("Iris Dataset Evaluation")
evaluate_knn_model(X_train_iris, X_test_iris, y_train_iris, y_test_iris, n_neighbors=3)

# --- Wine Dataset ---
wine = load_wine()
X_wine, y_wine = wine.data, wine.target
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)
print("\nWine Dataset Evaluation")
evaluate_knn_model(X_train_wine, X_test_wine, y_train_wine, y_test_wine, n_neighbors=5)

# --- Breast Cancer Dataset ---
breast_cancer = load_breast_cancer()
X_cancer, y_cancer = breast_cancer.data, breast_cancer.target
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)
print("\nBreast Cancer Dataset Evaluation")
evaluate_knn_model(X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer, n_neighbors=5)
