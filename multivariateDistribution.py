import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Iris dataset and create a DataFrame
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = iris_data.target
iris['species'] = iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Pairplot to visualize multivariate distributions
plt.figure(figsize=(10, 8))
sns.pairplot(iris, hue="species", diag_kind="kde", markers=["o", "s", "D"], palette="Set2")
plt.suptitle("Multivariate Distribution Plot for Iris Dataset", y=1.02)
plt.show()
