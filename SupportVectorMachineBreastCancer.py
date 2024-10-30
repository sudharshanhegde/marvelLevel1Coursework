import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the Breast Cancer Wisconsin dataset
bc_data = datasets.load_breast_cancer()
X = bc_data.data
y = bc_data.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=bc_data.feature_names)
df['target'] = y

# Display the first few rows of the DataFrame
print(df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM model with a linear kernel
model = SVC(kernel='linear')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the results (using 2 features for a simple 2D plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_test, style=y_pred, palette='Set1')
plt.title('SVM Classification of Breast Cancer Dataset')
plt.xlabel(bc_data.feature_names[0])
plt.ylabel(bc_data.feature_names[1])
plt.legend(title='Actual vs Predicted')
plt.show()
