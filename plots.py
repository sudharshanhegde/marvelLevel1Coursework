import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.datasets import load_iris

# Load Iris dataset
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = iris_data.target
iris['species'] = iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Line and Area Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Line Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot - Sine Wave")
plt.legend()
plt.show()

# Area Plot
plt.figure(figsize=(8, 4))
plt.fill_between(x, y, color="skyblue", alpha=0.4)
plt.plot(x, y, color="Slateblue", alpha=0.6, linewidth=2)
plt.title("Area Plot - Sine Wave")
plt.show()

# Scatter and Bubble Plot using Iris dataset
plt.figure(figsize=(8, 6))
sns.scatterplot(x="sepal length (cm)", y="sepal width (cm)", hue="species", data=iris)
plt.title("Scatter Plot - Iris Dataset")
plt.show()

# Bubble Plot (size varies with petal length)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="sepal length (cm)", y="sepal width (cm)", size="petal length (cm)", hue="species", data=iris, sizes=(20, 200), alpha=0.5)
plt.title("Bubble Plot - Iris Dataset")
plt.show()

# Bar Plot
species_counts = iris['species'].value_counts()

# Simple Bar Plot
plt.figure(figsize=(8, 4))
species_counts.plot(kind='bar', color='skyblue')
plt.title("Simple Bar Plot - Species Counts")
plt.show()

# Grouped Bar Plot
plt.figure(figsize=(8, 6))
sns.barplot(x="species", y="sepal length (cm)", hue="species", data=iris)
plt.title("Grouped Bar Plot - Sepal Length by Species")
plt.show()

# Stacked Bar Plot
stacked_data = iris.groupby('species').mean()[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
stacked_data.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
plt.title("Stacked Bar Plot - Average Measurements by Species")
plt.show()

# Histogram
plt.figure(figsize=(8, 4))
sns.histplot(iris["sepal length (cm)"], kde=True, color="purple")
plt.title("Histogram - Sepal Length")
plt.show()

# Pie Plot
plt.figure(figsize=(6, 6))
species_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Pie Plot - Species Distribution")
plt.ylabel('')
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="species", y="sepal length (cm)", data=iris)
plt.title("Box Plot - Sepal Length by Species")
plt.show()

# Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(x="species", y="sepal length (cm)", data=iris)
plt.title("Violin Plot - Sepal Length by Species")
plt.show()

# Marginal Plot
sns.jointplot(x="sepal length (cm)", y="sepal width (cm)", data=iris, kind="scatter", hue="species", marginal_kws=dict(bins=15, fill=True))
plt.suptitle("Marginal Plot - Sepal Dimensions")
plt.show()

# Contour Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(x="sepal length (cm)", y="sepal width (cm)", data=iris, fill=True, cmap="Blues", levels=10)
plt.title("Contour Plot - Sepal Dimensions Density")
plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = iris.iloc[:, :-1].corr()
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.title("Heatmap - Feature Correlation Matrix")
plt.show()

# 3D Plot (using Plotly)
fig = px.scatter_3d(iris, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)',
                    color='species', title="3D Scatter Plot - Iris Dataset")
fig.show()
