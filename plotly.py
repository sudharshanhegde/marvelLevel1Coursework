import pandas as pd
import plotly.express as px

# Load the Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Display the first few rows of the dataset
print(iris.head())

# --- 1. Scatter Plot ---
fig_scatter = px.scatter(iris, x='sepal_length', y='sepal_width', color='species',
                         title='Sepal Length vs Sepal Width',
                         labels={'sepal_length': 'Sepal Length (cm)', 'sepal_width': 'Sepal Width (cm)'})
fig_scatter.show()

# --- 2. Box Plot ---
fig_box = px.box(iris, x='species', y='petal_length', color='species',
                 title='Box Plot of Petal Length by Species',
                 labels={'petal_length': 'Petal Length (cm)', 'species': 'Species'})
fig_box.show()

# --- 3. Histogram ---
fig_histogram = px.histogram(iris, x='petal_length', color='species', barmode='overlay',
                              title='Histogram of Petal Length by Species',
                              labels={'petal_length': 'Petal Length (cm)', 'species': 'Species'})
fig_histogram.show()

# --- 4. Pair Plot ---
fig_pair = px.scatter_matrix(iris, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                              color='species', title='Scatter Matrix of Iris Features',
                              labels={'sepal_length': 'Sepal Length (cm)', 'sepal_width': 'Sepal Width (cm)',
                                      'petal_length': 'Petal Length (cm)', 'petal_width': 'Petal Width (cm)'})
fig_pair.show()

# --- 5. Violin Plot ---
fig_violin = px.violin(iris, y='petal_length', x='species', color='species', box=True, points='all',
                        title='Violin Plot of Petal Length by Species',
                        labels={'petal_length': 'Petal Length (cm)', 'species': 'Species'})
fig_violin.show()
