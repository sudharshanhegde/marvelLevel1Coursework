# MARVEL LEVEL 1



### Task 1: Linear Regression - Predicting Home Prices

In this task, I used linear regression to predict home prices based on various factors like location, size, and number of bedrooms. By using `scikit-learn`'s `linear_model.LinearRegression()` function, I built a model that could analyze these variables and output price predictions. The process included preparing the dataset—cleaning up missing values and selecting important features—to ensure accuracy. Ultimately, I was able to train the model and make realistic predictions on housing prices, gaining practical experience with linear regression techniques in Python.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/linearRegression.py)

### Task 2: 

**Subtask 1:Logistic Regression - Iris Species Classification**

In this task, I trained a logistic regression model to classify different species of the Iris flower based on features such as sepal length, sepal width, petal length, and petal width. Using `scikit-learn`'s `linear_model.LogisticRegression`, I built and trained the model on the famous Iris dataset, which provided labeled data for each flower species. After training, the model could successfully distinguish between species with a high accuracy. This exercise helped deepen my understanding of logistic regression for classification tasks in Python.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/logisticRegression.py)

**Subtask 2: Plotting Basics - Exploring Plot Characteristics**

In this task, I explored various fundamental characteristics of plotting using Python libraries, primarily `matplotlib`. I learned how to set axis labels and define axis limits to improve plot readability. Additionally, I created figures with multiple subplots to display various datasets side-by-side, added legends to distinguish between different data series, and saved the final plots as PNG files. This task provided a solid foundation in essential plotting techniques, which are crucial for data visualization and analysis.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/plot.py)
![basics of plotting](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_5-1.png)


**Subtask 3: Plot Types Exploration**

In this task, I explored a variety of plot types using Python's visualization libraries, primarily `matplotlib` and `seaborn`. This included creating line and area plots, scatter and bubble plots with the Iris dataset, and various bar plots (simple, grouped, and stacked) to represent categorical data. I also worked with histograms to analyze distributions, pie charts for proportional data, and box and violin plots to visualize data spread and variability.

Additionally, I explored more advanced visualization types, such as marginal plots, contour plots, heatmaps, and 3D plots, each offering unique insights for different data structures. This exercise helped me build a comprehensive toolkit for visualizing data in Python, tailored to different analytical needs.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/differentplots.py)
![bubble plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_9-1.png)
![bar plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_10-1.png)
![histogram](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_11-1.png)
![box plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_12-1.png)
![violin plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_13-1.png)
![marginal plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_14-1.png)
![contour plot](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_15-1.png)
![heat map](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_16-1.png)


**Subtask 4: Multivariate Distribution and Clustering**

In this task, I created a multivariate distribution plot using the given dataset to facilitate a classification task. This involved visualizing the relationships between multiple variables and understanding how they interact with one another. I employed techniques such as pair plots and contour plots to identify potential clusters within the data. 

This exploration provided an elementary understanding of clustering concepts, which will be further examined in future tasks. The insights gained from this analysis are crucial for determining how to group data points based on their features effectively.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/multivariateDistribution.py) 
![multivariate distribution](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/output_image_18-1.png)

### Task 3: NumPy Array Generation

In this task, I utilized NumPy to generate arrays with specific characteristics. First, I created an array by repeating a smaller array across each dimension, demonstrating how to manipulate array shapes and sizes effectively. This allowed for a deeper understanding of multidimensional arrays and their applications.

Additionally, I generated an array containing element indexes, ensuring that the array elements appeared in ascending order. This exercise reinforced my knowledge of array indexing and the importance of ordering in data analysis, showcasing NumPy's powerful capabilities for numerical computations.       
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/numpySmallArray.py) 

### Task 4:

**Subtask 1: Regression Metrics - Evaluating Algorithm Performance**

In this task, I focused on understanding and applying various regression metrics used to evaluate the performance of regression algorithms. I explored metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²), which are essential for assessing how well a model predicts continuous outcomes. 

By calculating these metrics for my regression models, I gained insights into their accuracy and effectiveness, allowing me to compare different models systematically. This knowledge is crucial for selecting the best-performing model for a given dataset.

[ code ](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/regressionMetrics.py) 


**Subtask 2: Classification Metrics - Evaluating Algorithm Performance**

In this task, I explored various classification metrics used to evaluate the performance of classification algorithms. Key metrics included Accuracy, Precision, Recall, F1 Score, and the Confusion Matrix. Understanding these metrics is essential for assessing how well a model distinguishes between different classes, especially in imbalanced datasets.

By calculating these metrics for my classification models, I was able to analyze their effectiveness and make informed decisions about model selection and tuning. This experience reinforced the importance of using appropriate metrics for evaluating classification tasks in machine learning.
[code ](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/classificationMetrics.py) 

### Task 5: Implementing K-Nearest Neighbors (KNN)

In this task, I implemented the K-Nearest Neighbors (KNN) algorithm using `scikit-learn`'s `neighbors.KNeighborsClassifier` on multiple suitable datasets. This hands-on experience allowed me to understand the algorithm's mechanics, including how it classifies data points based on the majority class of their nearest neighbors.

Additionally, I implemented KNN from scratch to deepen my understanding of the algorithm's underlying principles. By comparing the results of my custom implementation with those from `scikit-learn`, I evaluated the performance across different datasets. This exercise highlighted the efficiency and effectiveness of KNN, while also reinforcing the importance of choosing the right approach for specific classification tasks.
[code ](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/KNNwithSciKitComparision.py)

### Task 6: Data Visualization with Plotly

In this task, I utilized Plotly for data visualization, exploring its advanced features that make it more dynamic than traditional libraries like Matplotlib or Seaborn. I created interactive plots that allowed for greater user engagement and deeper insights into the data. 

Using Plotly's capabilities, I was able to produce a variety of visualizations, including scatter plots, line charts, and 3D plots, each enhancing the storytelling aspect of the data. This experience not only improved my skills in creating visually appealing graphics but also emphasized the importance of interactive data exploration in analytical projects.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/plotly.py) 


### Task 7: Decision Tree - Supervised Learning Algorithm

In this task, I explored the Decision Tree algorithm, a powerful supervised learning technique used for both regression and classification tasks. I learned how decision trees utilize a hierarchy of conditional statements to make predictions, effectively breaking down complex decision-making processes into simpler, understandable components. 

By implementing decision trees, I was able to visualize the model's structure, which provided clear insights into how it arrived at specific outcomes based on input features. This task enhanced my understanding of tree-based methods and their practical applications in real-world data analysis scenarios.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/decisionTree.py) 

### Task 8: 
**Support Vector Machines - Understanding SVMs**

In this task, I delved into Support Vector Machines (SVM), a supervised learning method used to create a non-probabilistic linear model for binary classification tasks. SVMs work by assigning data points to one of two classes while maximizing the margin between them. 

I learned that each data point is represented as a vector, and the goal is to identify the optimal hyperplane that separates the two classes. This hyperplane is chosen to maximize the distance between it and the nearest data points from both classes, effectively regularizing the loss. This task deepened my understanding of how SVMs can effectively classify data in various applications.

**Subtask 1: Breast Cancer Detection using Support Vector Machines**

In this task, I focused on the critical issue of breast cancer diagnosis, utilizing Support Vector Machines (SVM) to detect the possibility of breast cancer in patients. Understanding the importance of timely and accurate diagnosis in the medical field, I applied SVM techniques to analyze relevant datasets containing features indicative of cancerous conditions.

By training the SVM model, I aimed to classify data points as benign or malignant, maximizing the margin between the two classes for optimal decision-making. This task not only enhanced my skills in applying machine learning algorithms to healthcare problems but also underscored the potential of SVMs in improving diagnostic accuracy for life-threatening diseases like breast cancer.
[code](https://github.com/sudharshanhegde/marvelLevel1Coursework/blob/main/SupportVectorMachineBreastCancer.py)








