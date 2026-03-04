Overview

This project is a machine learning classification project using the Iris Flower dataset, a classic dataset in data science and ML. The aim is to predict the species of iris flowers based on their morphological features: sepal length, sepal width, petal length, and petal width. It uses Support Vector Machine for the classification.

Dataset Source: https://www.kaggle.com/datasets/arshid/iris-flower-dataset

The Iris dataset is a classic dataset in machine learning and statistics. It consists of 150 samples with four features each:
1. sepal length (cm)
2. sepal width (cm)
3. petal length (cm)
4. petal width (cm)

The goal of this project is to predict the species of an Iris flower given these features using an SVM classifier.

Libraries Used- 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Exploratory Data Analysis (EDA):
1. Checked for null values (none found).
2. Explored dataset distribution using value counts: all classes are equally represented.
Visualized feature relationships using:
1. sns.FacetGrid
2. sns.pairplot
This helps in understanding feature correlations and separability among species.

We started by loading the dataset and checking for missing values, ensuring data quality. The column names were simplified for convenience. Using statistical descriptions and visualizations like pair plots and scatter plots, we explored relationships among features and observed that petal measurements are more distinctive across species than sepal measurements. We also verified that the dataset is balanced, with 50 samples per species, making it suitable for classification.

Next, we separated the features and target labels, and split the data into training and testing sets in an 80-20 ratio. We trained a Support Vector Classifier (SVC) on the training data. SVM is an effective algorithm for small, well-separated datasets because it finds an optimal hyperplane that separates classes.

After training, we evaluated the model on the test data. The classifier achieved an accuracy of 93.33%, indicating strong performance in distinguishing between the three Iris species. We also tested the model on new sample measurements, and it correctly predicted their species, demonstrating its generalization capability.

From this project, we observed that petal length and width are the most important features for classification. The project highlights the effectiveness of SVM in multi-class classification. In future improvements, we could experiment with hyperparameter tuning, other classification algorithms, or even create a web application for real-time flower prediction.
