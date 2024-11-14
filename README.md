# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data: Load the Iris dataset, create a DataFrame with features and target labels, and display the first few rows.
2. Define Features and Target: Set X as all feature columns and y as the target column.
3. Split Data: Split X and y into training and testing sets with an 80/20 split.
4. Initialize Model: Create an SGDClassifier with a maximum iteration of 1000 and tolerance of 1e-3.
5. Train Model: Fit the SGD classifier on the training data (X_train, y_train).
6. Evaluate Model: Predict on X_test, calculate accuracy, and print the confusion matrix

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: KALAISELVAN J
RegisterNumber:  212223080022
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:

![image](https://github.com/user-attachments/assets/beeede28-eb80-4088-bb31-30d5294cb85d)


![image](https://github.com/user-attachments/assets/a58826f6-d3ab-4e40-ad63-d581bf93527d)


![image](https://github.com/user-attachments/assets/59a53f3f-366a-4259-a79d-3f672048f48f)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
