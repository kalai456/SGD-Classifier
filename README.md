# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Iris dataset and split into features (X) and target (y).

2.Standardize features and split into training and testing sets

3.Set up SGDClassifier with desired hyperparameters.

4.Fit the classifier to the training data.

5.Use the model to predict species on the test set.

6.Measure accuracy and other metrics to assess performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: KALAISELVAN J
RegisterNumber:  212223080022
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd=SGDClassifier(max_iter=1000,tol=1e-3)
sgd.fit(x_train,y_train)

y_pred=sgd.predict(x_test)
print("Prediction:\n",y_pred)

accu=accuracy_score(y_test,y_pred)
print("Accuracy:\n",accu)

confu=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confu)

report=classification_report(y_test,y_pred)
print("classification_report:\n",report)
```

## Output:
![image](https://github.com/user-attachments/assets/6e04323e-c036-4cd5-b050-a010c3ea43e9)

![image](https://github.com/user-attachments/assets/53c565af-d46c-407e-8ddc-0fbe864de797)

![image](https://github.com/user-attachments/assets/f30fee91-20bd-4110-a5c2-0641ac4dcc6d)

![image](https://github.com/user-attachments/assets/1b7a830f-c65a-43ce-a747-2d49f32a44fc)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
