# -*- coding: utf-8 -*-
"""7_LeafSpeciesDetection_DECISIONTREE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z9FDLyMtpxjr-SBDryDWVrQ_3ftM4SE2

# **Day-5 | Leaf Species Detection | DECISION TREE**

### *Import basic Libraries*
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

"""### *Load Dataset*"""

dataset = load_iris()

"""### *Summarize Dataset*"""

print(dataset.data)
print(dataset.target)

print(dataset.data.shape)

"""### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*"""

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X

Y = dataset.target
Y

"""### *Splitting Dataset into Train & Test*"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print(X_train.shape)
print(X_test.shape)

"""### *Finding best max_depth Value*"""

accuracy = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1, 10):
    model = DecisionTreeClassifier(max_depth = i, random_state = 0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')

"""### *Training*"""

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3, random_state = 0)
model.fit(X_train,y_train)

"""### *Prediction*"""

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""### *Accuracy Score*"""

from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))