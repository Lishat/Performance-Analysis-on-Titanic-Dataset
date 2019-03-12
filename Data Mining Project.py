#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[ ]:


data_train = pd.read_csv("./train.csv")
data_test = pd.read_csv("./test.csv")


# In[ ]:


data_train.head()


# In[ ]:


data_train = data_train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
data_test = data_test.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])


# In[ ]:


data_train = pd.concat([data_train.drop(columns=["Sex"]),pd.get_dummies(data_train["Sex"], drop_first=True)], axis=1)
data_test = pd.concat([data_test.drop(columns=["Sex"]),pd.get_dummies(data_test["Sex"], drop_first=True)], axis=1)


# In[ ]:


data_train = pd.concat([data_train.drop(columns=["Embarked"]),pd.get_dummies(data_train["Embarked"], drop_first=True)], axis=1)
data_test = pd.concat([data_test.drop(columns=["Embarked"]),pd.get_dummies(data_test["Embarked"], drop_first=True)], axis=1)


# In[ ]:


data_train = pd.concat([data_train.drop(columns=["Pclass"]),pd.get_dummies(data_train["Pclass"], drop_first=True)], axis=1)
data_test = pd.concat([data_test.drop(columns=["Pclass"]),pd.get_dummies(data_test["Pclass"], drop_first=True)], axis=1)


# In[ ]:


mean=np.mean(data_train['Age'])
data_train['Age']=data_train['Age'].fillna(mean)
data_test['Age']=data_test['Age'].fillna(mean)


# In[ ]:


data_train.corr(method="pearson")


# In[ ]:


y_train = data_train["Survived"]


# In[ ]:


x_train = data_train.drop("Survived", axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)


# In[ ]:


classifier1 = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
classifier1.fit(x_train, y_train)
print(classification_report(y_train, classifier1.predict(x_train)))
print(confusion_matrix(y_train, classifier1.predict(x_train)))
print(accuracy_score(y_train, classifier1.predict(x_train)))


# In[ ]:


print(classification_report(y_test, classifier1.predict(x_test)))
print(confusion_matrix(y_test, classifier1.predict(x_test)))
print(accuracy_score(y_test, classifier1.predict(x_test)))


# In[ ]:


SVC()


# In[ ]:


classifier2 = RandomForestClassifier()
classifier2.fit(x_train, y_train)
print(classification_report(y_train, classifier2.predict(x_train)))
print(confusion_matrix(y_train, classifier2.predict(x_train)))
print(accuracy_score(y_train, classifier2.predict(x_train)))
print(classification_report(y_test, classifier2.predict(x_test)))
print(confusion_matrix(y_test, classifier2.predict(x_test)))
print(accuracy_score(y_test, classifier2.predict(x_test)))


# In[ ]:


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
classifier3 = GridSearchCV(SVC(gamma='scale'), parameters, cv=2)
classifier3.fit(x_train, y_train)
print(classification_report(y_train, classifier3.predict(x_train)))
print(confusion_matrix(y_train, classifier3.predict(x_train)))
print(accuracy_score(y_train, classifier3.predict(x_train)))
print(classification_report(y_test, classifier3.predict(x_test)))
print(confusion_matrix(y_test, classifier3.predict(x_test)))
print(accuracy_score(y_test, classifier3.predict(x_test)))


# In[ ]:


classifier4 = XGBClassifier()
classifier4.fit(x_train, y_train)
print(classification_report(y_train, classifier4.predict(x_train)))
print(confusion_matrix(y_train, classifier4.predict(x_train)))
print(accuracy_score(y_train, classifier4.predict(x_train)))
print(classification_report(y_test, classifier4.predict(x_test)))
print(confusion_matrix(y_test, classifier4.predict(x_test)))
print(accuracy_score(y_test, classifier4.predict(x_test)))


# In[ ]:




