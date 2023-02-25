import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Function import *
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# Uploading data and creatin targets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking information about training dataset
print(train.info,train.isnull().sum())

# Filling empty cells
train = fillempty(train)

# Analisi training data
data_analisis(train)

# Creating targets
train_features = train.pop('Survived')

# Droping unnecessary columns
train.drop(['Name', 'Ticket', 'PassengerId','Age Range'], axis=1, inplace=True)

# Encoding data
encode_data(train)

# Spliting data
train_x,val_x,train_y,val_y = train_test_split(train,train_features,test_size=.33,random_state=42)

# Training models
models = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(random_state=11),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(max_iter=200)
]

training_models(models,train_x,train_y,val_x,val_y)













































