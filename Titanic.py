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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
# Uploading data and creatin targets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_output = pd.read_csv('gender_submission.csv')
test_output = test_output['Survived']
# Checking information about training dataset
print(train.info(),train.isnull().sum())

# Filling empty cells
train = fillempty(train)
test = fillempty(test)

# Analisi training data
data_analisis(train)

# Creating targets
train_features = train.pop('Survived')

# Droping unnecessary columns
train.drop(['Name', 'Ticket', 'PassengerId','Age Range'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Encoding data
encode_data(train)

# Spliting data
train_x,val_x,train_y,val_y = train_test_split(train,train_features,test_size=.30,random_state=42)

# Training models
models = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(random_state=11),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(max_iter=500),
        LinearDiscriminantAnalysis()
]
training_models(models,train_x,train_y,val_x,val_y)

# Cross validation
print('\n\n Cross validation ')
Average_Accuracy=[]
Models_name=[]
for model in models:
        Models_name.append(model.__class__.__name__)
        Average_Accuracy.append(cross_validation(model,train,train_features,5))

fig = plt.figure(figsize=(10,10))
ax = sns.barplot(x=Models_name,y=Average_Accuracy)
ax.bar_label(ax.containers[0])
plt.title('Cross-validation average accuracy',fontsize=25)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Predict Test data
print('\n\n Predict Test data ')
Prediction_accuracy = []
for model in models:
        Prediction_accuracy.append(prediction(model,test,test_output))

fig = plt.figure(figsize=(10,10))
ax = sns.barplot(x=Models_name,y=Prediction_accuracy)
ax.bar_label(ax.containers[0])
plt.title('Prediction accuracy',fontsize=25)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





































