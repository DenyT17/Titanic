import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Function import *
import joblib
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, decomposition

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
test.drop(['PassengerId'], axis=1, inplace=True)
columns = test.columns



# Droping unnecessary columns
train.drop(['Name', 'Ticket', 'PassengerId','Cabin','Age Range'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encoding data
encode_data(train)
encode_data(test)

# Rescaling Data
rescaling(train)
rescaling(test)


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
        LinearDiscriminantAnalysis(),
]
for model in models:
        training_models(model,train_x,train_y,val_x,val_y)

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


# Predict Test data
LR = LogisticRegression(max_iter=500)
GBC = GradientBoostingClassifier()
LDA = LinearDiscriminantAnalysis()


LR_model = LR.fit(train_x,train_y)
GBC_model = GBC.fit(train_x,train_y)
LDA_model = LDA.fit(train_x,train_y)

#Save model
filename = "LR_model.joblib"
joblib.dump(LR_model, filename)
filename = "GBC_model.joblib"
joblib.dump(GBC_model, filename)
filename = "LDA_model.joblib"
joblib.dump(LDA_model, filename)

LR_pred = LR_model.predict(test)
GBC_pred = GBC_model.predict(test)
LDA_pred = LDA_model.predict(test)


test_models = ['LogisticRegression','GradientBoostingClassifier','LinearDiscriminantAnalysis']
Prediction_accuracy=[]
Prediction_accuracy.append(accuracy_score(LR_pred , test_output))
Prediction_accuracy.append(accuracy_score(GBC_pred , test_output))
Prediction_accuracy.append(accuracy_score(LDA_pred , test_output))


fig = plt.figure(figsize=(10,10))
ax = sns.barplot(x=test_models,y=Prediction_accuracy)
ax.bar_label(ax.containers[0])
plt.title('Prediction accuracy',fontsize=25)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Predict survival user defined passenge

passanger = get_passanger(columns)
LR_pred = LR_model.predict_proba(passanger)
GBC_pred = GBC_model.predict_proba(passanger)
LDA_pred = LDA_model.predict_proba(passanger)

print("According to {0} is {1} % probability that passanger would die, and {2} % probability that passanger would survive".format(test_models[0],LR_pred[:,0],LR_pred[:,1]))
print("According to {0} is {1} % probability that passanger would die, and {2} % probability that passanger would survive".format(test_models[1],GBC_pred[:,0],GBC_pred[:,1]))
print("According to {0} is {1} % probability that passanger would die, and {2} % probability that passanger would survive".format(test_models[2],LDA_pred[:,0],LDA_pred[:,1]))

LR_model = joblib.load("LR_model.joblib")
GBC_model = joblib.load("GBC_model.joblib")
LDA_model = joblib.load("LDA_model.joblib")

gb = GradientBoostingClassifier()
lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()


