import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
def fillempty(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna('Empty', inplace=True)
    data['Fare'].fillna(0, inplace=True)

    return data
def rescaling(data):
    ages_data = np.array(data["Age"]).reshape(-1, 1)
    fares_data = np.array(data["Fare"]).reshape(-1, 1)

    data["Age"] = StandardScaler().fit_transform(ages_data)
    data["Fare"] = StandardScaler().fit_transform(fares_data)

def encode_data(data):
    features = ['Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data


def data_analisis(data):
    male_survived = len(data[(data['Sex'] == 'male') & (data['Survived'] == 1)])
    male_dead = len(data[(data['Sex'] == 'male') & (data['Survived'] == 0)])

    female_survived = len(data[(data['Sex'] == 'female') & (data['Survived'] == 1)])
    female_dead = len(data[(data['Sex'] == 'female') & (data['Survived'] == 0)])

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.pie([male_survived, male_dead], labels=['Survived', 'Dead'], autopct='%1.1f%%')
    plt.legend(bbox_to_anchor=(0.25, -0.25), loc="lower left")
    plt.title("Male")
    plt.subplot(1, 2, 2)
    plt.pie([female_survived, female_dead], labels=['Survived', 'Dead'], autopct='%1.1f%%')
    plt.title("Female")
    fig.suptitle('Survival by gender', fontsize=25)

    fig = plt.figure()
    sns.barplot(data=data,x='Pclass',y='Survived')
    plt.title('Survival by passenger class',fontsize=25)

    def age_range(age):
        category = ''
        if age <= 5:
            category = 'Age < 5'
        elif age <= 15:
            category = '5 < Age < 15'
        elif age <= 25:
            category = '15 < Age < 25'
        elif age <= 35:
            category = '25 < Age < 35'
        elif age <= 45:
            category = '35 < Age < 45'
        elif age <= 55:
            category = '45 < Age < 55'
        elif age <= 65:
            category = '55 < Age < 65'
        elif age <= 100:
            category = '65 < Age < 100'
        return category

    data['Age Range']= data['Age'].apply(lambda x: age_range(x))
    range=['Age < 5','5 < Age < 15','15 < Age < 25','25 < Age < 35','35 < Age < 45','45 < Age < 55','65 < Age < 100']
    fig=plt.figure()
    sns.barplot(data=data,x='Age Range',y='Survived',order=range)
    plt.xticks(rotation=45)
    plt.title('Survival by age',fontsize=25)
    plt.tight_layout()
    plt.show()


def training_models(model,x_train,y_train,val_x,val_y):
    model.fit(x_train,y_train)
    model_name = model.__class__.__name__
    print("{} model training accuracy: {}".format(model_name,model.score(val_x,val_y)))

def cross_validation(model,x_data,y_data,cv):
    scores = cross_val_score(model,x_data,y_data,cv=cv)
    model_name = model.__class__.__name__
    for iter_count, accuracy in enumerate(scores):
        print('Validation {0} Accuracy {1}'.format(iter_count, accuracy))
    print(' Average Accuracy for {} :{} '.format(model_name, np.mean(scores)))
    return np.mean(scores)
def get_passanger(columns):
    passanger = pd.DataFrame()
    for column in columns:
        passanger.at[0,column]=input(" Give me passanger {}".format(column))
    passanger_pred=passanger
    fillempty(passanger_pred)
    passanger_pred.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)
    rescaling(passanger_pred)
    encode_data(passanger_pred)

    return  passanger_pred

def prediction_passanger(model,test_input):
    model_name = model.__class__.__name__
    prediction = model.predict(test_input)
    print('{} model predict that passanger : {}'.format(model_name,"Survive" if prediction==1 else "Dead" ))
