# Titanic Survival Prediction 🚢
## Technologies 💡
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


## Description ❔❓
![Titanic-II-wyplynie-w-rejs-w-2022-roku -Czy-tym-razem-podroz-zakonczy-sie-szczesliwie_article](https://user-images.githubusercontent.com/122997699/221285433-66d6c0a8-2f9c-4875-ad10-cdfd10f734b5.jpg)

In this project, using the sklearn library, I will create a model thanks to which I will be able to make predictions about whether a given passenger would have survived the sinking of the Titanic.
When accuracy of prediction will be satysficed, I will want to make simple graphical user interface, thanks to which usser will be able to define features of passanger, and prediction his survival. 
## Dataset📁

Dataset which I use in this problem you can find below [link](https://www.kaggle.com/c/titanic)

## Project implementation description 🔍

### Contents of individual files

* Titanic.py - main file, in which functions are use with appropriate datasets and parameters
* Function.py - file in which all the necessary functions are defined 

### Loading, analisis and preprocesing dataset
In this part I load training and test data from csv files. Now i can look at training data and take some info about it.
![image](https://user-images.githubusercontent.com/122997699/221366208-db08e8ef-06de-47c2-a482-8e7d2d55006c.png)

Now I must check if in dataset are Nan values, and fill it correctly. For this case I creat ***fillempty*** function.
Number of NaN value in each column:

![image](https://user-images.githubusercontent.com/122997699/221366532-81470930-d916-41b6-a053-362e87a0f0bb.png)

***fillempty*** function : 
```python
def fillempty(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Cabin'].fillna('Empty', inplace=True)
    data['Embarked'].fillna('Empty', inplace=True)
    data['Fare'].fillna(0, inplace=True)

    return data
```

Now I can check the dependence of survival on individual characteristics of passengers. For this cas I create ***data_analisis*** function. This function return few typical chart of dependence of survival and indyvidual characteristics like :

*Passenger age

![Survival by age](https://user-images.githubusercontent.com/122997699/221367281-8b4a8d4a-b8d4-4233-ba37-44e80342254a.png)

* Passenger sex

![Survival by gender](https://user-images.githubusercontent.com/122997699/221367296-aebfea51-4147-420c-a6ab-29c6c0e48961.png)

* Passenger class

![Survival by age passenger class](https://user-images.githubusercontent.com/122997699/221367287-f3679705-8c1d-4d8f-97ef-3fd62ed596e6.png)

Now I can create targets using the pop function. 

```python
train_features = train.pop('Survived')
```
Columns such as _Name_, _Ticket_, _PassengerID_, and _Age Range_ (which was only created for the chart) will not be needed when training the model. So I can drop they. 
```python
train.drop(['Name', 'Ticket', 'PassengerId','Age Range'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
```
Another very important step is encoding string-type feature data to value between 0 and n_classes-1. I created for this purpose ***encode_data*** function. 
```python
def encode_data(data):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        data[feature] = le.transform(data[feature])
    return data
```
### Training models

First I split data from train and validation. Validation data will be 30% of all data. 
```python
train_x,val_x,train_y,val_y = train_test_split(train,train_features,test_size=.30,random_state=42)
```

First I must to choose best classifier. For this reason i put in to the list 7 typical classifier, and create ***training model*** function, thanks a witch I can train model with each classifier.
```python
models = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(random_state=11),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        LogisticRegression(max_iter=500),
        LinearDiscriminantAnalysis()
]
```
```python
def training_models(model,x_train,y_train,val_x,val_y):
    model.fit(x_train,y_train)
    model_name = model.__class__.__name__
    print("{} model training accuracy: {}".format(model_name,model.score(val_x,val_y)))
```
```python
for model in models:
        training_models(model,train_x,train_y,val_x,val_y)
```
![image](https://user-images.githubusercontent.com/122997699/221405966-b867b7ef-fe05-425d-af69-cded88f1413a.png)

In this case, best training accuracy have:
* LogisticRegression
* GradientBoostingClassifier
* AdaBoostClassifier
* LinearDiscriminantAnalysis

Now, I evaluate my classifiers using cross-validation. For this task I create ***cross_validation*** function. 
```python
def cross_validation(model,x_data,y_data,cv):
    scores = cross_val_score(model,x_data,y_data,cv=cv)
    model_name = model.__class__.__name__
    for iter_count, accuracy in enumerate(scores):
        print('Validation {0} Accuracy {1}'.format(iter_count, accuracy))
    print(' Average Accuracy for {} :{} '.format(model_name, np.mean(scores)))
    return np.mean(scores)
```
I use this function to each classifier, thanks to witch I can specify Average Accuracy and show it in graph. 
```python
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
```
![Cross Validation](https://user-images.githubusercontent.com/122997699/221407401-30a8a59d-bde2-4ebb-9fdb-b8cd15983efa.png)

In this case, best  cross-validation average accuracy have:
* RandomForestClassifier
* GradientBoostingClassifier
* AdaBoostClassifier
* LinearDiscriminantAnalysis


##### In next step I Will try to predict survival passenger by  from test.csv file, and passenger defined by user.
To this case i choose:
* LinearDiscriminantAnalysis
* GradientBoostingClassifier
* LogisticRegression

## Next goals 🏆⌛
* Increasing prediction accuracy as much as possible
* Trying other prediction models 
* Creating graphic user interface
* Feature Rescaling
* Increase accuracy of this classifiers by using GridSearch CV
