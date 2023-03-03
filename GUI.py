
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QListWidget, QVBoxLayout, QLabel, QGridLayout,QSlider
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QHBoxLayout
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon
import sys
from Function import *
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv('train.csv')
data = data.dropna()
cabins = data["Cabin"].values
LR_model = joblib.load("LR_model.joblib")
GBC_model = joblib.load("GBC_model.joblib")
LDA_model = joblib.load("LDA_model.joblib")

class Titanic_app(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    def initUI(self):
        self.passanger = pd.DataFrame()
        self.Name = QLabel("Name:")
        self.Sex= QLabel("Sex:")
        self.Class = QLabel("Class:")
        self.Age = QLabel("Age: ")
        self.age_value = QLabel('0')
        self.Sibsp = QLabel("Spouses aboard the Titanic ?")
        self.Parch = QLabel("Children aboard the Titanic ?")
        self.Ticket = QLabel("Imagine passanger ticket number")
        self.Fare = QLabel("Passenger fare: ")
        self.fare_value = QLabel('0')
        self.Cabin = QLabel("Cabin number ?")
        self.Embarked = QLabel("Port of Embarkation ?")
        self.LR_Pred = QLabel("Survival prediction by LogisticRegression: ")
        self.GBC_Pred = QLabel("Survival prediction by GradientBoostingClassifier: ")
        self.LDA_Pred = QLabel("Survival prediction by LinearDiscriminantAnalysis: ")




        self.pred_button = QPushButton('&MAKE PREDICTION ')
        self.pred_button.resize(self.pred_button.sizeHint())
        self.pred_button.clicked.connect(self.prediction)


        self.save_button = QPushButton('&SAVE TO CSV ')
        self.save_button.resize(self.pred_button.sizeHint())
        self.save_button.clicked.connect(self.save)

        self.name = QLineEdit()
        self.name.textChanged.connect(self.updatename)

        self.sex = QComboBox()
        self.sex.addItems(['','male','female'])
        self.sex.currentTextChanged.connect(self.sex_change)

        self.pclass = QComboBox()
        self.pclass.addItems(['',"First","Second","Third"])
        self.pclass.currentTextChanged.connect(self.pclass_change)

        self.age = QSlider(Qt.Horizontal)
        self.age.valueChanged[int].connect(self.updateage)
        self.age.setRange(0,100)


        self.sibsp = QComboBox()
        self.sibsp.addItems(['','No', 'Yes'])
        self.sibsp.currentIndexChanged.connect(self.sibsp_change)

        self.parch = QComboBox()
        self.parch.addItems(['','No', 'Yes'])
        self.parch.currentIndexChanged.connect(self.parch_change)

        self.ticket = QLineEdit()
        self.ticket.textChanged.connect(self.updateticket)


        self.fare = QSlider(Qt.Horizontal)
        self.fare.valueChanged[int].connect(self.updatefare)
        self.fare.setRange(0, 500)


        self.cabin = QComboBox()
        self.cabin.addItems(cabins)
        self.cabin.currentIndexChanged.connect(self.cabin_change)

        self.embarked = QComboBox()
        self.embarked.addItems(['','Cherbourg','Queenstown','Southampton'])
        self.embarked.currentTextChanged.connect(self.embarked_change)

        self.LR_pred = QLineEdit()
        self.GBC_pred = QLineEdit()
        self.LDA_pred = QLineEdit()
        self.LR_pred.setReadOnly(True)
        self.GBC_pred.setReadOnly(True)
        self.LDA_pred.setReadOnly(True)

        layoutT = QGridLayout()
        layoutT.addWidget(self.Name, 0, 0)
        layoutT.addWidget(self.name, 1, 0)
        layoutT.addWidget(self.Age, 2, 0)
        layoutT.addWidget(self.age, 3, 0)
        layoutT.addWidget(self.age_value, 3, 1)
        layoutT.addWidget(self.Fare, 4, 0)
        layoutT.addWidget(self.fare, 5, 0)
        layoutT.addWidget(self.fare_value, 5, 1)
        layoutT.addWidget(self.Sex, 6, 0)
        layoutT.addWidget(self.sex, 7, 0)
        layoutT.addWidget(self.Class, 8, 0)
        layoutT.addWidget(self.pclass, 9, 0)
        layoutT.addWidget(self.Sibsp, 0, 2)
        layoutT.addWidget(self.sibsp, 1, 2)
        layoutT.addWidget(self.Parch, 2, 2)
        layoutT.addWidget(self.parch, 3, 2)
        layoutT.addWidget(self.Ticket, 4, 2)
        layoutT.addWidget(self.ticket, 5, 2)
        layoutT.addWidget(self.Cabin, 6, 2)
        layoutT.addWidget(self.cabin,7, 2)
        layoutT.addWidget(self.Embarked,8, 2)
        layoutT.addWidget(self.embarked,9, 2)
        layoutT.addWidget(self.pred_button,10,0)
        layoutT.addWidget(self.save_button,10,2)
        layoutT.addWidget(self.LR_Pred,0,4)
        layoutT.addWidget(self.LR_pred,1,4)
        layoutT.addWidget(self.GBC_Pred,4,4)
        layoutT.addWidget(self.GBC_pred,5,4)
        layoutT.addWidget(self.LDA_Pred,8,4)
        layoutT.addWidget(self.LDA_pred,9,4)



        self.setLayout(layoutT)
        self.resize(900, 300)
        self.setWindowTitle("Titanic Survival Prediction")
        self.setWindowIcon(QIcon('image.png'))
        self.show()


    def updateage(self, value):
        self.age_value.setText(str(value))
        self.age_value.setStyleSheet("QLabel"
                                      "{"
                                      "background-color: lightgreen;"
                                      "}")
    def updatefare(self, value):
        self.fare_value.setText(str(value))
        self.fare_value.setStyleSheet("QLabel"
                               "{"
                               "background-color: lightgreen;"
                               "}")
    def updatename(self,value):
        self.name.setStyleSheet("QLineEdit"
                                      "{"
                                      "background-color: lightgreen;"
                                      "}")
    def updateticket(self,value):
        self.ticket.setStyleSheet("QLineEdit"
                                      "{"
                                      "background-color: lightgreen;"
                                      "}")
    def sex_change(self,value):
        self.sex.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def pclass_change(self,value):
        self.pclass.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def sibsp_change(self,value):
        self.sibsp.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def parch_change(self,value):
        self.parch.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def embarked_change(self,value):
        self.embarked.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def cabin_change(self,value):
        self.cabin.setStyleSheet("QComboBox"
                                     "{"
                                     "background-color: lightgreen;"
                                     "}")
    def prediction(self):
        self.passanger = self.get_passanger()
        self.passanger_data = self.passanger.iloc[:1]
        self.passanger.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        self.passanger = self.passanger.loc[:,
                         ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        encode_data(self.passanger)
        rescaling(self.passanger)
        self.passanger = self.passanger.iloc[:1]
        self.LR_proba = LR_model.predict_proba(self.passanger)
        self.LR_pred.setText("{0} % die, and {1} % survive"
                            .format(round(self.LR_proba.item(0)*100,2), round(self.LR_proba.item(1)*100,2)))

        self.LDA_proba = LDA_model.predict_proba(self.passanger)
        self.LDA_pred.setText("{0} % die, and {1} % survive"
                             .format(round(self.LDA_proba.item(0)*100,2), round(self.LDA_proba.item(1)*100,2)))
        self.GBC_proba = GBC_model.predict_proba(self.passanger)

        self.GBC_pred.setText("{0} % die, and {1} % survive"
                             .format(round(self.GBC_proba.item(0)*100,2), round(self.GBC_proba.item(1)*100,2)))

    def save(self):
        if self.LR_proba.item(0) < self.LR_proba.item(1):
            self.passanger_data.loc[1,'LR'] = 'Survival'
        else:
            self.passanger_data.loc[1, 'LR'] = 'Dead'
        if self.LDA_proba.item(0) < self.LDA_proba.item(1):
            self.passanger_data.loc[1, 'LDA'] = 'Survival'
        else:
            self.passanger_data.loc[1, 'LDA'] = 'Dead'
        if self.GBC_proba.item(0) < self.GBC_proba.item(1):
            self.passanger_data.loc[1, 'GBC'] = 'Survival'
        else:
            self.passanger_data.loc[1, 'GBC'] = 'Dead'
        self.passanger_data = self.passanger_data.loc[:,
                             ['Name','Age','Sex','Pclass','SibSp', 'Parch', 'Fare', 'Embarked','LR','LDA','GBC']]
        self.passanger_data.to_csv('Titanic Prediction')
    def get_passanger(self):
        data.at[1, 'Name'] = self.name.text()
        data.at[1, 'Sex'] = self.sex.currentText()
        data.at[1, 'Age'] = self.age.value()
        data.at[1, 'Fare'] = self.fare.value()
        data.at[1, 'Ticket'] = self.ticket.text()
        pclass = self.pclass.currentText()
        if pclass == 'First':
            data.at[1, 'Pclass'] = 1
        elif pclass == 'Second':
            data.at[1, 'Pclass'] = 2
        else:
            data.at[1, 'Pclass'] = 3

        sibsp = self.sibsp.currentText()
        if sibsp == 'No':
            data.at[1, 'SibSp'] = 0
        else:
            data.at[1, 'SibSp'] = 1
        parch = self.sibsp.currentText()
        if parch == 'No':
            data.at[1, 'Parch'] = 0
        else:
            data.at[1, 'Parch'] = 1
        embarked = self.embarked.currentText()
        if embarked == 'Southampton':
            data.at[1, 'Embarked'] = "S"
        elif embarked == 'Queenstown':
            data.at[1, 'Embarked'] = "Q"
        else:
            data.at[1, 'Embarked'] = "C"
        data.at[1, 'Cabin'] = self.cabin.currentText()
        return data

import sys
app = QApplication(sys.argv)
win = Titanic_app()
sys.exit(app.exec_())