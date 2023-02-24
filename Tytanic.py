import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Function import *

# Uploading data and creatin targets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Checking information about training dataset
print(train.info,train.isnull().sum())

# Filling empty cells
fillempty(train)

# Creating targets
train_features = train.pop('Survived')




















































# male_survived = len(train[(train['Sex'] == 'male') & (train['Survived'] == 1)])
# male_dead = len(train[(train['Sex'] == 'male') & (train['Survived'] == 0)])
#
# female_survived = len(train[(train['Sex'] == 'female') & (train['Survived'] == 1)])
# female_dead = len(train[(train['Sex'] == 'female') & (train['Survived'] == 0)])
#
# fig = plt.figure()
# plt.subplot(1,2,1)
# plt.pie([male_survived,male_dead],labels=['Survived','Dead'],autopct='%1.1f%%')
# plt.legend(bbox_to_anchor=(0.25,-0.25),loc="lower left")
# plt.title("Male")
# plt.subplot(1,2,2)
# plt.pie([female_survived,female_dead],labels=['Survived','Dead'],autopct='%1.1f%%')
# plt.title("Female")
# fig.suptitle('Survival by gender',fontsize=25)
# plt.show()