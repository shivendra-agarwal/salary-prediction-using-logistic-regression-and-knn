#### salary prediction using logistic regression and k nearest neighbor
#### Logistic regression Vs. K nearest neighbour

###### -*- coding: utf-8 -*-
"""

import pandas as pd

###### to perfom numerical operations
import numpy as np

###### to visualize data
import seaborn as sns

###### to partition the data
from sklearn.model_selection import train_test_split

###### import library for logistic regression
from sklearn.linear_model import LogisticRegression

###### import performance metrics, accuracy, confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

###### import data from csv
data_income=pd.read_csv("income.csv")

###### copy of data
data=data_income.copy()


"""Exploratory data analysis starts here
###### 1. Know your data
######2. Preprceossing data
#3. Cross tablesand visualizations
"""
#
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^1. KNOW YOUR DATA^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# variable types
print(data.info())

# missing values
print(data.isnull())
print(data.isnull().sum()) # no missing values

# summary of numerical variables
summary_num=data.describe()
print(summary_num)

# summary of categorical variables
summary_cat=data.describe(include="O")
print(summary_cat)

# frequency of each categories
data['JobType'].value_counts() #missing values in the form of ?
data['EdType'].value_counts()
data['maritalstatus'].value_counts()
data['occupation'].value_counts() #missing values in the form of ?
data['relationship'].value_counts()
data['race'].value_counts()
data['gender'].value_counts()
data['nativecountry'].value_counts()
data['SalStat'].value_counts()

# check for unique classes
print(np.unique(data['JobType']))

### we can see that, ' ?' ' Federal-gov' ' Local-gov'.... have white space 
### before the name and also a question mark. We can now re-read the data to 
### correct this error.
print(np.unique(data['occupation']))
### Same as above

# re-reading data to change ' ?' as nan
data=pd.read_csv("income.csv",na_values=[" ?"]) #siz of data remains same, only
# ? is treated as NaN

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2. Pre-processing^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# check null values
print(data.isnull().sum()) #JobType 1809 occupation 1816

# check where are nullvalues
missing=data[data.isnull().any(axis=1)] # 1816
# axis = 1, atleast one cloumn value is missing

"""
#   1. Missing values in JobType 1809
#   2. Missing values in occupation 1816
#   3. There are 1809 where both JobType and occupation have missing values
#   4. occupation inself has only 7 missing values where JobType is never-worked
"""
data2=data.dropna(axis=0)
# axis=0, drop all the rows with missing values #1816
# not able to get mechanism or relationship between variables
# explain about 3 types which can be follwoed

# relationship between independent numerical variables
correlation=data2.corr()
# no or week correlation found

# relationsship between categorical variables
data2.columns

# Gender proportion
gender=pd.crosstab(index = data2["gender"], columns = "count", normalize = False)
print(gender)
# gives count distribution
gender=pd.crosstab(index = data2["gender"], columns = "count", normalize = True)
print(gender)
# gives percentage distribution

# Gender-salesStat relation
gender_salstat=pd.crosstab(index = data["gender"], columns = data["SalStat"], margins = True, normalize = 'columns') #2x3
print(gender_salstat)
gender_salstat=pd.crosstab(index = data["gender"], columns = data["SalStat"], margins = True, normalize = 'index') #3x2
print(gender_salstat)
print(gender_salstat) #men are more likely to earn
# normalize = 'index' # gives row proportion = 1
# setting margins = true # add subcolumn "ALL"
# in classification problem, we need to know how balanced classes are. 

# Frequency distrubution of salary status using seaborn
SalStat = sns.countplot(data["SalStat"])

# Histogram of age
sns.distplot(data["age"], bins = 10, kde = False) # kde = False to have the frequency on y-axis
# people with age 20-45  are high in frequency

# box-plot - age Vs. SalStat
sns.boxplot(data["SalStat"],data["age"])
# bivaritate analysis - relation between age and SalStat
# we can see that, people between age 25 - 45 morelikely earn less than or equal to 50,000
# whereas people between age 35 - 50 morelikely earn greater than 50,000
# we can generate more relationships -  - -   - - - - - - - - 

data2.groupby("SalStat")["age"].median() # it gives excat median age with respect to SalStat

# relationship between JobType and SalStat.
sns.countplot(y=data["JobType"], hue = data["SalStat"])

JobType_SalStat=pd.crosstab(index = data2["JobType"], columns = data["SalStat"],margins = True, normalize = 'index')
print(JobType_SalStat)
# 55.8% of self employed earns more than 50,000 USD per year

# relationship between EdType and SalStat.
sns.countplot(y=data["EdType"], hue = data["SalStat"])

EdType_SalStat=pd.crosstab(index = data2["EdType"], columns = data["SalStat"], margins = True, normalize = 'index')
print(EdType_SalStat)

# People with Doctorate, Masters, prof-school are more likely to earn more than 50,000. it can be
# infulencing variables

# relationship between occupation and SalStat.
sns.countplot(y=data["occupation"], hue = data["SalStat"])

occapution_SalStat=pd.crosstab(index = data2["occupation"], columns = data["SalStat"],margins = True, normalize = 'index')
print(occapution_SalStat)

# People with Exec-managerial, Prof-specialty are morelikely to earn more than 50,000.

# Captial gain using histogram
sns.distplot(data["capitalgain"], bins = 10, kde = False)
# 92% of the capital gain is 0(27611)

# Captial loss using histogram
sns.distplot(data['capitalloss'], bins = 10, kde = False)
# 95% of the capital loss is 0(28721)

# box-plot - hoursperweek Vs. SalStat
sns.boxplot(data["SalStat"],data["hoursperweek"])
# those who spend 40-50 hours are morelikely to earn more than 50,000


# ^^^^^^^^^^^^^^Logistic Regression^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

# A classification technique

# reindexing salary status names to 0,1
data2.loc[:,'SalStat']=data2.loc[:,'SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2.loc[:,'SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)

# storing the column name
columns_list = list(new_data.columns)
print(columns_list)

# seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# storing the output values in y
y=new_data['SalStat'].values
print(y)

# storing the output values in x
x=new_data[features].values
print(x)

# splitting the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state = 0)

# make an instance
logistic = LogisticRegression()

# fit the values train_x and train_y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

# confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)
#             prediction
#               yes           No

# Actual Yes     6338         485 
#        No      941        1285

# !so many wrong classification

# calculating accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

# printing misclassified values
print('Misclassified samples: %d' % (test_y != prediction).sum())


# ^^^^^^^^^^^^^^Logistic Regression - removing insignificant variables^^^^^^^^^^^^^^^^^^^^^#

# reindexing salary status names to 0,1
data2.loc[:,'SalStat']=data2.loc[:,'SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2.loc[:,'SalStat'])

Cols = ['gender', 'nativecountry', 'race','JobType']
new_data=data2.drop(Cols, axis=1)

new_data=pd.get_dummies(data2, drop_first=True)

# storing the column name
columns_list = list(new_data.columns)
print(columns_list)

# seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# storing the output values in y
y=new_data['SalStat'].values
print(y)

# storing the output values in x
x=new_data[features].values
print(x)

# splitting the data into train and test
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3, random_state = 0)

# make an instance
logistic = LogisticRegression()

# fit the values train_x and train_y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

# confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)
#             prediction
#               yes           No

# Actual Yes     6338         485 
#        No      941        1285

# !so many wrong classification

# calculating accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

# printing misclassified values
print('Misclassified samples: %d' % (test_y != prediction).sum())

# ^^^^^^^^^^^^^^^^^^^^^k nearest neighbor^^^^^^^^^^^^^^^^^^^^^^^^^^^#

# import sklearn KNN library
from sklearn.neighbors import KNeighborsClassifier

# import matplot for visualization
import matplotlib.pyplot as plt

# storing the K nearest neighbors classifiers
KNN_classifier = KNeighborsClassifier(n_neighbors=5)

# fitting the values of x and y
KNN_classifier = KNN_classifier.fit(train_x, train_y)

# prediction of test data
prediction = KNN_classifier.predict(test_x)

# perform metric check
confusion_matrix = confusion_matrix(test_y, prediction)

print("\t","Predicted values")
print("\t","Original values")

# calculating accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print("Misclassified samples: %d" % (test_y != prediction).sum())

"""
# Effect of K value on classifier
"""

Misclassified_sample = []
# calculaiting error of k values between 1 and 20
for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, test_y)
    pred_i = knn.predict(train_x)
    Misclassified_sample.append((test_y!=pred_i).sum())

print(Misclassified_sample)

"""
# We can say that on the given dataset logistic regression and k nearest neighbor does not works well because the accuracy lies between 80% to 85% for both the algorithms. 
