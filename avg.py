# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data_folder = "./"

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv(data_folder + "train.csv")
test_df =  pd.read_csv(data_folder + "test.csv")
dummy_df = train_df
dummy_df = dummy_df.append(test_df, ignore_index=True)

# Output data
test_passenger_ids = test_df['PassengerId']
result = pd.DataFrame({"PassengerId": test_passenger_ids})

# Fit encoders with both train and test data
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()
titles_encoder = LabelEncoder()

sex_encoder.fit(dummy_df['Sex'].as_matrix())

dummy_df['Embarked'].fillna('U', inplace=True)
embarked_encoder.fit(dummy_df['Embarked'].as_matrix())

names = [name.split(',')[1] for name in dummy_df['Name'].as_matrix()]
dummy_df['Title'] = [name.split('.')[0].strip() for name in names]
dummy_df['Title'].fillna('U', inplace=True)
titles_encoder.fit(dummy_df['Title'].as_matrix())

# Fit the scaler on both train and test data
# For some reason, batch scaling them isn't working here, so using separate scalers.
scalers = [preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()]
scaled_col = ['Fare', 'Age']
for i, c in enumerate(scaled_col):
  dummy_df[c] = pd.to_numeric(dummy_df[c], errors='coerce')
  dummy_df[c].fillna(0.0, inplace=True)
  values = dummy_df[c].as_matrix().reshape(-1, 1)
  scalers[i].fit(values)

def prep_data(input_df):
  names = [name.split(',')[1] for name in input_df['Name'].as_matrix()]
  input_df['Title'] = [name.split('.')[0].strip() for name in names]
  
  input_df['Embarked'].fillna('U', inplace=True)
  input_df['Age'].fillna(0.0, inplace=True)
  input_df['Title'].fillna('U', inplace=True)
  input_df['HasCabin'] = [1 if isinstance(cabin, str) else 0 for cabin in input_df['Cabin'].as_matrix()]
  input_df['HasRelatives'] = [int(i[0]+i[1]>0) for i in zip(input_df['SibSp'].as_matrix(), input_df['Parch'].as_matrix())]
  
  # encode
  input_df['Sex'] = sex_encoder.transform(input_df['Sex'].as_matrix())
  input_df['Embarked'] = embarked_encoder.transform(input_df['Embarked'].as_matrix())
  input_df['Title'] = titles_encoder.transform(input_df['Title'].as_matrix())
  
  # scale
  for i, c in enumerate(scaled_col):
    input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
    input_df[c].fillna(0.0, inplace=True)
    input_df[c] = scalers[i].transform(input_df[c].as_matrix().reshape(-1, 1))
    input_df[c] = round(input_df[c], 3)
      
  columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
  input_df.drop(columns_to_drop, inplace=True, axis=1)
  
  input_df.dropna(inplace=True)
  input_df.reset_index(inplace=True)
  input_df.drop(['index'], axis=1, inplace=True)
  return input_df
    
def get_targets(data_frame):
  X = data_frame.drop(['Survived'], axis=1).as_matrix()
  y = data_frame['Survived'].as_matrix()
  return X, y

def split_data(input_X, input_y):
  return train_test_split(input_X, input_y, test_size=0.4)

df = prep_data(train_df)
X, y = get_targets(df)
# y = y.reshape(-1, 1) # 2D for tensorflow, 1D for sklearn.
X_train, X_test, y_train, y_test = split_data(X, y)
test_input = prep_data(test_df).as_matrix()

# ==============================
# K-NN
# ==============================

from sklearn.neighbors import KNeighborsClassifier

min_neighbors = 1
max_neighbors = 100
best_score = 0
best_neighbors = None
best_model = None

for i in range(min_neighbors, max_neighbors+1):
  model = KNeighborsClassifier(n_neighbors=i)
  model.fit(X_train, y_train)
  score = model.score(X_test, y_test)
  if score > best_score:
    best_neighbors = i
    best_score = score
    best_model = model

print(f"Best K-NN ({best_neighbors} neighbors): {round(best_score*100,2)}%")
result['knn'] = best_model.predict(test_input)

# ==============================
# Logistic Regression
# ==============================

from sklearn.linear_model import LogisticRegression

# liblinear
model = LogisticRegression(solver="liblinear", max_iter=2000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Logistic Regression with liblinear solver: {round(score*100,2)}%")
result['lr_ll'] = model.predict(test_input)

# newton-cg
model = LogisticRegression(solver="newton-cg", max_iter=2000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Logistic Regression with newton-cg solver: {round(score*100,2)}%")
result['lr_nc'] = model.predict(test_input)

# ==============================
# Stochastic Gradient Descent
# ==============================

from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log', max_iter=5000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"SGD with log loss: {round(score*100,2)}%")
result['sgd_log'] = model.predict(test_input)

model = SGDClassifier(loss='perceptron', learning_rate='invscaling', eta0=0.01, max_iter=5000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"SGD with perceptron: {round(score*100,2)}%")
result['sgd_log'] = model.predict(test_input)

# ==============================
# Support Vector Machine
# ==============================

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Support Vector Classification: {round(score*100,2)}%")
result['svc'] = model.predict(test_input)

# ==============================
# Random Forest
# ==============================

from sklearn.ensemble import RandomForestClassifier

min_estimators = 1
max_estimators = 30
best_score = 0
best_n_est = None
best_model = None

for i in range(min_estimators, max_estimators+1):
  model = RandomForestClassifier(n_estimators=i)
  model.fit(X_train, y_train)
  score = model.score(X_test, y_test)
  if score > best_score:
    best_score = score
    best_model = model
    best_n_est = i

print(f"Random forest ({best_n_est} estimators): {round(best_score*100,2)}%")
result['rf'] = model.predict(test_input)

predictions = result.drop(['PassengerId'], axis=1)
model_col = predictions.columns
n_models = len(model_col)
predictions = predictions.sum(axis=1)
predictions = predictions.apply(lambda row: int(round((row/n_models*1.0)+0.01)))
result['Survived'] = predictions
result.drop(model_col, inplace=True, axis=1)

result.to_csv('./output.csv', index=False)