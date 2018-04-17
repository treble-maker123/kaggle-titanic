# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data_folder = "./"

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv(data_folder + "train.csv")
test_df =  pd.read_csv(data_folder + "test.csv")
dummy_df = train_df
dummy_df = dummy_df.append(test_df, ignore_index=True)

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

def prep_data(input_df, is_training=True):
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
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = split_data(X, y)

test_passenger_ids = test_df['PassengerId']
test_input = prep_data(test_df, is_training=False).as_matrix()

# ==============================
# Tensorflow Configuration
# ==============================

tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# ==============================
# Neural network parameters
# ==============================

learn_rate = 0.001
epochs = 600
input_size = len(X[0])
output_size = 1

# ==============================
# Setting up model
# ==============================

tf_x = tf.placeholder(tf.float32, [None, input_size])
tf_y = tf.placeholder(tf.float32, [None, output_size])

l1_output_size = input_size * 5
l2_output_size = input_size * 4
l3_output_size = input_size * 3

weights = {
  "l1": tf.Variable(tf.random_normal([input_size, l1_output_size], stddev=0.03), name="W1"),
  "l2": tf.Variable(tf.random_normal([l1_output_size, l2_output_size], stddev=0.3), name="W2"),
  "l3": tf.Variable(tf.random_normal([l2_output_size, l3_output_size], stddev=0.3), name="W3"),
  "out": tf.Variable(tf.random_normal([l3_output_size, output_size], stddev=0.3), name="WOut")
}

biases = {
  "l1": tf.Variable(tf.random_normal([l1_output_size]), name='b1'),
  "l2": tf.Variable(tf.random_normal([l2_output_size]), name='b2'),
  "l3": tf.Variable(tf.random_normal([l3_output_size]), name='b3'),
  "out": tf.Variable(tf.random_normal([output_size]), name='bOut')
}

layer_1 = tf.add(tf.matmul(tf_x, weights["l1"]), biases["l1"])
layer_1 = tf.nn.relu(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, weights["l2"]), biases["l2"])
layer_2 = tf.nn.relu(layer_2)
layer_3 = tf.add(tf.matmul(layer_2, weights["l3"]), biases["l3"])
layer_3 = tf.nn.relu(layer_3)

logits = tf.add(tf.matmul(layer_3, weights["out"]), biases["out"])
out = tf.nn.sigmoid(logits)

model_output = tf.round(out)

cost = tf.losses.sigmoid_cross_entropy(tf_y, out)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate).minimize(cost)

correct_prediction = tf.equal(model_output, tf_y)
accuracy = tf.scalar_mul(100, tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

# ==============================
# Training
# ==============================

best_test = 0
file_name = './titanic_nn.ckpt'

with tf.Session() as session:
  session.run(init)
  for epoch in range(epochs):
    session.run(optimizer, feed_dict={tf_x: X_train, tf_y: y_train})
    
    test_acc = session.run(accuracy, feed_dict={tf_x: X_test, tf_y: y_test})
    if test_acc > best_test:
      print(f"Epoch {epoch} - acc: {test_acc}, saving model.")
      best_test = test_acc
      saver.save(session, file_name)
                
# ==============================
# Predict
# ==============================

result = pd.DataFrame({"PassengerId": test_passenger_ids})

with tf.Session() as session:
  saver.restore(session, file_name)
  predictions = session.run(model_output, feed_dict={tf_x: test_input})

result['Survived'] = [int(p) for p in predictions]
result.to_csv(data_folder + 'output.csv', index=False)