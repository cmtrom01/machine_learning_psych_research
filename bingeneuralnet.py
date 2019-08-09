#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:44:21 2019

@author: christrombley
"""


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from livelossplot.keras import PlotLossesCallback


dataset = pd.read_csv('/Users/christrombley/Downloads/nueralNetDataPurge1.csv')

#dataset = pd.read_csv('/Users/christrombley/Downloads/nueralNetDataBinge4.csv')
iv = dataset.iloc[:, 0:51].values
dv = dataset.iloc[:, 52].values

counter = 0
            
        
    

X_train, X_test, y_train, y_test = train_test_split(iv, dv, test_size = 0.5)


from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense
import keras
import tensorflow as tf

# Initialize the constructor
model = Sequential()

model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_dim=51))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(8, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(4, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=100, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[PlotLossesCallback()],
              verbose=0,validation_split=0.5, shuffle=True)

 
y_pred = model.predict_classes(X_test)


score = model.evaluate(X_test, y_test,verbose=1)

print(score)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, auc

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

print(precision_score(y_test, y_pred, average = "weighted"))

print(recall_score(y_test, y_pred, average = "weighted"))

print(f1_score(y_test,y_pred, average = "weighted"))

print(cohen_kappa_score(y_test, y_pred))


# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,0]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# method II: ggplot
from ggplot import *
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')



##auc(y_test, y_pred)
