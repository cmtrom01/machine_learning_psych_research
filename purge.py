#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:44:18 2019

@author: christrombley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 12:04:10 2019

@author: christrombley
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import _pickle as pickle

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
from keras.layers import Dropout
import numpy as np





dataset = pd.read_csv('/Users/christrombley/Downloads/nueralNetDataPurge1.csv')
iv = dataset.iloc[:, 0:51].values
dv = dataset.iloc[:, 52].values

print(iv[1])

x_train, x_test, y_train, y_test = train_test_split(iv, dv, test_size = 0.2, random_state = 1)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##change one to however many hidden layer you want
##each y row in array
classifier = Sequential()
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=51, units = 51, name = "1"))
#classifier.add(Dropout(rate=0.1))
classifier.add(Dense(output_dim=51, kernel_initializer="uniform", activation = "relu", name = "2"))
#classifier.add(Dropout(rate=0.1))
classifier.add(Dense(output_dim=51, kernel_initializer = "uniform", activation="sigmoid", name="3"))
#classifier.add(Dropout(rate=0.1))
classifier.add(Dense(output_dim=1, kernel_initializer = "uniform", activation="sigmoid", name="4"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(x_train, y_train, batch_size = 32, nb_epoch = 10)


inputWeights = list(classifier.layers[0].get_weights())
outputWeights = list(classifier.layers[2].get_weights())


#print(inputWeights[0])
## save to variable
w_l1 = pd.DataFrame(inputWeights[0])
w_l2 = pd.DataFrame(outputWeights[0])


def connection_weights(A, B):
    """
    Computes Connection weights algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = matrix of weights of hidden-output layer (rows=hidden & cols=output)
    """    
    cw = np.dot(A, B)
    
    # normalize to 100% for relative importance
    ri = cw / cw.sum()
    return(ri)
    
    
df = connection_weights(w_l1, w_l2)
df = pd.DataFrame(df)
df.head()
print(df)

n = 50

f, ax = plt.subplots(figsize=(25, 6))
sns.boxplot(w_l2[:n].transpose())
plt.xlabel("Input feature")
plt.ylabel("weight of input-hidden node")
plt.title("Input-hidden node weights of the first {} features".format(n))
plt.show()

f, ax = plt.subplots(figsize=(25, 30))
df.plot(kind="line", ax=ax)
plt.xlabel("Input feature")
plt.ylabel("Relative importance")
plt.savefig("relativefeatureimportance.pdf")
plt.show()


# Partitioning features to classes by max weight
df.idxmax(axis=1).value_counts(sort=True).plot(kind="bar")
plt.xlabel("Index of output class")
plt.ylabel("Count")
plt.title("Input features assigned to output class by maximizing importance")
plt.savefig("featureimportancebinge.pdf")
    