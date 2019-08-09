#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:46:08 2019

@author: christrombley
"""

# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
import pandas as pd

 
from sklearn import tree


dataset = pd.read_csv('/Users/christrombley/Downloads/nueralNetDataBinge4.csv')
X = dataset.iloc[:, 0:51].values
Y = dataset.iloc[:, 52].values


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
tree.plot_tree(clf)

import graphviz 

Ystr = str(Y)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("features") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X[0],  
                     class_names=Ystr,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
print("feat importance = " + str(feat_importance))

import numpy as np
import matplotlib.pyplot as plt

plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
plt.show()

importances = clf.feature_importances_
indices = np.argsort(importances)


feat_importances = pd.Series(clf.feature_importances_, index=pd.DataFrame(X).columns)
feat_importances.nlargest(20).plot(kind='barh')
 