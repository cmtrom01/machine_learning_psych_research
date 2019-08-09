# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader
import pandas as pd
from sklearn import metrics 
from sklearn import tree
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('/Users/christrombley/Downloads/nueralNetDataBinge4.csv')
iv = dataset.iloc[:, 0:51].values
dv = dataset.iloc[:, 52].values

x_train, x_test, y_train, y_test = train_test_split(iv, dv, test_size = 0.2)

from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
x_train = scaling.transform(x_train)
x_test = scaling.transform(x_test)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
tree.plot_tree(clf)

y_pred=clf.predict(x_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average="weighted"))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average="weighted"))

# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)


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

