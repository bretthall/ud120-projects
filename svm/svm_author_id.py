#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

clf = svm.SVC(kernel = "rbf", C = 10000)
print "training..."
t = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time() - t, 3), "s"

print "Predicting..."
t = time()
res = clf.predict(features_test)
print "prediction time: ", round(time() - t, 3), "s"

from sklearn.metrics import accuracy_score

print "accuracy: ",accuracy_score(labels_test, res)

print "pred[10] = ", res[10]
print "pred[26] = ", res[26]
print "pred[50] = ", res[50]

numChris = sum((x == 1) for x in res)
print "num chris = ", numChris

#########################################################


