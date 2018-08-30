from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process import GaussianProcessClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

#[Height, weight, shoe size]

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37], [166,65,40], [190,90,47],[175,64,39],[177,70,40],[159,55,37], [171,75,42],[181,84,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clf1 =tree.DecisionTreeClassifier()
clfDT =clf1.fit(X, Y)

clf2 = KNeighborsClassifier(n_neighbors=3)
clfKN =clf2.fit(X, Y)

clf3 = GaussianProcessClassifier()
clfGP = clf3.fit(X, Y)

clf4 = RandomForestClassifier(n_estimators=10)
clfRF = clf4.fit(X, Y)


test = [[165,54,38]]

#Classifiers results
predictDT = clfDT.predict (test)
predictKN = clfKN.predict (test)
predictGP = clfGP.predict (test)
predictRF = clfRF.predict (test)


#Probabilities
probDT = clfDT.predict_proba(test)
probKN = clfKN.predict_proba(test)
probGP = clfGP.predict_proba(test)
probRF = clfRF.predict_proba(test)


print "DT Classifier test data {} is predicted as {} with probability of {}".format(test, predictDT, probDT)

print "KN classifier test data {} is predicted as {} with probability of {}".format(test, predictKN, probKN)

print "GP classifier test data {} is predicted as {} with probability of {}".format(test, predictGP, probGP)

print "RF Classifier test data {} is predicted as {} with probability of {}".format(test, predictRF, probRF)
