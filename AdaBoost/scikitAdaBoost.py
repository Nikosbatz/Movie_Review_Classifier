from sklearn.ensemble import AdaBoostClassifier
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
from sklearn.metrics import classification_report

X_train, y_train = loadTrainData()
X_test, y_test = loadTestData()
vocab = createVocabulary(700, 50, 15000)
X_train = createVector(X_train, vocab)

X_test = createVector(X_test, vocab)

ab = AdaBoostClassifier(n_estimators=50)
ab.fit(X_train, y_train)

print(classification_report(y_test, ab.predict(X_test),
                            zero_division=1))