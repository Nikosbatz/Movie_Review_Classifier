from sklearn.ensemble import AdaBoostClassifier
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
from sklearn.metrics import classification_report

X_train, y_train = loadTrainData()
vocab = createVocabulary(500, 100, 1000)
X_train = createVector(X_train, vocab)


ab = AdaBoostClassifier(n_estimators=35)
ab.fit(X_train, y_train)
print(classification_report(y_train, ab.predict(X_train),
                            zero_division=1))
"""print(classification_report(y_test, ab.predict(X_test),
                            zero_division=1))"""