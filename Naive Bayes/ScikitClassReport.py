from sklearn.naive_bayes import GaussianNB
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
from sklearn.metrics import classification_report

X_train, y_train = loadTrainData()
X_test, y_test = loadTestData()
vocab = createVocabulary( m=6000, n=300, k=9000)
X_train = createVector(X_train, vocab)

X_test = createVector(X_test, vocab)

nb = GaussianNB()
nb.fit(X_train, y_train)

print(classification_report(y_test, nb.predict(X_test),
                            zero_division=1))