# importing custom functions.
#--------
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------


class AdaBoost:

    def __init__(self, estimators, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.estimators = estimators

    def fit(self, xTrain):
        
        vocabulary = createVocabulary(self.m, self.n, self.k)
        xVector = createVector(xTrain, vocabulary)
        
        xTrainLen = len(xTrain)
        weights = ([1]*xTrainLen) / xTrainLen

    def predict(self):
        None