import os
# importing custom functions.
#--------
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
#--------

class DecisionStump:

    def __init__(self, vocab):
        self.feature_index = 0
        self.category = 0
        self.vocabulary = vocab        
        

    def fit(self, xTrain, yTrain, weights):
        lines = len(xTrain)
        columns = len(xTrain[0])
        minError = -1
        for feature_index in range(columns):

            for category in range(2):
                predictions = list()
                for line in xTrain:
                    if line[feature_index] == category:
                        predictions.append(1)
                    else:
                        predictions.append(0)

                    error = 0
                    for i in range(predictions):
                        if predictions[i] != yTrain[i]:
                            error += error + weights[i]

                    if error < minError:
                        minError = error
                        self.feature_index = feature_index
                        self.category = category


    def predict(self, xTest):
        results = []
        for line in xTest:
            results.append(xTest[self.feature_index] == self.category)

        return results


x, y = loadTrainData()
d = DecisionStump(None)


