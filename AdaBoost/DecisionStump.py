import os
from statistics import mode
import random
import numpy as np
# importing custom functions.
#--------
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------

class DecisionStump:

    
    def __init__(self, existing_features):
        self.feature_index = 0
        self.existing_features = existing_features

              
    def fit(self, xTrain, yTrain, sampleWeights):

        sampleWeights = np.array(sampleWeights)
        columns = len(xTrain[0])
        minError = 1

        # iterate over every feature of the xTrain list
        for feature_index in range(columns):
            
            
            
            # i == 1 -> if feature_index == 1 (word exists) then classify the review as -1 (bad) else the review is 1 (good)
            # i == 0 -> if feature_index == 0 (word doesnt exist) then classify the review as -1 (bad) else the review is 1 (good)
            for i in range (2):
                
                    predictions = np.ones(len(xTrain))
                    predictions[ xTrain[:, feature_index] == i] = -1
                    error = np.sum(sampleWeights * (predictions != yTrain))
                    
                    # save the values of the stump with minimum error so far.
                    if error < minError and feature_index not in self.existing_features:
                        
                        minError = error
                        self.feature_index = feature_index
                        self.i = i
                        
                    
        print(minError)

    def predict(self, xTest):
        predictions = []
        xTest = np.array(xTest)
        # if xTest is 2D array-like matrix
        if not isinstance(xTest[0], int):
            
            predictions = np.ones(len(xTest))
            predictions[xTest[:, self.feature_index] == self.i] = -1
        # if xTest is 1D array-like matrix
        else:
            prediction = 1
            return -1 if prediction[self.feature_index] == self.i else 1             
                
                    
                        
        
        return predictions


"""x, y = loadTrainData()



vocab = createVocabulary(100, 300, 10000)

xVector = createVector(x, vocab)


from sklearn.metrics import accuracy_score

d = DecisionStump()
d.fit(xVector, y)
print(list(vocab.keys())[d.feature_index])
print(d.feature_index)


accuracy = accuracy_score(y, d.predict(xVector))
print(accuracy)"""

