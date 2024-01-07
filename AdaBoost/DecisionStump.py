import os
import numpy as np
# importing custom functions.
#--------
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------

class DecisionStump:

    
    def __init__(self):
        self.feature_index = 0
        
              
        

    def fit(self, xTrain, yTrain):
        count = 0
        lines = len(xTrain)
        weights = np.ones(lines) / lines
        columns = len(xTrain[0])
        minError = 1
        # iterate over every feature of the xTrain list
        for feature_index in range(columns):
            count +=1
            #print(count)
            
            # features are binary values
            
            predictions = list()
            # iterate over every list that xTrain contains. (xTrain is a 2d list) 
            for line in xTrain:
                   
                if line[feature_index] == 1:
                    predictions.append(1)
                else:
                    predictions.append(0)

            error = 0
            for i in range(len(predictions)):
                if predictions[i] != yTrain[i]:
                    error += weights[i] 

            # save the values of the stump with minimum error so far.
            if error < minError:
                minError = error
                self.feature_index = feature_index
                


    def predict(self, xTest):
        results = []
        for line in xTest:
            results.append(xTest[self.feature_index] == 1)

        return results


x, y = loadTrainData()



vocab = createVocabulary(100, 100, 10000)

xVector = createVector(x, vocab)


from sklearn.metrics import accuracy_score

d = DecisionStump()
d.fit(xVector, y)
print(list(vocab.keys())[d.feature_index])
print(d.feature_index)


accuracy = accuracy_score(y, d.predict(xVector))
print(accuracy)

