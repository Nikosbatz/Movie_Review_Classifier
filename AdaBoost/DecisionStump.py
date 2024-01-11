import os
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
        columns = len(xTrain[0])
        minError = 1
        # iterate over every feature of the xTrain list
        for feature_index in range(columns):
                    
            # features are binary values
            
            predictions = list()
            exists_pos = 0
            exists_neg = 0
            NoExist_pos = 0
            NoExist_neg = 0
            # iterate over every list that xTrain contains. (xTrain is a 2d list)
            for k in range(len(xTrain)):
                if xTrain[k][feature_index] == 1:
                    if yTrain[k] == 1:
                        exists_pos +=1
                    else:
                        exists_neg +=1 
                else:
                    if yTrain[k] == 1:
                        NoExist_pos +=1 
                    else:
                        NoExist_neg +=1 

            for line in xTrain:
                if line[feature_index] == 1:
                    if exists_pos > exists_neg:
                        predictions.append(1)
                        
                    else:
                        predictions.append(0)
                         
                else:
                    if NoExist_pos > NoExist_neg:
                        predictions.append(1)
                        
                    else:
                        predictions.append(0)
                        
            
            error = 0
            
            for i in range(len(predictions)):
                if predictions[i] != yTrain[i]:
                      
                    error += sampleWeights[i] 
            
            #print(c)
            # save the values of the stump with minimum error so far.
            
            if error < minError and feature_index not in self.existing_features:
                 
                minError = error
                
                self.feature_index = feature_index
                self.existsCategory = exists_pos > exists_neg
                self.noExistsCategory = NoExist_pos > NoExist_neg
                
                    #print(mistakes[:10])
                #print(self.feature_index, error)


    def predict(self, xTest):
        predictions = []

        if isinstance(xTest[0], list):
            for line in xTest:
                
                if line[self.feature_index] == 1:
                    
                    predictions.append(self.existsCategory)
                            
                else:
                    predictions.append(self.noExistsCategory)
        else:
            return (self.existsCategory if xTest[self.feature_index] == 1 else self.noExistsCategory)             
                
                    
                        
        
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

