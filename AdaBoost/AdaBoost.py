from DecisionStump import *

from math import log2
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
        self.stumpWeights = []
        self.stumps = []
        self.vocab = None
        

    def fit(self, xTrain, yTrain):
        
        self.vocab = createVocabulary(self.m, self.n, self.k)
        xVector = createVector(xTrain, self.vocab)
        
        xTrainLen = len(xTrain)
        weights = []
        for i in range(xTrainLen):
            weights.append(1/xTrainLen)
        
        count = 0
        for i in range(self.estimators):
            count += 1
            print(count, "------")

            stump = DecisionStump()
            
            stump.fit(xVector, yTrain, weights)
            print(stump.feature_index)
            
            self.stumps.append(stump)
            #print(self.vocab[stump.feature_index])
            predictions = stump.predict(xVector)
            

            count1 = 0
            error = 0
            for k in range (xTrainLen):
                if predictions[k] != yTrain[k]:
                    count1 += 1
                    error += weights[k]
                #error += weights[k] if predictions[k] != yTrain[k] else 0
                
            
            print("Weighted error: ",error)
            if error >= 0.5 :
                
                print("ERROR >= 0.5")
                print(error)
                break
            
            for k in range(xTrainLen):
                if predictions[k] == yTrain[k]:
                    weights[k] *= error/(1-error)
                else:
                    None
                #weights[k] *= error/(1-error) if predictions[k] == yTrain[k] else 1
                
            
            for k in range(len(weights)):
                
                weights[k] = weights[k] / sum(weights)
            
            
             
            self.stumpWeights.append( 1/2 * log2((1-error) / error ))

            
            
            

    def predict(self, xTest):
        
        # Converts reviews in binary vectors
        xVector = createVector(xTest, self.vocab)
        
        # Intialize final predictions list
        y = list()
        ls = []

        count = 0
        # Iterates over each item of the input given
        for review in xVector:
            
            sum_pos = 0
            sum_neg = 0

            # Calculates the predictions of each stump
            for i in range(self.estimators):

                prediction = self.stumps[i].predict(review)
                # Calculates the review's sum based on stump weights
                if prediction == 1:
                    sum_pos += self.stumpWeights[i]
                else:
                    sum_neg += self.stumpWeights[i]

            
            y.append(1 if sum_pos>sum_neg else 0)
        
        return y
    

xTrain, yTrain = loadTrainData()

a = AdaBoost(50, 1000, 200, 1000)

a.fit(xTrain, yTrain)
print(a.stumpWeights)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


r = classification_report(yTrain, a.predict(xTrain))
print(r)
"""accuracy = accuracy_score(yTrain, a.predict(xTrain))
print(accuracy)"""


