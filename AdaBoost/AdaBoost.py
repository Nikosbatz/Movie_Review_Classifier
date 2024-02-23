from DecisionStump import *

from math import *
# importing custom functions.
#--------
import os
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------

# ----- Custom DecisionStump implementation ------

class AdaBoost:

    def __init__(self, estimators, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.estimators = estimators
        self.stumpWeights = []
        self.stumps = []
        self.vocab = None
        self.existing_features = []
        

    def fit(self, xTrain, yTrain):
        
        self.vocab = createVocabulary(self.m, self.n, self.k)
        xVector = createVector(xTrain, self.vocab)
        yTrain = list(yTrain)
        # Convert predictions from 0 or 1 to -1 or 1
        for y in range(len(yTrain)):
            yTrain[y] = yTrain[y] * 2 -1
        

        xTrainLen = len(xTrain)
        weights = []
        
        for i in range(xTrainLen):
            weights.append(1/xTrainLen)
        
        count = 0
        #######
        yTrain = np.array(yTrain)
        xVector = np.array(xVector)
        #####
       
        for i in range(self.estimators):
            count += 1
            print(count, "------")

            stump = DecisionStump(self.existing_features)
            
            stump.fit(xVector, yTrain, weights)
            
            self.existing_features.append(stump.feature_index)
            
            self.stumps.append(stump)
            #print(self.vocab[stump.feature_index])
            
            predictions = stump.predict(xVector)
            
            predictions = np.array(predictions)
            
            
            error = 0
            for k in range (xTrainLen):
                if predictions[k] != yTrain[k]:
                    error += weights[k]
                
            
            # Calculate current stump's weight.
            self.stumpWeights.append( 1/2 * log((1-error) / error ))
            
            # if error greater than 0.5 break the loop
            if error >= 0.5 :
                
                print("ERROR >= 0.5")
                print(error)
                break
            
               
            # if predictions[k] == yTrain[k] -> e^-self.stumWeights[i] < 1 so weights[k] decreases
            # if predictions[k] != yTrain[k] -> e^-self.stumWeights[i] > 1 so weights[k] increases
            # In Conclusion if a prediction is correct its weight decreases if it is mistaken then it increases.
            weights *= np.exp(-self.stumpWeights[i] * yTrain * predictions)
            
            s = sum(weights)
            for k in range(len(weights)):                
                weights[k] = weights[k] / s

            
            
            

    def predict(self, xTest):
        
        xTest = createVector(xTest, self.vocab)
        xTest = np.array(xTest)
        predictions = np.zeros(xTest.shape[0])
        for alpha, stump in zip(self.stumpWeights, self.stumps):
            predictions += alpha * np.array(stump.predict(xTest))

        # Convert back to binary predictions (0 or 1)
        return (predictions > 0).astype(int)
        
    

"""xTrain, yTrain = loadTrainData()
xTrain, yTrain = shuffleData(xTrain, yTrain)
a = AdaBoost(80, 500, 10, 1000)

a.fit(xTrain[:15000], yTrain[:15000])


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#r = classification_report(yTrain[12500:], a.predict(xTrain[12500:]))
#print(r)

xTest, yTest = loadTestData()


accuracy = accuracy_score(yTrain, a.predict(xTrain))
print(accuracy)

accuracy = accuracy_score(yTest, a.predict(xTest))
print(accuracy)
"""