from DecisionStump import *
from sklearn.tree import DecisionTreeClassifier
from math import *
# importing custom functions.
#--------
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------

# ----- Scikit Learn DecisionTreeClassifier is used in this implementation -----

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

            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(xVector, yTrain, sample_weight=weights)
            self.stumps.append(stump)
            
            
            predictions = stump.predict(xVector)
            predictions = np.array(predictions)
            
            error = 0
            for k in range (xTrainLen):
                if predictions[k] != yTrain[k]:
                    error += weights[k]
                
            
            self.stumpWeights.append( 1/2 * log((1-error) / error ))
            
            print("Weighted error: ",error)
            if error >= 0.5 :
                
                print("ERROR >= 0.5")
                print(error)
                break
            # Decreasing weights of correct predictions
            for k in range(xTrainLen):
                if predictions[k] == yTrain[k]:
                    weights[k] *= error/(1-error)

            # Normalizing
            s = sum(weights)
            for k in range(len(weights)):                
                weights[k] = weights[k] / s

            
            
    
    def predict(self, xTest):
        
        xTest = createVector(xTest, self.vocab)
        xTest = np.array(xTest)
        predictions = np.zeros(len(xTest))
        for k in range(len(self.stumps)):
            pred = self.stumps[k].predict(xTest)
            for i in range(len(pred)):
                
                predictions[i] += -1*self.stumpWeights[k] if pred[i]==-1 else 1*self.stumpWeights[k]

        for i in range(len(predictions)):
            predictions[i] = 0 if predictions[i] <= 0 else 1


        print(predictions)
        return predictions
        
        

        
"""xTrain, yTrain = loadTrainData()

ab = AdaBoost(50, 500, 50, 1000)
ab.fit(xTrain, yTrain)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yTrain, ab.predict(xTrain))
print(accuracy)"""