import random
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *



class NB:
    

    def __init__(self, negativeWordsProbs = {}, positiveWordsProbs = {}, probNeg = 0, probPos = 0, countNeg = 0, countPos = 0):
        self.negativeWordsProbs = negativeWordsProbs
        self.positiveWordsProbs = positiveWordsProbs
        self.probNeg = probNeg
        self.probPos = probPos
        self.countNeg = countNeg
        self.countPos = countPos
    

    def fit(self, xTrain, yTrain) -> None:
        positiveWordFreq = {}
        negativeWordFreq = {}

        # Counting each word's occurences on each classification
        for i in range(len(xTrain)):
            insertedWords = []
            reviewWords = xTrain[i].split()
            
            if yTrain[i] == 0:
                self.countNeg += 1
                for word in reviewWords:
                    word = word.lower()
                    if word not in insertedWords:
                        insertedWords.append(word)
                        if word in negativeWordFreq.keys():
                            negativeWordFreq[word] += 1
                        else:
                            negativeWordFreq[word] = 1

            else:
                self.countPos += 1
                for word in reviewWords:
                    word = word.lower()
                    if word not in insertedWords:
                        insertedWords.append(word)
                        if word in positiveWordFreq.keys():
                            positiveWordFreq[word] += 1
                        else:
                            positiveWordFreq[word] = 1
            
        self.probNeg = self.countNeg / (self.countNeg + self.countPos)
        self.probPos = self.countPos / (self.countNeg + self.countPos)

        # Calculating P(Xi|C)
        for item in negativeWordFreq.items():
            self.negativeWordsProbs[item[0]] = item[1] / self.countNeg
        
        for item in positiveWordFreq.items():
            self.positiveWordsProbs[item[0]] = item[1] / self.countPos

        self.negativeWordsProbs = dict(sorted(self.negativeWordsProbs.items(), key= lambda item: item[1], reverse=True))
        self.positiveWordsProbs = dict(sorted(self.positiveWordsProbs.items(), key= lambda item: item[1], reverse=True))
             

        
        

    def predict(self, xTest, m=6000, n=300, k=9000) -> list:

        # creates custom dictionary based on the hyper-parameters (m, n, k) given.
        customDict = createVocabulary(m, n, k)
        
        

        
        count = 0
        predictedValues = []
        
        for review in xTest:
            count += 1
            reviewList = review.split()
            
            review = {}
            # Initiate the dictionary containing the vector
            for i in reviewList:
                review[i] = 0
            # Specifying which words the review contains (0 or 1 in the vector attributes)
            for item in review.keys():
                review[item] =  1 if customDict.get(item) == 0 else 0
            
            
            
            # NEGATIVE
            negProdResult = 1
            for item in review.items():
                if item[1] == 1:
                    if item[0] in self.negativeWordsProbs:
                        negProdResult *= self.negativeWordsProbs[item[0]]
                    else:
                        negProdResult *= 1 / self.countNeg +2

            negProdResult *= self.probNeg
            
            # POSITIVE
            posProdResult = 1
            for item in review.items():
                if item[1] == 1:
                    if item[0] in self.positiveWordsProbs:
                        posProdResult *= self.positiveWordsProbs[item[0]]
                    else:   
                        posProdResult *= 1 / self.countPos +2
                        

            posProdResult *= self.probPos
            
            predictedValues.append( 1 if posProdResult>negProdResult else 0)
        
            
        return predictedValues
            
        
        
    def get_params(self, deep=True):
        return {"negativeWordsProbs":self.negativeWordsProbs, "positiveWordsProbs":self.positiveWordsProbs, "probNeg":self.probNeg, 
                "probPos":self.probPos, "countPos":self.countPos, "countNeg":self.countNeg}
    
