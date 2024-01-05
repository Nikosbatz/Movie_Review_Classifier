
import os
import random


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
             

        #fPos = open(os.getcwd() + "\\Raw data processing\\PositiveFreq.txt", "r")
        #fNeg = open(os.getcwd() + "\\Raw data processing\\NegativeFreq.txt", "r")
        
        

        #self.countPos = int(fPos.readline().split()[1])
        #self.countNeg = int(fNeg.readline().split()[1])
        
        #self.probNeg = self.countNeg / (self.countNeg + self.countPos)
        #self.probPos = self.countPos / (self.countNeg + self.countPos)
        
        # Calculating P(Xi|C) using Laplace smoothing
        """line = fPos.readline()
        while line != "":
            self.positiveWordsProbs[line.split()[0]] = int(line.split()[1]) +1 / self.countPos +2
            line = fPos.readline()
    
        line = fNeg.readline()
        while line != "":
            self.negativeWordsProbs[line.split()[0]] = int(line.split()[1]) +1/ self.countNeg +2
            line = fNeg.readline()

        fPos.close()
        fNeg.close()"""
        

    def predict(self, xTest, m=6000, n=300, k=9000) -> list:

        # creates custom dictionary based on the hyper-parameters (m, n, k) given.
        customDict = {}
        OverallFile = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", "r")
        lines =  OverallFile.readlines()
        lines = (lines[n:-k])[:m]
        for i in lines:
            customDict [i.split()[0]] = 0
            
        OverallFile.close()

        

        
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
            


    def loadTestData(self) -> list:
        data = []
        expected = []
        path = os.getcwd() + "\\Raw data processing\\aclImdb\\test\\neg\\"
        for file in os.listdir(path):
            f = open(path + file, "r", errors="ignore")
            data.append(f.read())
            expected.append(0)
        
        path = os.getcwd() + "\\Raw data processing\\aclImdb\\test\\pos\\"
        for file in os.listdir(path):
            f = open(path + file, "r", errors="ignore")
            data.append(f.read())
            expected.append(1)

        return [data, expected]
    

    def loadTrainData(self) -> list:
        data = []
        expected = []
        path = os.getcwd() + "\\Raw data processing\\aclImdb\\train\\neg\\"
        for file in os.listdir(path):
            f = open(path + file, "r", errors="ignore")
            data.append(f.read())
            expected.append(0)
        
        path = os.getcwd() + "\\Raw data processing\\aclImdb\\train\\pos\\"
        for file in os.listdir(path):
            f = open(path + file, "r", errors="ignore")
            data.append(f.read())
            expected.append(1)

        return [data, expected]
        
        
    def get_params(self, deep=True):
        return {"negativeWordsProbs":self.negativeWordsProbs, "positiveWordsProbs":self.positiveWordsProbs, "probNeg":self.probNeg, 
                "probPos":self.probPos, "countPos":self.countPos, "countNeg":self.countNeg}
    

    def shuffleData(self, data, result):
        ls = list(zip(data,result))
        random.shuffle(ls)
        data, result = zip(*ls)
        return [data, result]