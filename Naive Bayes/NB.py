
import os


class NB:
    

    def __init__(self):
        self.negativeWordsProbs = {}
        self.positiveWordsProbs = {}
        self.probNeg = 0
        self.probPos = 0
        self.countNeg = 0
        self.countPos = 0
    
    def train(self) -> None:
        fPos = open(os.getcwd() + "\\Raw data processing\\PositiveFreq.txt", "r")
        fNeg = open(os.getcwd() + "\\Raw data processing\\NegativeFreq.txt", "r")
        
        

        self.countPos = int(fPos.readline().split()[1])
        self.countNeg = int(fNeg.readline().split()[1])
        
        self.probNeg = self.countNeg / (self.countNeg + self.countPos)
        self.probPos = self.countPos / (self.countNeg + self.countPos)
        
        # Initiating Dictionaries containing calculated word probabilities (So Laplace smoothing can be used).
        
        
        


        # Calculating P(Xi|C) using Laplace smoothing
        line = fPos.readline()
        while line != "":
            self.positiveWordsProbs[line.split()[0]] = int(line.split()[1]) +1 / self.countPos +2
            line = fPos.readline()
    
        line = fNeg.readline()
        while line != "":
            self.negativeWordsProbs[line.split()[0]] = int(line.split()[1]) +1/ self.countNeg +2
            line = fNeg.readline()

        fPos.close()
        fNeg.close()
        

    def predict(self, xTest, m, n, k) -> list:

        # creates custom dictionary based on the hyper-parameters (m, n, k) given.
        customDict = []
        OverallFile = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", "r")
        lines =  OverallFile.readlines()
        lines = (lines[n:-k])[:m]
        for i in lines:
            customDict.append( i.split()[0])
        
        OverallFile.close()

        predictedValues = []
        for review in xTest:
            reviewList = review.split()
            review = {}
            # Initiate the dictionary containing the vector
            for i in reviewList:
                review[i] = 0
            # Specifying which words the review contains (0 or 1 in the vector attributes)
            for item in review.keys():
                review[item] = item in customDict
            
            # NEGATIVE
            negProdResult = 1
            for item in review.items():
                if item[1] == 1:
                    if item[0] in self.negativeWordsProbs:
                        negProdResult *= self.negativeWordsProbs[item[0]]
                    else:
                        negProdResult = 1 / self.countNeg +2

            negProdResult *= self.probNeg
            
            # POSITIVE
            posProdResult = 1
            for item in review.items():
                if item[1] == 1:
                    if item[0] in self.positiveWordsProbs:
                        posProdResult *= self.positiveWordsProbs[item[0]]
                    else:
                        posProdResult = 1 / self.countPos +2

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
        
