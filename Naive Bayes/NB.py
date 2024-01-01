from  osDir import splitData
import os


class NB:
    

    def __init__(self):
        self.negativeWordsProbs = {}
        self.positiveWordsProbs = {}
        self.probNeg = 0
        self.probPos = 0
    
    def train(self):
        fPos = open(os.getcwd() + "\\Raw data processing\\PositiveFreq.txt", "r")
        fNeg = open(os.getcwd() + "\\Raw data processing\\NegativeFreq.txt", "r")
        

        countPos = int(fPos.readline().split()[1])
        countNeg = int(fNeg.readline().split()[1])
        
        self.probNeg = countNeg / (countNeg + countPos)
        self.probPos = countPos / (countNeg + countPos)
        


        line = fPos.readline()
        while line != "":
            self.positiveWordsProbs[line.split()[0]] = int(line.split()[1]) / countPos
            line = fPos.readline()
    
        line = fNeg.readline()
        while line != "":
            self.negativeWordsProbs[line.split()[0]] = int(line.split()[1]) / countNeg
            line = fNeg.readline()

        fPos.close()
        fNeg.close()
        

    def predict(self, m, n, k):
        customDict = []
        OverallFile = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", "r")
        lines =  OverallFile.readlines()
        lines = (lines[n:-k])[:m]
        for i in lines:
            customDict.append( i.split()[0])
        
        OverallFile.close()

        # TESTING FILES
        file = open(os.getcwd() + "\\Raw data processing\\aclImdb\\test\\neg\\4_4.txt", "r")

        review = file.read()
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
                negProdResult *= self.negativeWordsProbs[item[0]]
                print(negProdResult)

        negProdResult *= self.probNeg
        

        # POSITIVE
        posProdResult = 1
        for item in review.items():
            if item[1] == 1:
                posProdResult *= self.positiveWordsProbs[item[0]]
                print(posProdResult)

        posProdResult *= self.probPos
        
        print( posProdResult > negProdResult)

