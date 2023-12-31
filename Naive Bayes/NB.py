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

        print(customDict.__len__())

