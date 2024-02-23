import os

# Code needed to import the function from a diffrent folder
""""import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from createVocabulary import *
"""
#-----------

def createVocabulary(m, n, k):

    vocabulary = {}
    OverallFile = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", "r", errors="ignore") 
    lines =  OverallFile.readlines()
    lines = (lines[n:-k])[:m]
    for i in lines:
        vocabulary [i.split()[0]] = 0
            
    OverallFile.close()
    return vocabulary


def createVector(x, vocab):
    xVector = []
    keys = list(vocab.keys())
    for review in x: 
        reviewVector = [0]*len(vocab)
        review = review.split()
        
        for word in review:
            if vocab.get(word) == 0:
                reviewVector[keys.index(word)] = 1

        xVector.append(reviewVector)

    return xVector

