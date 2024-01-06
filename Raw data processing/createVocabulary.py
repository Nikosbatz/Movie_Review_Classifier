import os

# Code needed to import the function from a diffrent folder
""""import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from createVocabulary import createVocabulary
"""
#-----------

def createVocabulary(m, n, k):

    vocabulary = {}
    OverallFile = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", "r")
    lines =  OverallFile.readlines()
    lines = (lines[n:-k])[:m]
    for i in lines:
        vocabulary [i.split()[0]] = 0
            
    OverallFile.close()
    return vocabulary