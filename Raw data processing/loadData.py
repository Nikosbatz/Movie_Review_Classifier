import os

# Code needed to import the function from a diffrent folder
""""import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
"""
#-----------


def loadTestData() -> list:
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
    

def loadTrainData() -> list:
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

import random 

def shuffleData(data, result):
        ls = list(zip(data,result))
        random.shuffle(ls)
        data, result = zip(*ls)
        return [data, result]