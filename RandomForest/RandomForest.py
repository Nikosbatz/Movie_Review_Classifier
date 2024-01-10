from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from id3 import ID3
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from createVocabulary import *
from loadData import *


class RandomForest:
    def __init__(self, n_estimators, features, m, n, k):
        self.n_estimators = n_estimators
        self.features = features
        self.m = m
        self.n = n
        self.k = k
        self.trees = []

    def fit(self, x, y):
        for _ in range(self.n_estimators):
            # Bootstrap sampling - random sampling with replacement
            x_sample, y_sample = resample(x, y, replace=True, random_state=np.random.randint(1000))
            
            # Create and train a decision tree using ID3
            tree = ID3(self.features,self.m , self.n, self.k)
            tree.fit(x_sample, y_sample)
            
            # Add the trained tree to the forest
            self.trees.append(tree)

    def predict(self, x):
        # Make predictions using each tree and combine them through majority voting
        predictions = np.array([tree.predict(x) for tree in self.trees])
        ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return ensemble_predictions

# Example usage:
vocabulary = createVocabulary(100, 10, 1000)
xTrain, yTrain = loadTrainData()

RF = RandomForest(5, vocabulary, 100, 10, 1000)
RF.fit(xTrain, yTrain)
RF.predict(xTrain)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(yTrain, RF.predict(xTrain))
print(accuracy)