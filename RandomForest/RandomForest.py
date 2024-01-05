from AI_Project_2 import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees = 10, max_depth = 10, min_samples_split = 2, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            X_sample, Y_sample = self.bootstrap_samples(X, Y)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)

    def bootstrap_samples(self, X, Y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], Y[idxs]
    
    def most_common_label(self, Y):
        counter = Counter(Y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)