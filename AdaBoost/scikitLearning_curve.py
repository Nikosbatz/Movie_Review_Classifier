import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
from sklearn.ensemble import AdaBoostClassifier

def plot_learning_curve(estimator, title,
                        X_for_val, y_for_val,
                        X_for_test=None, y_for_test=None,
                        ylim=None,
                        val_cv=None,
                        test_cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_for_val, y_for_val,
        cv=val_cv, n_jobs=-1, scoring='accuracy',
        train_sizes=train_sizes)


    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1,
                     color="green")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="green",
             label="Validation score")


    

    plt.legend(loc="lower right")
    plt.show()
    return plt



xTrain, yTrain = loadTrainData()
xTrain, yTrain = shuffleData(xTrain, yTrain)
vocab = createVocabulary(700, 50, 15000)
xTrain = createVector(xTrain, vocab)
ada = AdaBoostClassifier(n_estimators=50)


plot_learning_curve(estimator=ada, title='Learning Curve',
                    X_for_val=xTrain, 
                    y_for_val=yTrain,
                    val_cv=None)

