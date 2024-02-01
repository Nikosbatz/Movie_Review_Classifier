from RandomForest import RandomForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
sys.path.append(os.getcwd() + "\\Raw data processing")
from createVocabulary import *
from loadData import *
import time

def custom_learning_curve(x_train, y_train,
                           x_test, y_test,
                          n_splits):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    split_size = int(len(x_train) / n_splits)
    print(type(x_train))
    x_splits = np.split(x_train, n_splits)  # must be equal division
    y_splits = np.split(y_train, n_splits)

    train_accuracies = list()
    test_accuracies = list()
    curr_x = x_splits[0]
    curr_y = y_splits[0]

    vocabulary = createVocabulary(500, 50, 750)
    RF = RandomForest(10, vocabulary, 500, 50, 750, 20)

    # Initial fit
    start_time = time.time()
    RF.fit(curr_x, curr_y)
    end_time = time.time()
    print(f"Initial Fit Time: {end_time - start_time:.2f} seconds")

    train_accuracies.append(accuracy_score(curr_y, RF.predict(curr_x)))
    test_accuracies.append(accuracy_score(y_test, RF.predict(x_test)))

    for i in range(1, len(x_splits)):
        curr_x = np.concatenate((curr_x, x_splits[i][:500]), axis=0)  # Adjust the sample size as needed
        curr_y = np.concatenate((curr_y, y_splits[i][:500]), axis=0)  # Adjust the sample size as needed

        # Fit for the current iteration
        start_time = time.time()
        RF.fit(curr_x, curr_y)
        end_time = time.time()
        print(f"Fit Time (Iteration {i}): {end_time - start_time:.2f} seconds")

        train_accuracies.append(accuracy_score(curr_y, RF.predict(curr_x)))
        acc = accuracy_score(y_test, RF.predict(x_test))
        print(f"Test Accuracy (Iteration {i}): {acc:.4f}")

        test_accuracies.append(acc)

    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), train_accuracies, 'o-', color="b",
             label="Training accuracy")
    plt.plot(list(range(split_size, len(x_train) + split_size, split_size)), test_accuracies, 'o-', color="red",
             label="Testing accuracy")
    plt.legend(loc="lower right")
    plt.xlabel('Percentage of data')
    plt.ylabel('Accuracy')
    plt.show()

# Load and shuffle data
xData, yData = loadTrainData()
xData, yData = shuffleData(xData, yData)

# Split data into 70-30 train-test sets
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# Call the custom_learning_curve function
custom_learning_curve(xTrain, yTrain, xTest, yTest, 5)