from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from AdaBoost import AdaBoost
# importing custom functions.
#--------
import os
import sys
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#--------


def custom_learning_curve(x_train, y_train,
                           x_test, y_test,
                          n_splits):

  
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  split_size = int(len(x_train) / n_splits)
  
  x_splits = np.split(x_train, n_splits) # must be equal division
  y_splits = np.split(y_train, n_splits)
  print("-------------------")
  train_accuracies = list()
  
  test_accuracies = list()
  curr_x = x_splits[0]
  #print(curr_x.shape)
  curr_y = y_splits[0]
  #print(curr_y.shape)
  a = AdaBoost(50, 700, 30, 15000)
  a.fit(curr_x, curr_y)
  train_accuracies.append(accuracy_score(curr_y,
                                         a.predict(curr_x)))

  
  test_accuracies.append(accuracy_score(y_test, a.predict(x_test)))

  for i in range(1, len(x_splits)):
    #knn = KNeighborsClassifier(n_neighbors=best_k)
    a = AdaBoost(50, 700, 30, 15000)
    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    print(curr_x.shape)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    print(curr_y.shape)
    a.fit(curr_x, curr_y)

    train_accuracies.append(accuracy_score(curr_y,
                                           a.predict(curr_x)))
    print(train_accuracies[i])

    #val_accuracies.append(accuracy_score(y_val, a.predict(x_val)))
    acc = accuracy_score(y_test, a.predict(x_test))
    print(acc)
    test_accuracies.append(acc)

  plt.plot(list(range(split_size, len(x_train) + split_size,
                      split_size)), train_accuracies, 'o-', color="b",
             label="Training accuracy")
  
  plt.plot(list(range(split_size, len(x_train) + split_size,
                      split_size)), test_accuracies, 'o-', color="red",
           label="Testing accuracy")
  plt.legend(loc="lower right")
  plt.xlabel('Percentage of data')
  plt.ylabel('Accuracy')
  plt.show()



xTrain, yTrain = loadTrainData()
xTest, yTest = loadTestData()
print("----")
xTest, yTest = shuffleData(xTest, yTest)
xTrain, yTrain = shuffleData(xTrain, yTrain)


custom_learning_curve(xTrain, yTrain, xTest, yTest, 20)

