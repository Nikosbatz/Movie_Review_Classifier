from RandomForest import RandomForest
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from createVocabulary import *
from loadData import *
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve


def custom_learning_curve(x_train, y_train,
                           x_test, y_test,
                          n_splits):

  
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  split_size = int(len(x_train) / n_splits)
  print(type(x_train))
  x_splits = np.split(x_train, n_splits) # must be equal division
  y_splits = np.split(y_train, n_splits)
  
  train_accuracies = list()
  val_accuracies = list()
  test_accuracies = list()
  curr_x = x_splits[0]
  #print(curr_x.shape)
  curr_y = y_splits[0]
  #print(curr_y.shape)
  vocabulary = createVocabulary(500, 100, 1000)
  RF = RandomForest(5, vocabulary, 500, 100, 1000, 30)
  RF.fit(curr_x, curr_y)
  print("======= FIT ========")
  train_accuracies.append(accuracy_score(curr_y,
                                         RF.predict(curr_x)))

  #val_accuracies.append(accuracy_score(y_val, RF.predict(x_val)))
  test_accuracies.append(accuracy_score(y_test, RF.predict(x_test)))

  for i in range(1, len(x_splits)):
    #knn = KNeighborsClassifier(n_neighbors=best_k)
    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    #print(curr_x.shape)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    #print(curr_y.shape)
    RF.fit(curr_x, curr_y)

    train_accuracies.append(accuracy_score(curr_y,
                                           RF.predict(curr_x)))
    print(train_accuracies[i])

    #val_accuracies.append(accuracy_score(y_val, RF.predict(x_val)))
    acc = accuracy_score(y_test, RF.predict(x_test))
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
xTrain, yTrain = shuffleData(xTrain, yTrain)
xTest, yTest = loadTestData()
xTest, yTest = shuffleData(xTest, yTest)

custom_learning_curve(xTrain, yTrain, xTest, yTest, 5)



