from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from NB import NB


def custom_learning_curve(x_train, y_train,
                           x_test, y_test,
                          n_splits):

  
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  split_size = int(len(x_train) / n_splits)
  print(type(x_train))
  x_splits = np.split(x_train, n_splits) # must be equal division
  y_splits = np.split(y_train, n_splits)
  print("-------------------")
  train_accuracies = list()
  val_accuracies = list()
  test_accuracies = list()
  curr_x = x_splits[0]
  #print(curr_x.shape)
  curr_y = y_splits[0]
  #print(curr_y.shape)
  nb = NB()
  nb.fit(curr_x, curr_y)
  train_accuracies.append(accuracy_score(curr_y,
                                         nb.predict(curr_x)))

  #val_accuracies.append(accuracy_score(y_val, nb.predict(x_val)))
  test_accuracies.append(accuracy_score(y_test, nb.predict(x_test)))

  for i in range(1, len(x_splits)):
    #knn = KNeighborsClassifier(n_neighbors=best_k)
    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    #print(curr_x.shape)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    #print(curr_y.shape)
    nb.fit(curr_x, curr_y)

    train_accuracies.append(accuracy_score(curr_y,
                                           nb.predict(curr_x)))
    print(train_accuracies[i])

    #val_accuracies.append(accuracy_score(y_val, nb.predict(x_val)))
    acc = accuracy_score(y_test, nb.predict(x_test))
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


nb = NB()
xTrain, yTrain = nb.loadTrainData()
xTest, yTest = nb.loadTestData()
print("----")
xTest, yTest = nb.shuffleData(xTest, yTest)
xTrain, yTrain = nb.shuffleData(xTrain, yTrain)


custom_learning_curve(xTrain, yTrain, xTest, yTest, 5)

