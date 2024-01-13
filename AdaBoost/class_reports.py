from AdaBoost import AdaBoost
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
# importing files from another folder path
import sys
import os
sys.path.append(os.getcwd() + "\\Raw data processing")
from loadData import *
from createVocabulary import *
#----------------------------

#nb = NB()
Ada = AdaBoost(50, 700, 50, 15000)


xTrain, yTrain = loadTrainData()
xTest, yTest = loadTestData()
xTrain, yTrain = shuffleData(xTrain, yTrain)
predictions = []
classifications = []
split_size = int(len(xTrain) / 5)
x_splits = np.split(np.array(xTrain), 5) # must be equal division
y_splits = np.split(np.array(yTrain), 5)
curr_x = x_splits[0]
curr_y = y_splits[0]
Ada.fit(curr_x, curr_y)
predictions.append(Ada.predict(xTest))
classifications.append(classification_report(yTest, predictions[0], output_dict=True))

for i in range(1,5):
    Ada = AdaBoost(50, 700, 50, 15000)
    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    Ada.fit(curr_x, curr_y)
    predictions.append(Ada.predict(xTest))
    classifications.append(classification_report(yTest, predictions[i], output_dict=True ))

precisions = []
f1 = []
recall = []
for i in classifications:
    precisions.append((i["0"]["precision"] + i["1"]["precision"])/2 )
    f1.append((i["0"]["f1-score"] + i["1"]["f1-score"])/2 )
    recall.append((i["0"]["recall"] + i["1"]["recall"])/2 )
    




plt.plot(list(range(split_size, len(xTrain) + split_size,
                    split_size)), precisions, 'o-', color="b",
            label="precision score")

plt.plot(list(range(split_size, len(xTrain) + split_size,
                    split_size)), f1, 'o-', color="red",
            label="f1-score")

plt.plot(list(range(split_size, len(xTrain) + split_size,
                    split_size)), recall, 'o-', color="yellow",
            label="recall score")

plt.legend(loc="lower right")
plt.xlabel('Amount of data')
plt.ylabel('Score')
plt.show()


print(classification_report(yTest, predictions[4]))