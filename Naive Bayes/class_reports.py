from NB import NB
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

"""naiveB = NB()

print(classification_report(yTest, predicted,
                            zero_division=1))"""


nb = NB()

xTrain, yTrain = nb.loadTrainData()
xTest, yTest = nb.loadTestData()
xTrain, yTrain = nb.shuffleData(xTrain, yTrain)
classifications = []
split_size = int(len(xTrain) / 5)
x_splits = np.split(np.array(xTrain), 5) # must be equal division
y_splits = np.split(np.array(yTrain), 5)
curr_x = x_splits[0]
curr_y = y_splits[0]
nb.fit(curr_x, curr_y)
classifications.append(classification_report(yTest, nb.predict(xTest), output_dict=True))

for i in range(1,5):

    curr_x = np.concatenate((curr_x, x_splits[i]), axis=0)
    curr_y = np.concatenate((curr_y, y_splits[i]), axis=0)
    nb.fit(curr_x, curr_y)
    classifications.append(classification_report(yTest, nb.predict(xTest), output_dict=True ))

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
