from NB import NB
from sklearn.metrics import classification_report

naiveB = NB()
naiveB.train()
print("-----------------")
#naiveB.predict(500, 50, 9000)
xTest, yTest = naiveB.loadTestData()
predicted = naiveB.predict(xTest=xTest, m=500, n=50, k=9000)

print("-----------------")

print(classification_report(yTest, predicted,
                            zero_division=1))

