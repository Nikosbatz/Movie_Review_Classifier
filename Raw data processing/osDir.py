import os

dictPositive = {}
print(os.getcwd())
dataDir = os.getcwd() + "\\aclImdb\\train\\pos\\"
for file in os.listdir(dataDir):
    f = open(dataDir + file, "r", errors="ignore")
    comment = f.read()
    commList = comment.split()
    for c in commList:
        c = c.lower()
        if c in dictPositive.keys():
            dictPositive[c] += 1
        else:
            dictPositive[c] = 1



dictNegative = {}
dataDir = os.getcwd() + "\\aclImdb\\train\\neg\\"
for file in os.listdir(dataDir):
    f = open(dataDir + file, "r", errors="ignore")
    comment = f.read()
    commList = comment.split()
    for c in commList:
        c = c.lower()
        if c in dictNegative.keys():
            dictNegative[c] += 1
        else:
            dictNegative[c] = 1

f.close()

dictNegative = dict(sorted(dictNegative.items(), key= lambda item: item[1], reverse=True))
dictPositive = dict(sorted(dictPositive.items(), key= lambda item: item[1], reverse=True))


f = open("NegativeFreq.txt", 'a')
for item in dictNegative.items():
    f.write(item[0]+" "+str(item[1])+"\n")

f.close()
    
f = open("PositiveFreq.txt", 'a')
for item in dictPositive.items():
    f.write(item[0]+" "+str(item[1])+"\n")

f.close()












