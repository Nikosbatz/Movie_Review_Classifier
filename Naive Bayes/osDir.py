import os

def splitData():
    dictOverall = {}
    countPos = 0
    countNeg = 0
    dictPositive = {}
    
    
    dataDir = os.getcwd() + "\\Raw data processing\\aclImdb\\train\\pos\\"
    
    for file in os.listdir(dataDir): 
        words = []
        f = open(dataDir + file, "r", errors="ignore")
        comment = f.read()
        countPos += 1
        commList = comment.split()
        for c in commList:
            c = c.lower()
            if c not in words:
                words.append(c)
                if c in dictPositive.keys():
                    dictPositive[c] += 1
                    
                else:
                    dictPositive[c] = 1
                    



    dictNegative = {}
    dataDir = os.getcwd() + "\\Raw data processing\\aclImdb\\train\\neg\\"
    for file in os.listdir(dataDir):
        words = []
        f = open(dataDir + file, "r", errors="ignore")
        comment = f.read()
        countNeg += 1
        commList = comment.split()
        for c in commList:
            c = c.lower()
            if c not in words:
                words.append(c)
                
                if c in dictNegative.keys():
                    dictNegative[c] += 1
                else:
                    dictNegative[c] = 1

    f.close()

    dictNegative = dict(sorted(dictNegative.items(), key= lambda item: item[1], reverse=True))
    dictPositive = dict(sorted(dictPositive.items(), key= lambda item: item[1], reverse=True))


    f = open(os.getcwd() + "\\Raw data processing\\NegativeFreq.txt", 'w')
    f.write("countNeg "+ str(countNeg)+"\n")
    for item in dictNegative.items():
        f.write(item[0]+" "+str(item[1])+"\n")

    f.close()


        
    f = open(os.getcwd() + "\\Raw data processing\\PositiveFreq.txt", 'w')
    f.write("countPos "+ str(countPos)+"\n")
    for item in dictPositive.items():
        f.write(item[0]+" "+str(item[1])+"\n")

    f.close()


    dictOverall = dictPositive.copy()
    for c in dictNegative.keys():

        if c in dictOverall.keys():
            dictOverall [c] = dictNegative[c] + dictOverall[c]
        else:
            dictOverall[c] = dictNegative[c]

    dictOverall = dict(sorted(dictOverall.items(), key= lambda item: item[1], reverse=True))
    
    f = open(os.getcwd() + "\\Raw data processing\\OverallFreq.txt", 'w')
    for item in dictOverall.items():
        f.write(item[0]+" "+str(item[1])+"\n")

    f.close()


