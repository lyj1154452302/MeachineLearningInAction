import numpy as np
import re
import random
import operator
import feedparser

def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec =[0, 1, 0, 1, 0, 1]   # 1代表侮辱性，0代表没有侮辱性
    return postingList, classVec

def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet | set(document)    # 得到两个集合的并集
    return list(vocabSet)

def setOfWord2Vec(vocabList, input):
    # 每个词的出现与否作为一个特征，每个词只能出现一次，这称为词集模型

    returnVec = [0] * len(vocabList)   # 先创建一个全0向量

    for word in input:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("{} 没有出现在词汇表中".format(word))
    return returnVec

def bagOfWords2VecMN(vocabList, input):
    # 每个词的出现次数作为一个特征，每个词可出现多次，这称为词袋模式
    returnVec = [0] * len(vocabList)

    for word in input:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# training, 根据现有数据得到各种概率,  参数类型要求为nparray
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)   # 计算侮辱性留言的概率
    p0Num, p1Num = np.ones(numWords), np.ones(numWords) # 初始化条件概率的分子    为了解决 “某个词的条件概率为0则p(w1..wx|ci)就会为0” 的问题，将所有词的出现次数初始化为1，分母初始化为2
    p0Denom, p1Denom = 2.0, 2.0                           # 初始化条件概率的分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]         # 类别为1的文档中，词汇表中所有词出现的次数统计
            p1Denom += sum(trainMatrix[i])  # 类别为1的文档中，所有词一共出现了多少次
        else:
            p0Num += trainMatrix[i]         # 类别为0的文档中，词汇表中所有词出现的次数统计
            p0Denom += sum(trainMatrix[i])  # 类别为0的文档中，所有词一共出现了多少次

    p1Vect = np.log(p1Num / p1Denom)        # p1Vect[i]：词汇表中下标为i的词在类别1中的条件概率，即 p(wi|c1)
    p0Vect = np.log(p0Num / p0Denom)        # p0Vect[i]：词汇表中下标为i的词在类别0中的条件概率，即 p(wi|c0)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):  #pClass1表示类别1的概率，即p(c1)
    # 分别计算 p0 和 p1， 两者谁大，该文档就属于哪类
    # vec2Classify * p1Vec (对应元素相乘) 表示要分类的文档中出现的单词的条件概率
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)     #log(a*b) = log(a) + log(b)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataset()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pClass1 = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pClass1))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pClass1))

def textParse(bigString):
    # 加入正则化，分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W', bigString)
    # 若遇到URL这样的字符串，会出现很多长度小于3的字符串且不是单词，如parameter=en, 所以过滤掉小于3的字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    # docList以文档为单位存储所有文档，fullText以单词为单位存储所有文档中出现的单词
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        # print(i)
        wordList = textParse(open('email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)
    # 分割训练集和测试集（下标，之后通过下标在docList中找到对应的doc，注意：一个doc是一个样本）
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # trainingSet.pop(randIndex)
        del(trainingSet[randIndex])
    trainMat, trainClass = [], []   # trainMat是训练样本集，每一个样本的形式是一个词向量（大小为词汇表长度的向量，元素表示词汇表对应位置的词在所有文档中出现的次数）
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0

    for docIndex in testSet:
        word2Vec = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(word2Vec), p0V, p1V, pSpam) != classList[docIndex]:
            print("error doc : ", docList[docIndex], " correct class:", classList[docIndex])
            errorCount += 1

    return float(errorCount) / len(testSet)
    # print("the error rate is : ", float(errorCount) / len(testSet))

def calcMostFreq(vocabList, fullText):
    # 计算fullText的词在vocabList中的出现频率
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  # 将迭代器中的tuple按每个tuple中的第1个index元素逆序排序(从0开始)
    return sortedFreq[:30]    # 返回的是迭代器，形式为[('a',1), ('b',2),...]

def localWords(feed1, feed0):
    docList, classList, fullText = [], [], []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print(minLen)
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        # pairW[0]代表单词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    print(trainingSet)
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        print(randIndex)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        doc = docList[docIndex]
        trainMat.append(bagOfWords2VecMN(vocabList, doc))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVec, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is : ", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

if __name__ == "__main__":
    # listOPosts, listClass = loadDataset()
    # myVocabList = createVocabList(listOPosts)

    # print(myVocabList.index('stupid'))
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    # trainMat的一个元素代表一句话的向量，一共6句话，每句话向量的长度都是32，即词汇表的长度，0代表该词未出现，1代表该词出现

    # p0V, p1V, pAb = trainNB0(trainMat, listClass)
    # print(pAb)
    # print(p0V)
    # print(p1V)
    # testingNB()
    # epochs = 10
    # error = 0.0
    # for epoch in range(epochs):
    #     error_rate = spamTest()
    #     error += error_rate
    #     print("epoch ", epoch, ", error rate:", error_rate)
    #
    # print("The average error rate: ", error/epochs)
    ny = feedparser.parse('https://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xmls')
    print(len(ny['entries']))
    print(len(sf['entries']))
    vocabList, pSF, pNY = localWords(ny, sf)