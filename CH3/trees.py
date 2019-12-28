from math import log
import operator
from treePlotter import retrieveTree
import pickle

def calcShannonEnt(dataSet):   # 根据样本label 计算 H = 累加 i=1到n p(xi) * log(p(xi))  n为分类总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:      # 按 feature[axis]来分，把所有feature[axis]==value的分成一组
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])   # 去除这一特征，保留其余特征
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1    # 特征数量
    baseEntropy = calcShannonEnt(dataSet)   # 整个数据集的信息熵
    bestInfoGain = 0.0    # 信息增益最大值
    bestFeature = -1      # 哪个特征信息增益值最大

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)   # 该特征的所有取值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)   # 获得该特征取值为value的所有样本（样本中已去除该特征）
            prob = len(subDataSet) / float(len(dataSet))   # 此时数据集按 该特征值来分，则 p(value) = (feature == value的样本数) / 数据集总样本数
            newEntropy += prob * calcShannonEnt(subDataSet)  # Gain(D, axi) = Ent(D) - (累加i到V(|Dv|/|D|)*Ent(Dv))
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):   # 当特征已经分完，但类标签依然不是唯一的，则采用多数表决的方法决定该叶子节点的分类
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):   # labels是所有特征的标签，目的是给出数据明确的含义，分类之后可以给出解释
    copyLabels = labels.copy()
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):   # 数据集中所有样本均为同类，则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:                      # 数据集中所有特征已划分过，只剩类别标签，则通过投票决定当前叶子节点的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}     # 每个特征以自己为根节点，构造一颗子树
    del (copyLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 剩余的特征标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):    # tree的格式： {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    firstStr = list(inputTree.keys())[0]     # 根节点的特征名称
    secondDict = inputTree[firstStr]   # 根节点的子树（根节点特征的所有取值）
    featIndex = featLabels.index(firstStr)  # 找到根节点特征，在样本中所对应的位置
    for key in secondDict.keys():      # 遍历根节点特征的所有取值
        if testVec[featIndex] == key:   # 如果测试样本的该特征等于这个值
            if type(secondDict[key]).__name__ == 'dict':     # 如果该值下面还有子树（还有可分的特征），则递归调用，传入子树
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:                                            # 如果该值是叶子节点，则该测试样本就属于该叶子节点的类别
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw, 0)
    fw.close()

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    # myDat, labels = createDataSet()
    # myDat[0][-1] = 'maybe'
    # print(calcShannonEnt(myDat))
    # print(chooseBestFeatureToSplit(myDat))
    # print(createTree(myDat, labels))
    # myTree = retrieveTree(0)
    # print(classify(myTree, labels, [1, 0]))
    # print(classify(myTree, labels, [1, 1]))
    # storeTree(myTree, 'classifierStorage.txt')
    # print(grabTree('classifierStorage.txt'))
    fr = open('lenses.txt')
    trainingSet, testSet = [], []
    lines = fr.readlines()
    testLen = round(len(lines) * (1.0 - 0.2))
    trainingSet = [inst.strip().split('\t') for inst in lines[0:len(lines) - testLen]]
    testSet = [inst.strip().split('\t') for inst in lines[len(lines) - testLen:len(lines)]]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(trainingSet, lensesLabels)

    errotCount = 0.0
    for data in testSet:
        testVec = data[0:-1]
        testLabel = data[-1]
        result = classify(lensesTree, lensesLabels, data)
        if result != testLabel:
            errotCount += 1.0
    print('the error rate is:', errotCount/float(len(testSet)))