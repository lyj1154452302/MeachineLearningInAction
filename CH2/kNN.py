import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # np.tile(a, (3, 1)) 将a按照纵向平铺复制3个a
    # print(np.tile(inX, (dataSetSize, 1)))
    # print(dataSet)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 按照样本数据集的样本个数复制输入数据，再与样本数据集做减运算，得到输入数据与样本数据集的每个特征的差值

    # 欧式距离计算，每个特征差值平方和再开根号
    sqDiffMat =diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # 在第二维上做sum操作
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()  # 按照元素从小到大排序，返回排序后对应元素的下标
    classCount = {}    # 统计前k个最近邻的样本中，每个class出现的次数

    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # classCount.items()是iterable类型，而iterable的元素是元组tuple，所以key=operator.itemgetter(1)表示按照元组的第二个元素排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 逆序排序，从大到小
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabels = []
    index = 0
    for line in lines:
        lineArr = line.strip().split('\t')
        returnMat[index, :] = lineArr[0:3]
        classLabels.append(int(lineArr[-1]))
        index += 1
    return returnMat, classLabels

def autoNorm(dataSet):   # 归一化公式: newValue = (oldValue - min) / (max - min)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10  # 测试集数据占比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 数据预处理
    normMat, ranges, minVals = autoNorm(datingDataMat)      # 数据特征值归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)   # 测试集数据量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:", classifierResult, ", the real answer is:", datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is:", errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)   # 注意要把输入值特征做归一化处理，且训练的数据是归一化之后的数据
    print("You will probably like this person: ", resultList[classifierResult - 1])

def img2Vector(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')   # 从目录文件获取所有文件名
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileName = fileNameStr.split('.')[0]
        classLabel = int(fileName.split('_')[0])
        trainingMat[i, :] = img2Vector('trainingDigits/'+fileNameStr)
        hwLabels.append(classLabel)

    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileName = fileNameStr.split('.')[0]
        classLabel = int(fileName.split('_')[0])
        testData = img2Vector('testDigits/'+fileNameStr)
        predict = classify0(testData, trainingMat, hwLabels, 3)
        print("the classifier came back with: ", predict, ", the real answer is: ", classLabel)
        if predict != classLabel:
            errorCount += 1.0
    print("the total error rate is :", errorCount / float(mTest))


if __name__ == "__main__":
    # group, labels = createDataSet()
    # print(classify0([0,0], group, labels, 3))
    dataMatrix, classLabels = file2matrix("datingTestSet2.txt")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMatrix[:, 1], dataMatrix[:, 0], 15.0*np.array(classLabels), 15.0*np.array(classLabels))   # 根据label为分类样本上色
    # plt.show()
    # normMat, ranges, minVals = autoNorm(dataMatrix)
    #     # print(normMat)
    # datingClassTest()
    # classifyPerson()
    handwritingClassTest()