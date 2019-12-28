import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat, labelMat = [], []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 设置每个样本的 x0=1.0（bias的系数）, x1, x2系数
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error   #  疑问： 梯度为什么是dataMatrix.transpose() * error ???
    return weights

def plotBestFit(wei):
    weights = wei.getA()      # 获取numpy.ndarray形式
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]    # 样本个数
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):  # 随机梯度上升法  一次仅有一个样本来更新回归系数，其实也就是一个epoch，然后每个样本都更新一次回归系数
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n).astype(np.float64)
    print(n)

    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights))   # h和error都是数值而不是向量
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):  # dataMatrix类型必须为np.array
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):                       # i表示本次迭代中第i个选出来的样本，而参与计算并用于更新梯度的样本是下标为randIndex的样本
            alpha = 4 / (1.0 + j + i) + 0.01     # 学习速率alpha在每次迭代时都需要调整，不断变小
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):  # 定义分类函数, 输入是测试样本的特征和已经训练好的回归系数, 他们的shape一致
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        # 每行有21个特征，1个label
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []   # 存储每个样本的特征
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: ", errorRate)
    return errorRate

def Test(epochs):
    errorSum = 0.0
    for epoch in range(epochs):
        errorSum += colicTest()
    print("after ", epochs, " epochs, the average error rate is:", errorSum/float(epochs))


if __name__ == "__main__":
    # dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)
    # weights = stocGradAscent0(dataArr, labelMat)
    # weights = stocGradAscent1(np.array(dataArr), labelMat)
    # plotBestFit(weights)
    # print(dataArr)
    # print(labelMat)
    # print(type(weights))
    # print(weights.shape)
    # print(weights)
    # print(np.mat(weights).shape)
    # plotBestFit(np.mat(weights).transpose())
    Test(10)