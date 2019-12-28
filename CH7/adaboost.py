import numpy as np

def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, thresIneq):
    """
    通过第dimen个特征及阈值threshVal来划分数据集
    :param dataMatrix:
    :param dimen: 第几个特征作为划分属性
    :param threshVal: 阈值
    :param thresIneq: 大于(lt)为-1还是小于(dt)为-1
    :return:
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 先初始化所有类别都为1

    if thresIneq == 'lt':
        retArray[np.where(dataMatrix[:, dimen] <= threshVal)] = -1.0    # 比对每个样本的第dimen个特征值是否小于等于阈值，是的话，改变对应样本的类别
    else:
        retArray[np.where(dataMatrix[:, dimen] > threshVal)] = -1.0     # 比对每个样本的第dimen个特征值是否大于阈值，是的话，改变对应样本的类别
    return retArray

def buildStump(dataArr, classLabels, D):
    """
    根据当前样本权重，寻找最佳的单层决策树
    :param dataArr:  样本
    :param classLabels: 所有样本的正确分类结果
    :param D: 所有样本的权重
    :return:
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)   # m 样本数， n 特征数
    numSteps = 10.0         # 总步数
    bestStump = {}          # 存储最好的单层决策树
    bestClassEst = np.mat(np.zeros((m, 1)))   # 存储最好的分类结果
    minError = np.inf       #
    for i in range(n):      # 遍历每个特征
        rangeMin = dataMatrix[:,i].min()   # 特征i的最小取值
        rangeMax = dataMatrix[:,i].max()   # 特征i的最大取值
        stepSize = (rangeMax - rangeMin) / numSteps   # 在取值上，每步的步长
        for j in range(-1, int(numSteps)+1):   # 除了整个取值范围之外，还添加了两个取值范围之外的值，即j=-1和int(numSteps)+1时
            for inequal in ['lt', 'gt']:       # 两种分类方式
                threshVal = (rangeMin + float(j) * stepSize)      # 阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)   # 根据该特征该阈值该分类方式获得分类结果
                errArr = np.mat(np.ones((m, 1)))                  # 错误数组，初始都为1（都错）
                errArr[np.where(predictedVals == labelMat)] = 0   # 预测正确的样本在错误数组相应位置将元素值置为0
                weightedError = D.T * errArr    # 1, m * m, 1 = 1, 1 计算加权错误率
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))

                if weightedError < minError:     # 加权错误率最小，记录最好的分类信息
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    alpha = 0.5 * log((1-error)/error)
    Di(t+1) = Di(t)*e^(-alpha) / sum(D)    正确分类时
            = Di(t)*e^(alpha) / sum(D)     错误分类时
    :param dataArr: shape : m, n
    :param classLabels: shape : 1, m   由1或-1组成
    :param numIt:  最大弱分类器数
    :return:
    """
    weakClassArr = []  # 存储所有弱分类器的信息
    m = np.shape(dataArr)[0]  # 样本数量
    D = np.mat(np.ones((m, 1))/m)  # 所有样本的权重，初始值都一样
    aggClassEst = np.mat(np.zeros((m, 1)))  # 记录总类别

    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)    # 得到该样本权重下最好的单层分类决策树   classEst shape: m, 1 由1或-1组成
        # print("D:", D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算该弱分类器的权重alpha   max(error, 1e-16))是防止除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst:", classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)   # np.multiply 是对应元素相乘
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst    # 计算总类别   m, 1
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0:  #  若错误率降到0则跳出循环
            break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadHorseData(filename):
    numFeat = len(open(filename).readline().split('\t'))   # 特征数量
    dataMat, labelMat = [], []
    fr = open(filename)
    for line in fr.readlines():
        data = []
        lineArr = line.split('\t')
        for i in range(numFeat-1):
            data.append(float(lineArr[i]))
        dataMat.append(data)
        labelMat.append(float(lineArr[-1]))

    return dataMat, labelMat

if __name__ == '__main__':
    # D = np.mat(np.ones((5,1))/5)
    # dataMat, classLabels = loadSimpData()
    # # best = buildStump(dataMat, classLabels, D)
    # # print(best)
    # classifierArray = adaBoostTrainDS(dataMat, classLabels, 9)
    # res = adaClassify([[0,0], [1,4]], classifierArray)
    # print(res)
    dataMat, classLabels = loadHorseData('horseColicTraining.txt')
    classifyArr = adaBoostTrainDS(dataMat, classLabels, 10)

    testDataMat, testClassLabels = loadHorseData('horseColicTest.txt')
    res = adaClassify(testDataMat, classifyArr)
    # print(res)
    print(np.mat(testClassLabels).T.shape)
    print(res.shape)
    errArr = np.mat(np.ones((np.mat(testDataMat).shape[0], 1)))
    errCount = errArr[res != np.mat(testClassLabels).T].sum()
    print(errCount)
