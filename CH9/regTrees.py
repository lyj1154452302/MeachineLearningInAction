import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将每行数据映射成浮点型数据

        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]    # 找出在feature位置上大于value的所有样本
    mat2 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]   # 找出在feature位置上小于等于value的所有样本
    return mat1, mat2

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]   # 总方差 = 方差 * 样本个数

def chooseBestSplit(dataSet, leafType, errType, ops):
    tolS = ops[0]   # 容许的最小误差下降值，即如果划分后的误差与划分前的误差的相差值在tolS之下，则不按此划分，而按划分前的所有样本的均值划分
    tolN = ops[1]   # 切分的最少样本数，即如果划分后的样本数少于tolN，则不按此方式划分
    # print(dataSet[:, -1])
    # print(set(dataSet[:, -1].T.tolist()[0]))
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # set只包含不重复的元素，所以如果set中的元素只有一个，说明当前dataset中的所有样本的y值都一样
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)    # 未划分前的样本总方差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)

            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}            # 存储四个东西 1.待切分的特征（特征索引） 2.待切分的特征值 3.左子树 4.右子树
    retTree['spInd'] = feat
    retTree['spVal'] = val

    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 针对只有两个feature的样本集，ex00.txt
def plotDataSet(filename):
    dataMat = loadDataSet(filename)
    n = len(dataMat)
    xcord = []
    ycord = []

    for i in range(n):
        xcord.append(dataMat[i][0])
        ycord.append(dataMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def plotDataSet1(filename):
    dataMat = loadDataSet(filename)
    n = len(dataMat)  # 样本总个数

    xcord = []
    ycord = []

    for i in range(n):
        xcord.append(dataMat[i][1])
        ycord.append(dataMat[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

def isTree(obj):
    """
    判断输入变量是否是一颗树
    :param obj:
    :return:
    """
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    """
    对树进行塌陷处理（即返回树的平均值）
    :param tree:
    :return: 树的平均值（该树的所有叶节点的平均值）
    """
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])

    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])

    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):          # 剪枝操作，自底向上对所有分支节点进行剪枝操作
    if np.shape(testData)[0] == 0:  # 如果剪枝到此刻，测试集的样本划分的样本个数为0，则进行塌陷处理
        return getMean(tree)

    if (isTree(tree['left'])) or (isTree(tree['right'])):   # 只要该分支节点有左子树或右子树，则根据该分支节点来划分测试集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if (isTree(tree['left'])):    # 对左子树进行剪枝操作
        tree['left'] = prune(tree['left'], lSet)

    if (isTree(tree['right'])):   # 对右子树进行剪枝操作
        tree['right'] = prune(tree['right'], rSet)

    if not isTree(tree['left']) and not isTree(tree['right']):    # 该分支节点只有叶子节点，则用最小二乘损失函数来判断是否要剪枝
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])   # 根据该分支节点划分测试数据集
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2)) # 未剪枝前的误差大小
        newMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - newMean, 2))

        if errorMerge < errorNoMerge:    # 如果两个叶节点合并后，误差比合并前小，则进行剪枝（即合并两个叶节点）
            print("merging")
            return newMean
        else:
            return tree    # 不需要剪枝，则返回当前tree
    else:
        return tree        # 当左右节点一个为子树一个为叶节点时，且子树不需要剪枝，就会是这个情况

def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]   # X的第一列初始固定为1，代表线性模型中的回归系数中的x0（常数项）
                                    # 疑问：这里为什么是dataSet[:, 0:n-1]而不是dataSet[:, 0:n] 0:n-1只有 0~n-2啊，最后一个特征是不是就被漏掉了？
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:  # 不可逆
        raise NameError('This matrix is singular, can not do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))   # 对输入数据进行格式化，其中添加一个特征值且初始为1的特征(回归系数中的x0)
    X[:, 1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

if __name__ == '__main__':
    # testMat = np.mat(np.eye(4))
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # print(mat1)
    # myDat = loadDataSet('ex00.txt')
    # myMat = np.mat(myDat)
    # tree = createTree(dataSet=myMat, ops=(0,1))
    # print(tree)
    # plotDataSet('ex2.txt')

    # myDat1 = loadDataSet('ex0.txt')
    # myMat1 = np.mat(myDat1)
    # tree = createTree(dataSet=myMat1)
    # plotDataSet1('ex0.txt')
    # print(tree)

    # myDat2 = loadDataSet('ex2.txt')
    # myMat2 = np.mat(myDat2)
    # myTree = createTree(myMat2, ops=(0,1))
    # myDatTest = loadDataSet('ex2test.txt')
    # myMat2Test = np.mat(myDatTest)
    # print(prune(myTree, myMat2Test))

    # myMat2 = np.mat(loadDataSet('exp2.txt'))
    # print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))

    trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))

    # 回归树
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    # 模型树
    myTree = createTree(testMat, modelLeaf, modelErr, (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    # 线性回归模型
    ws, X, Y = linearSolve(trainMat)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])



