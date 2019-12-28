import numpy as np

def sigmoid(x):
    return 1.0 / (1+np.exp(-x))

def stocGradAscent(dataMatrix, classLabels, epochs=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)

    for e in range(epochs):
        trainingSet = list(range(m))

        for i in range(m):
            # alpha = 1 / (e + i) + 0.01   # 这样会出现除以0的情况
            alpha = 4 / (1.0 + e + i) + 0.01
            randIndex = int(np.random.uniform(0, len(trainingSet)))    # np.random.uniform 从均匀分布中随机采样，返回类型是float，有可能是小数，所以要用int()
            # print("dataMatrix[randIndex]", dataMatrix[randIndex])
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h

            weights = weights + alpha * dataMatrix[randIndex] * error
            del(trainingSet[randIndex])

    return weights

def classify(weights, data):
    h = sigmoid(sum(weights * data))
    if h > 0.5:
        return 1.0
    else:
        return 0.0

def trainAndTest(epoch):
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        lineArr = line.strip().split('\t')
        data = []
        for i in range(21):
            data.append(float(lineArr[i]))
        trainingSet.append(data)
        trainingLabels.append(float(lineArr[21]))
    # print(trainingSet)
    weights = stocGradAscent(np.array(trainingSet), trainingLabels, 500)

    errorCount = 0.0
    count = 0.0
    for line in frTest.readlines():
        count += 1.0
        lineArr = line.strip().split('\t')
        data = []
        for i in range(21):
            data.append(float(lineArr[i]))
        if int(classify(weights, data)) != int(float(lineArr[21])):
            errorCount += 1.0
    print("epoch ", epoch, " , the error rate is ", errorCount / count)
    return errorCount / count

if __name__ == "__main__":
    epochs = 10
    error = 0.0
    for i in range(epochs):
        error += trainAndTest(i)

    print("after ", epochs, " iteration, the error rate is ", error / float(epochs))