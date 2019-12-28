import numpy as np
import operator
from os import listdir

# 复现KNN-手写数字识别

def img2Vec(filename):
    vec = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineArr = fr.readline()   # 这里是一个字符串，所以下面要用强制转换类型
        for j in range(32):
            vec[0, i*32+j] = int(lineArr[j])
    return vec

def files2Matrix(dirname):
    listOfFilename = listdir(dirname)
    dataMatrix, classLabels = [], []
    for filename in listOfFilename:
        dataMatrix.extend(img2Vec(dirname+'/'+filename))
        label = int(filename.strip().split('.')[0].split('_')[0])
        classLabels.append(label)
    return np.array(dataMatrix), classLabels  # 返回numpy数组，方便后续操作

def classify(input, dataMatrix, classLabels, k=3):
    # 计算input与dataMatrix中每个样本的欧式距离 每个特征差的平方和再开根号
    dataSize = len(dataMatrix)  # 一共有多少个训练样本
    input_ = np.tile(input, (dataSize, 1))
    diff = (input_ - dataMatrix) ** 2
    distances = np.sum(diff, 1) ** 0.5  # shape: dataSize,   一维
    sortedIndices = np.argsort(distances) # 得到distances从小到大排序后的下标

    # 从所有欧式距离中，取最短的前K个，距离最近的前K个样本中，取类别数多的为最终分类结果
    classCount = {}
    for i in range(k):
        label = classLabels[sortedIndices[i]]
        if label not in classCount.keys():
            classCount[label] = 1
        else:
            classCount[label] += 1
    sortedClass = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 返回从小大到排序的元素为元组的list

    return sortedClass[0][0]

def handwritingTest():
    trainingMatrix, trainingLabels = files2Matrix('trainingDigits')
    testMatrix, testLabels = files2Matrix('testDigits')

    errorCount = 0.0
    for data, label in zip(testMatrix, testLabels):
        if label != classify(data, trainingMatrix, trainingLabels, 3):
            errorCount += 1.0

    print('the test error rate is ', errorCount/float(len(testLabels)))

if __name__ == '__main__':
    handwritingTest()




