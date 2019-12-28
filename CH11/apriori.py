import numpy as np

def loadDataSet():
    return [[1,3,4], [2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)   # python3 中返回的是迭代器类型

def scanD(D, Ck, minSupport):
    """
    扫描数据集D，计算候选项集Ck各项的支持度，小于最小支持度的候选项将被过滤掉，最终得到 Lk
    :param D: 数据集
    :param Ck: 候选项集
    :param minSupport: 最小支持度
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.__contains__(can):   # 如果has_key返回的是false
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # print(D)
    numItems = float(len(D))   # 数据集中的记录总数
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  # 头插至列表
        supportData[key] = support  # 记录所有候选项集的支持度
    return retList, supportData

def aprioriGen(Lk, k):
    """
    生成 Ck 候选项集
    :param Lk: 第k个满足最小支持度的候选项集 Lk
    :param k: 要生成的第 几 个候选项集，这里的k比上一参数中的k大1
    :return:
    """
    retList = []
    lenLk = len(Lk)
    # 项集 两两比较然后结合
    for i in range(lenLk):
        L1 = list(Lk[i])[:k-2]  # 前 k-2个元素

        for j in range(i+1, lenLk):
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:   # 如果前k-2个元素相同，再合并，这样做是为了确保遍历次数最少（详见机器学习实战）
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = list(createC1(dataSet))
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    # print(L1)
    L = [L1]
    # print(L)
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


if __name__ == "__main__":
    dataSet = loadDataSet()
    # C1 = list(createC1(dataSet))
    # D = list(map(set, dataSet))
    # L1, suppData0 = scanD(D, C1, 0.5)
    # print(L1)
    L, supportData = apriori(dataSet)
    print(L)
    print("C1:", aprioriGen(L[0], 2))
    print("L1:", L[1])
    print("C2:", aprioriGen(L[1], 3))
    print("L2:", L[2])



