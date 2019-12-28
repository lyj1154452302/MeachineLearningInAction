import operator

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, '  ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup = 1):

    """

    :param dataSet: {frozen([...]): 1, frozen([...]): 1, frozen([...]): 1, frozen([...]): 1}
    :param minSup:
    :return:
    """
    headerTable = {}    # 项头表

    # 第一次扫描数据集，建立项头表
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]   # 统计每个 一项集 出现的频数

    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:    # 过滤小于最小支持度的 一项集
            del headerTable[k]

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable.keys():
        headerTable[k] = [headerTable[k], None]   # 改成{'a':[2, None], 'b':[5, None]}

    retTree = treeNode('Null Set', 1, None)
    # 第二次扫描数据集，将每一条事务记录按项头表的中一项集的频数排序
    for tranSet, count in dataSet.items():   # tranSet表示一条事务，count表示这个事务出现的频次
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        print(sorted(localD.items(), key=operator.itemgetter(1), reverse=True))
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=operator.itemgetter(1), reverse=True)]   # 按照项头表的频数从大到小排序事务中的项
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:     # 如果该节点有在inTree节点的孩子节点里，则节点计数加1
        inTree.children[items[0]].inc(count)
    else:                               # 如果该节点没有在inTree节点的孩子节点里，则创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)   # 若项头表中此节点没有头指针，则新建头指针
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:  # 如果该事务不止有一个节点，则以该节点为inTree节点，事务除该节点的剩余传入递归函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:  # 沿着节点链表找到链表末尾节点，在末尾插入新节点
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpleData():
    simpData = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpData

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:    # 根节点null是存在的，不是None
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)  # 上溯整棵FP子树

def findPrefixPath(basePat, treeNode):  # 寻找前缀路径
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:   # 注意这里不是大于0而是大于1，因为该节点的前缀路径是不需要包含该节点的，而prefixPath在上溯中必然会被包含
            condPats[frozenset(prefixPath[1:])] = treeNode.count   # 以该节点的计数值为 该前缀路径的值
        treeNode = treeNode.nodeLink

    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  # 按频次从小到大排序

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


if __name__ == '__main__':
    simpDat = loadSimpleData()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    # myFPtree.disp()
    prefixPath = findPrefixPath('r', myHeaderTab['r'][1])
    print(prefixPath)
