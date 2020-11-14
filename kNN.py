from numpy import *
import operator

def createDataSet():#一个简单数据集创建
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):#不限维数，推荐低维数据
    #距离计算
    dataSetSize=dataSet.shape[0]#读取矩阵行数，即数据集中样本数
    diffMat=tile(inX,(dataSetSize,1))-dataSet#建立差值矩阵，用于后续计算
    sqDiffMat=diffMat**2#平方
    sqDistances=sqDiffMat.sum(axis=1)#对二维数组/矩阵而言，sum(axis=1)为矩阵内矩阵元素求和，即对数据集差值矩阵中的每个样本与待测分类数据的坐标差求和
    distances=sqDistances**0.5#开方
    sortedDistIndicies=distances.argsort()#argsort()返回排序后的索引值，即下标，可以根据下标追溯数据
    classCount={}#建立空 字典

    #选取距离最小k个点排序
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]#按已经排序的索引值/下标取出对应标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#标签计数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#in py3.x dict.iteritems() is dict.item()
    #根据之前的循环计数排序，进行分类
    return sortedClassCount[0][0]

def file2matrix(filename):#txt文本转矩阵，按行处理，返回数据和标签
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOflines=len(arrayOLines)
    returnMat=zeros((numberOflines,12))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split("	")
        d=len(listFromLine)
        returnMat[index,0:d-1]=listFromLine[1:d]
        classLabelVector.append(listFromLine[0])
        index+=1
    return returnMat,classLabelVector

