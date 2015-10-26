# -*- coding: cp936 -*-
import csv
import random
import math
import cPickle
def loadCsv(filename):
        lines = csv.reader(open(filename, "rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset
def splitDataset(dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
                index = random.randrange(len(copy))
                trainSet.append(copy.pop(index))
        return [trainSet, copy]
def separateByClass(dataset):
        separated = {}
        for i in range(len(dataset)):
                vector = dataset[i]
                if (vector[-1] not in separated):#字典里面没有的标签
                        separated[vector[-1]] = []
                separated[vector[-1]].append(vector)#加入到对应标签里面
        return separated
def computeTheta (numbers,classlen,features):#theta = (特征出现次数+平滑）/（类文本总长度+平滑（词汇表长度））
        return (sum(numbers)+1)/float(classlen+features)#每个特征计算theta
def summarize(dataset):
        classlen = 0
        #统计类文本总长度
        for attribute in dataset:
                classlen += sum(attribute)
        features = len(dataset[0])-1
        #通过zip函数提取dataset里面的每一列计算theta
        summarizes = [computeTheta(attribute,classlen,features) for attribute in zip(*dataset)]
        del summarizes[-1]
        return  summarizes
def summarizeByClass(dataset):
        separated = separateByClass(dataset)
        summaries = {}#summaries字典类型
        for classValue, instances in separated.iteritems():
                summaries[classValue] = summarize(instances)
        return summaries
def calculateProbability(x, theta):#注意x=0的情况，不能直接返回
        if x == 0:
                return 0
        else:
                return math.pow(theta,x)
def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}#probabilities字典类型
        for classValue, classSummaries in summaries.iteritems():#summarie字典类型
                probabilities[classValue] = 0#先验,待定
                for i in range(len(classSummaries)):
                        theta = classSummaries[i]
                        x = inputVector[i]
                        p = calculateProbability(x, theta)
                        if p != 0:
                                probabilities[classValue] += math.log(p)
        return probabilities
def predict(summaries, inputVector):
        probabilities = calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
                if bestLabel is None or probability > bestProb:
                        bestProb = probability
                        bestLabel = classValue
        return bestLabel
def getPredictions(summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
                result = predict(summaries, testSet[i])
                predictions.append(result)
        return predictions
def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
                if testSet[x][-1] == predictions[x]:
                        correct += 1
        return (correct/float(len(testSet))) * 100.0
def main(filename):
        splitRatio = 0.67
        dataset = loadCsv(filename)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
        # prepare model
        summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        accuracy = getAccuracy(testSet, predictions)
        print('result %r:') %(filename)
        print('Accuracy: {0}%').format(accuracy)
def trainModel(filename):
        dataset = loadCsv(filename)
        # prepare model
        summaries = summarizeByClass(dataset)
##        writer = csv.writer(open('NBCweight','wb'))
##        for classValue, weight in summaries.iteritems():
##                writer.writerow(summaries[classValue])
        f = open('NBCweight','wb')
        cPickle.dump(summaries, f)
        f.close()
if __name__ == '__main__':
    #main('vectorize-tfidf-967.csv')
    trainModel('vectorize-tfidf-967.csv')
