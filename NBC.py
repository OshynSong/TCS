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
                if (vector[-1] not in separated):#�ֵ�����û�еı�ǩ
                        separated[vector[-1]] = []
                separated[vector[-1]].append(vector)#���뵽��Ӧ��ǩ����
        return separated
def computeTheta (numbers,classlen,features):#theta = (�������ִ���+ƽ����/�����ı��ܳ���+ƽ�����ʻ���ȣ���
        return (sum(numbers)+1)/float(classlen+features)#ÿ����������theta
def summarize(dataset):
        classlen = 0
        #ͳ�����ı��ܳ���
        for attribute in dataset:
                classlen += sum(attribute)
        features = len(dataset[0])-1
        #ͨ��zip������ȡdataset�����ÿһ�м���theta
        summarizes = [computeTheta(attribute,classlen,features) for attribute in zip(*dataset)]
        del summarizes[-1]
        return  summarizes
def summarizeByClass(dataset):
        separated = separateByClass(dataset)
        summaries = {}#summaries�ֵ�����
        for classValue, instances in separated.iteritems():
                summaries[classValue] = summarize(instances)
        return summaries
def calculateProbability(x, theta):#ע��x=0�����������ֱ�ӷ���
        if x == 0:
                return 0
        else:
                return math.pow(theta,x)
def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}#probabilities�ֵ�����
        for classValue, classSummaries in summaries.iteritems():#summarie�ֵ�����
                probabilities[classValue] = 0#����,����
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
