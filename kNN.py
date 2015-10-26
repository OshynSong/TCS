#encoding=utf-8
###########################################################
### @author OshynSong                                   ###
### @time   2014-12                                     ###
### @desc   kNN algorithm to classfication              ###
###########################################################

import os
import sys

from numpy import *
import operator
import random

FEATURES_NUM = 344
TRAINSET_NUM = 4860
TESTSET_NUM = 540

def randLoadDataSet(fp, featNum, trainNum, testNum):
    if os.path.exists(fp):
        try:
            fh = open(fp, 'r')
            rtnTrainSet = zeros((trainNum, featNum))
            trainLabel = []
            rtnTestSet = zeros((testNum, featNum))
            testLabel = []
            i = 0
            indexs = []
            while i < trainNum:
                r = 0
                while True:
                    r = random.randint(0, trainNum+testNum-1)
                    if indexs.count(r) > 0:
                        continue
                    else:
                        break
                indexs.append(r)
                i += 1
            i = 0; j = 0; k = 0
            lines = fh.readlines()
            for line in lines:
                line = line.strip()
                terms = line.split(',')
                if indexs.count(i) > 0:
                    rtnTrainSet[j,:] = terms[0:featNum]
                    trainLabel.append(int(terms[featNum]))
                    j += 1
                else:
                    rtnTestSet[k, :] = terms[0:featNum]
                    testLabel.append(int(terms[featNum]))
                    k += 1
                i += 1
        except Exception, msg:
            print 'An unexcepted error occur: ', msg
        finally:
            fh.close()
        return rtnTrainSet,trainLabel,rtnTestSet,testLabel
    else:
        print "The data file does not exists!"
        return None

def normalize(ds):
    minVals = ds.min(0)
    maxVals = ds.max(0)
    ranges = maxVals - minVals
    normDS = zeros(shape(ds))
    n = ds.shape[0]
    normDS = ds - tile(minVals, (n, 1))
    normDS = normDS / tile(ranges, (n, 1))
    return normDS

def kNNclassify(ds, labels, k, inputX):
    dsSize = ds.shape[0]
    diff = tile(inputX, (dsSize, 1)) - ds
    sqDiff = diff ** 2
    sqDist = sqDiff.sum(axis = 1);
    dist = sqDist ** 0.5
    sortedDist = dist.argsort()
    classCount = {}
    for i in range(k):
        votedLabel = labels[sortedDist[i]]
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    ''' print "Please import the module to other file to use!"'''
    
    a,b,c,d = randLoadDataSet('../vectorize/vectorize-tfidf-344.csv', 344, 4860, 540)
    a = normalize(a)
    c = normalize(c)
##    print len(a), len(b), len(d), len(d);
##    print type(c[4,:]),c[4,:],type(d)
##    print kNNclassify(a, b, 10, c[4,:])
##    exit()
    import matplotlib.pyplot as plt
    arrK = range(5, 50, 5)
    acc = []
    for k in arrK:
        cnt = 0.0
        for i in range(TESTSET_NUM):
            real = d[i]
            predicted = kNNclassify(a, b, k, c[i,:])
            if int(real) == int(predicted):
                cnt += 1
        acc.append(cnt / TESTSET_NUM)
    plt.xlim(0, 50)
    plt.ylim(0.5, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.plot(arrK, acc)
    plt.grid(True)
    plt.show()
    
