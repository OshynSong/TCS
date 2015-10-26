#encoding=utf-8
###########################################################
### @author OshynSong                                   ###
### @time   2014-12                                     ###
### @desc   Rocchio algorithm to classfication          ###
###########################################################

import os
import sys
from numpy import *
import random

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

def train(ds, labels):
    classes = {}
    classLen = {}
    i = 0
    for l in labels:
        if classes.has_key(l):
            c = classes[l]
            classes[l] = c + ds[i,:]
            classLen[l] = classLen.get(l, 0) + 1
        else:
            classes[l] = ds[i,:]
            classLen[l] = 1
        i += 1
    for c in classes:
        classes[c] = classes[c] / classLen[c]
    return classes

def predict(classes, toBePredictX):
    maxDist = 0.0
    maxC = 0
    for c in classes:
        distX = (sum(toBePredictX ** 2)) ** 0.5
        distC = (sum(classes[c] ** 2)) ** 0.5
        if (distX * distC) == 0:
            continue
        dist = sum(toBePredictX * classes[c]) *1.0 / (distX * distC)
        if dist > maxDist:
            maxDist = dist
            maxC = c
    return maxC

if __name__ == "__main__":
    ''' print "Please import the module to other file to use!" '''
    
    a,b,c,d = randLoadDataSet('vectorize-tfidf-344.csv', 344, 4860, 540)
    ##print len(a), len(b), len(d), d
    
    cnt = 0.0
    for i in range(540):
        real = d[i]
        model = train(a, b)
        predicted = predict(model, c[i])
        if int(real) == int(predicted):
            cnt += 1
    print "Accuracy: ", cnt / 540
