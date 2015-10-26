#encoding=utf-8
###########################################################
### @author OshynSong                                   ###
### @time   2014-12                                     ###
### @desc   classify the article by jieba library       ###
###########################################################

import sys
sys.path.append('.')
import os
import re
import math
from numpy import *
import jieba

import kNN
import Rocchio
import NBCpredict
import SVMpredict
import ANNpredict

CLASS_LABEL = {
    'C000008':[1, u'财经'],
    'C000010':[2, 'IT'],
    'C000013':[3, u'健康'],
    'C000014':[5, u'体育'],
    'C000016':[4, u'旅游'],
    'C000020':[6, u'教育'],
    'C000022':[7, u'招聘'],
    'C000023':[8, u'文化'],
    'C000024':[9, u'军事']
    }
IDF_PATH = '../idf.txt'
FEAT_FILE = '../features/featsByDFandIG-967.txt'
CLASSIFY_FILE = '../vectorize/vectorize-tfidf-967.csv'

def splitWord(article):
    wordsList = []
    try:
        wordsList = list(jieba.cut(article))
    except Exception, msg:
        print 'An unexcepted error occur: ', msg
    validWords = []
    for a in wordsList:
        w = re.search(ur'[\w\u4e00-\u9fee]+', unicode(a))
        if w != None:
            validWords.append(a)
    del wordsList
    return validWords

def vectorArticleByTFIDF(article, featFile = FEAT_FILE, idfFile = IDF_PATH):
    feats = []
    idfDict = {}
    try:
        fF = open(featFile, 'r')
        for f in fF:
            feats.append(f.split(':')[0])
        vector = zeros((1, len(feats)))
        idfF = open(idfFile, 'r')
        for f in idfF:
            i = f.split(' ')
            idfDict[i[0]] = float(i[1].strip())
        words = splitWord(article)
        tfDict = {}
        for w in words:
            w = w.encode('utf-8')
            if tfDict.has_key(w):
                tfDict[w] = tfDict.get(w, 0) + 1
            else:
                tfDict[w] = 1
        i = 0
        for f in feats:
            if tfDict.has_key(f):
                idf = idfDict[f]
                v = 1.0*math.log10(1 + tfDict[f]) * idf
                vector[0, i] = v;
            else:
                vector[0, i] = 0
            i += 1
    except Exception, msg:
        print "Error :", msg
    finally:
        fF.close()
        idfF.close()
    return vector

def loadDataSet(fname, trainNum, featNum):
    ds = zeros((trainNum, featNum))
    labels = []
    fh = open(fname, 'rb')
    i = 0
    for line in fh:
        line = line.strip()
        feats = line.split(',')
        ds[i,:] = feats[0:featNum]
        labels.append(float(feats[featNum]))
        i += 1
    return ds, labels
    
def classifyBykNN(dataSet, labels, k, predictX):
    dataSet = kNN.normalize(dataSet)
    return kNN.kNNclassify(dataSet, labels, k, predictX)

def classifyByRocchio(dataSet, labels, predictX):
    classes = Rocchio.train(dataSet, labels)
    return Rocchio.predict(classes, predictX)

def classifyByNBC(predictX):
    return NBCpredict.predict(predictX)

def classifyBySVM(predictX):
    return SVMpredict.predict(predictX)

def classifyByANN(predictX):
    return ANNpredict.predict(predictX)

if __name__ == "__main__":
    ''' Classify the document using the given method '''
    ds, labels = loadDataSet(CLASSIFY_FILE, 5400, 967)
    fh = open('D:/e.txt', 'r')
    x = vectorArticleByTFIDF(fh.read())
    fh.close()
    
    ds = kNN.normalize(ds)
    c1 = int(classifyBykNN(ds, labels, 10, x))
    c2 = int(classifyByRocchio(ds, labels, x))
    ##print c1;exit()
    for c in CLASS_LABEL:
        cid = CLASS_LABEL[c][0]
        cname = CLASS_LABEL[c][1]
        if cid == c1:
            print 'Predicted by kNN: %s - %d(%s)' % (c, cid, cname)
        if cid == c2:
            print 'Predicted by Rocchio: %s - %d(%s)' % (c, cid, cname)
