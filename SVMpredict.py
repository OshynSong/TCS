#encoding=utf-8
import csv
import random
import subprocess
from svmutil import *
def loadCsv(filename):
        lines = csv.reader(open(filename, "rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset
def preprocess(inputX):
    ###dataset = loadCsv(filename)
    #output the trainset
    outfilename = 'libsvm_testdata.txt'
    dataset = inputX
    outfile = open(outfilename,'w')
    for item in dataset:
        #type = item[-1]
        v = []
        v.append(4.0)
        for i in range(len(item)):
            #item[-i-1] = item[-i-2]
            v.append(item[i])
        #item[0] = type
        ##print >> outfile, '%r ' %(item[0]),
        ##outfile.write(str(v[0]) + ' ')
        print >> outfile, '%r ' %(v[0]),
        for i in range(len(v) - 1):
            print>> outfile, '%r:%r ' %(i+1, v[i+1]),
        outfile.write('\n')
def svm_pre(inputX):
    preprocess(inputX)
    p = subprocess.Popen('svm-scale -l 0 -s svmrange libsvm_testdata.txt > libsvm_testdata.scale.txt',\
                     shell = True, stdout=subprocess.PIPE)
    retval = p.wait()
def predict(inputX):
    ##svm_pre(inputX)
    preprocess(inputX)
    y, x = svm_read_problem('libsvm_testdata.txt')
    prob = svm_problem(y, x)
    model = svm_load_model('cgi-bin/svmmodel.model')
    p_label, p_acc, p_val = svm_predict(y, x, model)
    return p_label
if __name__ == '__main__':         
    predict('vectorize-tfidf-967.csv')
