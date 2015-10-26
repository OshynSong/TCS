import csv
import random
import subprocess
def loadCsv(filename):
        lines = csv.reader(open(filename, "rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset
def preprocess(filename):
    dataset = loadCsv(filename)
    #output the trainset
    outfilename = 'libsvm_testdata.txt'
    outfile = open(outfilename,'w')
    for item in dataset:
        type = item[-1]
        for i in range(len(item)-1):
            item[-i-1] = item[-i-2]
        item[0] = type
        print >> outfile, '%r ' %(item[0]),
        for i in range(len(item)-1):
            print>> outfile, '%r:%r ' %(i+1, item[i+1]),
        outfile.write('\n')
        
preprocess('vectorize-tfidf-967.csv')
p = subprocess.Popen('svm-scale -l 0 -s svmrange libsvm_testdata.txt > libsvm_testdata.scale.txt',\
                 shell = True, stdout=subprocess.PIPE)
retval = p.wait()
