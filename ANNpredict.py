import bpnn
import csv

def loadCsv(filename):
        lines = csv.reader(open(filename, "rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset

def predict(inputfile):

    debug = True
    global classnum
    global featurenum
    global hiddennum

    inputweight = loadCsv('cgi-bin/inputweight')
    hiddenweight = loadCsv('cgi-bin/hiddenweight')
    hiddenbias = loadCsv('cgi-bin/hiddenbias')
    outputbias = loadCsv('cgi-bin/outputbias')
    # used to normalize the input data
    maxVector = loadCsv('cgi-bin/maxI')
    
    hiddenbias = hiddenbias[0]
    outputbias = outputbias[0]
    maxVector = maxVector[0]
    
    featurenum = 967
    classnum = 9
    hiddennum = 32
    
    # prepare model
    ann = bpnn.NN(featurenum, hiddennum, classnum)
    ann.loadnn(inputweight, hiddenweight, hiddenbias, outputbias)

    # normalization
    classset = []
    #dataset = loadCsv(inputfile)
    dataset = inputfile
    if debug == True:        
        for i in range(len(dataset)):
            classset.append(dataset[i][-1])
            for j in range(featurenum):
                dataset[i][j] = 2*(dataset[i][j])/(maxVector[j]) - 1
    predictions = []
    for p in dataset:
        resultarray = ann.update(p)
        result = 0
        for i in range(classnum):
            if resultarray[i] > result:
                result = resultarray[i]
                resultindex = i + 1
        predictions.append(resultindex)
##    #count the correct number
##    correct = 0
##    for x in range(len(dataset)):
##        #print ('class is %r, prediction is %r') %(testset[x][-1],predictions[x])
##        if classset[x] == predictions[x]:
##                correct += 1
##    print (correct/float(len(dataset))) * 100.0
    return predictions

if __name__ == '__main__':         
    predict('anntestdata.txt')
