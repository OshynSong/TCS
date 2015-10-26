import NBC
import cPickle

def loadModel():
    global summaries
    f = open('cgi-bin/NBCweight.txt','r')
    summaries = cPickle.load(f)

def predict(inputX):
    #testSet = NBC.loadCsv(filename)
    loadModel()
    testSet = inputX
    # make predictions
    predictions = NBC.getPredictions(summaries, testSet)
    ##accuracy = NBC.getAccuracy(testSet, predictions)
    ##print accuracy
    return predictions

if __name__ == '__main__':
    loadModel()
    predict('vectorize-tfidf-967.csv')
