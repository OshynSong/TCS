import math
import random
import string
import csv

random.seed(0)

def loadCsv(filename):
        lines = csv.reader(open(filename, "rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset

#split the data in to train and test set
def splitDataset(dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
                index = random.randrange(len(copy))
                trainSet.append(copy.pop(index))
        return [trainSet, copy]

# calculate the best hidden layer node number
def calculateHiddenNum(ni, no, alpha):
    return int(math.sqrt(ni+no)+alpha)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# Make a vector
def makeVector(J, fill=0.0):
    m = []
    for i in range(J):
        m.append(fill)
    return m

# our sigmoid function
def sigmoid(x):
    return 1/(1+math.exp(-x))

# derivative of our sigmoid function
def dsigmoid(y):
    return y*(1-y)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # bias for nodes
        self.bh = [1.0]*self.nh
        self.bo = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        
        # set them to random values
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # set bias to random values
        for i in range(self.nh):
                self.bh[i] = rand(-0.2, 0.2)
        for i in range(self.no):        
                self.bo[i] = rand(-0.2, 0.2)
        
        # weight gradient 
        self.gi = makeMatrix(self.ni, self.nh)
        self.go = makeMatrix(self.nh, self.no)

        # bias gradient
        self.gbh = makeVector(self.nh)
        self.gbo = makeVector(self.no)
        
    # load neural network from the file builded by matlab
    def loadnn(self, nwi, nwo, nbh, nbo):
        self.wi = nwi
        self.wo = nwo
        self.bh = nbh
        self.bo = nbo
        
    def update(self, inputs):
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni):#not include bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            # sum with bias
            sum = 0.0 + self.bh[j]
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            # sum with bias
            sum = 0.0 + self.bo[k]
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')
        
        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = self.ao[k] - targets[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weight gradient
        for j in range(self.nh):
            for k in range(self.no):
                self.go[j][k] += output_deltas[k]*self.ah[j]# gradient

        # update input weight gradient
        for i in range(self.ni):
            for j in range(self.nh):
                self.gi[j][k] += hidden_deltas[j]*self.ai[i] #gradient

        # update output bias gradient
        for i in range(self.no):
                self.gbo[i] += output_deltas[i]
                
        # update output bias gradient
        for i in range(self.nh):
                self.gbh[i] += hidden_deltas[i]
                
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, testset):
        predictions = []
        for p in testset:
            inputs = p[0:featurenum]
            targets = p[featurenum:]
            # the probability of each class
            resultarray = self.update(inputs)
            #print ('result array is %r') %(resultarray)
            result = 0
            resultindex = 0
            for i in range(classnum):
                    if resultarray[i] > result:
                            result = resultarray[i]
                            resultindex = i + 1
            predictions.append(resultindex)
        #count the correct number
        correct = 0
        for x in range(len(testset)):
                #print ('class is %r, prediction is %r') %(testset[x][-1],predictions[x])
                if testset[x][-1] == predictions[x]:
                        correct += 1
        return (correct/float(len(testset))) * 100.0

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=50, N=0.5, M=0.1, lamda=0.01, r = 10):
        self.lasterror = MAXERROR
        for i in range(iterations):
            error = 0.0
            # weight gradient 
            self.gi = makeMatrix(self.ni, self.nh)
            self.go = makeMatrix(self.nh, self.no)
            # bias gradient
            self.gbh = makeVector(self.nh)
            self.gbo = makeVector(self.no)
            print ('in interation:%r') %(i)
            lenp = len(patterns)
            for p in patterns:
                inputs = p[0:featurenum]
                targets = p[featurenum:]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                
            #update the weight
            for i in range(self.ni):
                for j in range(self.nh):
                    self.wi[i][j] -= N * (self.gi[i][j]/lenp + r*self.wi[i][j])
            for i in range(self.nh):
                for j in range(self.no):
                    self.wo[i][j] -= N * (self.go[i][j]/lenp + r*self.wo[i][j])
                    
            #update the bias
            for i in range(self.no):
                self.bo[i] -= N*(self.gbo[i]/lenp)
            for i in range(self.nh):
                self.bh[i] -= N*(self.gbh[i]/lenp)
                
            #dynamic update the learning rate
            if error >= self.lasterror:
                N = N * math.exp(-lamda)
                print 'update! new learning rate is %r' %(N)
            else:
                N = 1.05 * N
            self.lasterror = error
            print('error %-.5f' % error)

# process the data to the one vs all from
def processData(dataset, classnum):
    for i in range(len(dataset)):
        data = [0] * (featurenum + classnum)
        data[0:featurenum] = dataset[i][0:-1]
        dataset[i][-1] = int(dataset[i][-1])
        data[featurenum + dataset[i][-1] - 1] = 1
        dataset[i] = data
    return dataset

def main(filename):
    splitRatio = 0.67
    global classnum
    global featurenum
    global hiddennum
    global dataset
    #used to dynamic update the learning rate
    global MAXERROR
    MAXERROR = 0xffffffff
    #store the result
    global resultset
    
    #class number. Here we have 9 classes of text
    classnum = 9

    #used to choose the hidden layer number
    alpha = 3

    #load the file
    dataset = loadCsv(filename)

    #record the result
    resultset = []
    for a in dataset:
            resultset.append(a[-1])

    #feature number
    featurenum = len(dataset[0]) - 1

    #hidden layer node number
    hiddennum = calculateHiddenNum(featurenum,classnum,alpha)

    #split the data to the train set and test set
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    
    #process the result to the one vs all form
    ntrain = processData(trainingSet, classnum)
    
    #prepare model
    ann = NN(featurenum, hiddennum, classnum)

    #train the model
    ann.train(ntrain, 200, 0.05)

    #test the model
    accuracy = ann.test(testSet)
    print('Accuracy: {0}%').format(accuracy)

if __name__ == '__main__':
    main('vectorize-tf-344.csv')
