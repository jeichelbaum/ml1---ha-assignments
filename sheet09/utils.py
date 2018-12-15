import numpy

def getMNIST56(path='data'):

    Xtrain = numpy.fromfile(open('%s/train-images-idx3-ubyte'%path,'r'),dtype='ubyte',count=16+784*60000)[16:].reshape([60000,784])
    Ttrain = numpy.fromfile(open('%s/train-labels-idx1-ubyte'%path,'r'),dtype='ubyte',count=8+60000)[8:]

    Xtest = numpy.fromfile(open('%s/t10k-images-idx3-ubyte'%path,'r'),dtype='ubyte',count=16+784*10000)[16:].reshape([10000,784])
    Ttest = numpy.fromfile(open('%s/t10k-labels-idx1-ubyte'%path,'r'),dtype='ubyte',count=8+10000)[8:]

    # Extract handwritten digit 5 or 6 and label them as (+1/-1)
    Xtrain = Xtrain[(Ttrain==5)|(Ttrain==6)]
    Ttrain = Ttrain[(Ttrain==5)|(Ttrain==6)]
    Ttrain = 1.0*(Ttrain==5)-1.0*(Ttrain==6)

    Xtest = Xtest[(Ttest==5)|(Ttest==6)]
    Ttest = Ttest[(Ttest==5)|(Ttest==6)]
    Ttest = 1.0*(Ttest==5)-1.0*(Ttest==6)

    m = Xtrain.mean(axis=0)

    Xtrain = Xtrain - m
    Xtest  = Xtest  - m

    s = Xtrain.std()

    Xtrain = Xtrain / s
    Xtest  = Xtest  / s

    R = numpy.random.mtrand.RandomState(1234).permutation(len(Xtrain))[:1000]

    return Xtrain[R],Ttrain[R],Xtest,Ttest

