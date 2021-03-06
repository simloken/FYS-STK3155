from solvers import TensorFlow, LogisticRegression, Multinomial, DecisionTree, RandomForest, SupportVector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
"""
    TF is a function that handles all of our Tensorflow and keras modules.
    It takes the following inputs:
        
    inputs : array-like
            An array that effectively works as our design matrix
    labels : array-like
            The values we wish to map inputs onto
    Type : string
            The method or type of solver we wish to use.
            Must be {'sgd', 'adagrad', 'adam', 'nadam'}
    nlayers : int or array-like
            The neurons on a given layer. If nlayers is an array, then a given
            index i corresponds to the layer i.
            Must be same length as alayers and pens
    alayers : string or array-like
            The activation function for a given layer. If alayer is an array
            then a given index i corresponds to the layer i.
            Must be same length as nlayers and pens
            Must be {'sig', 'sm', 'sp', 'relu', 'elu', 'tanh'}
    outa : string
            The final activation function to call.
            Must be {'sig', 'sm', 'sp', 'relu', 'elu', 'tanh'}
    pens : string
            The penalty function of a given layer. If pens is an array then
            a given index i corresponds to the layer i.
            Must be same length as nlayers and alayers
            Must be {'l1', 'l2', 'l1l2'}
    epochs : integer
            Times to loop over a number of iterations.
    batchSize : integer
            Random points to take. We then use that random data
            to obtain a gradient.
    learns : float or array-like
            Different learning rates to try
            If learns is a float then so must lmbds, and one_return must be
            True.
    lmbds : float or array like
            Different parameters to try
            If lmbds is a float then so must learns, and one_return must be
            True.
    one_return : boolean
            Is False by default. Should be True if and only if learns and lmbds
            are floats.
    plotting : boolean
            Is False by default
            Should be True if and only if learns and lmbds are both array-likes.
            If True, returns a plot showing the accuracy for the different
            lmbds and learns values given.  
    retModel : boolean
            Is False by default
            Returns the model instead of accuracy scores. one_return should also
            be true if this is true
"""
def TF(inputs,
       labels,
       Type = 'nadam',
       nlayers = [70,50,30,10],
       alayers = ['sig', 'sig', 'sig', 'sig'],
       outa = 'sm',
       pens = ['l2', 'l2', 'l2', 'l2'],
       epochs = 70,
       batchSize = 50,
       learns=np.logspace(-4,-2,3),
       lmbds=np.logspace(-6,-3,12),
       one_return=False,
       plotting=True,
       retModel=False):
    
    np.random.seed(2)
    
    if outa.lower() == 'sm':
        outa = 'softmax'
    elif outa.lower() == 'sig':
        outa = 'sigmoid'
        
    out = len(np.unique(labels))
    model = TensorFlow(Type, nlayers, alayers, outa, pens, out, retModel)
    scores = model.fitter(inputs, labels, epochs, batchSize, learns, lmbds, one_return)
    if retModel == True:
        return scores
    if one_return==False and plotting==True:
        plt.figure()
        for i in range(scores.shape[0]):
            plt.semilogx(lmbds, scores[i][:], label=r'$\eta = {}$'.format(learns[i]))
        plt.legend()
        plt.title(r"""Accuracy for all different $\eta$ with varying $\lambda$
using the {} method""".format(model.Func))
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Accuracy [%]')
        plt.show()
        
        
"""
    The LogisticRegression handler
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    runs : int
        the number of runs to average over
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
        if set to true, then number of runs is ignored (as it returns the model)
"""            
def LR(
       inputs,
       labels,
       runs,
       scaled,
       retModel=False):
    
    score = []
    model = LogisticRegression(inputs,labels,scaled,retModel)
    if retModel == False:
        for i in range(runs):
            score.append(model.fitter())
        return np.mean(score), runs
    else:
        return model.fitter()

"""
    The NaiveBayes handler
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    alpha : float
        a parameter. must be greater than (or equal to) 0
    runs : int
        the number of runs to average over
    retModel : boolean
        whether or not to return the model instead of just a set of values.
        if set to true, then number of runs is ignored (as it returns the model)
"""        
def NB(
    inputs,
    labels,
    alpha=1,
    runs=1,
    retModel=False):
    
    score = []
    if retModel==False:
        for i in range(runs):
            
            X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2) 
            model = Multinomial(alpha=0.9)
            model.fitter(X_train, z_train)
            score.append(model.accuracy(X_test, z_test))
            
        return np.mean(score), runs
    else:
        X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2) 
        model = Multinomial(alpha=0.9)
        model.fitter(X_train, z_train)
        return model
"""
    The DecisionTree handler
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    runs : int
        the number of runs to average over
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
        if set to true, then number of runs is ignored (as it returns the model)
"""    
def DT(
       inputs,
       labels,
       runs,
       scaled,
       retModel=False):
    
    score = []
    model = DecisionTree(inputs,labels,scaled,retModel)
    if retModel == False:
        for i in range(runs):
            score.append(model.fitter())
        return np.mean(score), runs
    else:
        return model.fitter()
"""
    The RandomForest handler
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    runs : int
        the number of runs to average over
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
        if set to true, then number of runs is ignored (as it returns the model)
"""    
def RF(
       inputs,
       labels,
       runs,
       scaled,
       retModel=False):
    score = []
    model = RandomForest(inputs,labels,scaled,retModel)
    if retModel == False:
        for i in range(runs):
            score.append(model.fitter())
        return np.mean(score), runs
    
    else:
        return model.fitter()
    
"""
    The SupportVector handler
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    runs : int
        the number of runs to average over
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
        if set to true, then number of runs is ignored (as it returns the model)
"""        
def SV(
       inputs,
       labels,
       runs,
       scaled,
       retModel=False):
    
    score = []
    model = SupportVector(inputs,labels,scaled,retModel)
    if retModel == False:
        for i in range(runs):
            score.append(model.fitter())
        return np.mean(score), runs
    else:
        
        return model.fitter()

"""
    Our function for testing and casting predictions based on a set of input
    strings for all different methods
    
    cats : array-like
        an array of (y x 57), where y is the number of strings. Run the strings through
        string2cats before they go through here.
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    runs : int
        the number of runs to average over.
"""
def testAgainstStrings(
        cats,
        inputs,
        labels,
        runs):
    k = 0
    if np.shape(cats) != (inputs.shape[1],):
        store = np.zeros((np.shape(cats)[0], 5))
        for j in cats:
            lst1, lst2, lst3, lst4, lst5 = [], [], [], [], []
            for i in range(runs):
                j = j.reshape(1, -1)
                modelLR = LR(inputs, labels, 1, True, retModel=True)
                modelNB = NB(inputs, labels, 1, 1, True)
                modelDT = DT(inputs, labels, 1, False, retModel=True)
                modelRF = RF(inputs, labels, 1, False, retModel=True)
                modelSV = SV(inputs, labels, 1, False, retModel=True)
                lst1.append(modelLR.predict(j))
                lst2.append(modelNB.predictor(j))
                lst3.append(modelDT.predict(j))
                lst4.append(modelRF.predict(j))
                lst5.append(modelSV.predict(j))
            store[k, 0] = np.mean(lst1)
            store[k, 1] = np.mean(lst2)
            store[k, 2] = np.mean(lst3)
            store[k, 3] = np.mean(lst4)
            store[k, 4] = np.mean(lst5)
            
            k += 1
            
        return store.T
    else:
        cats = cats.reshape(1, -1)
        store = np.zeros((5))
        lst1, lst2, lst3, lst4, lst5 = [], [], [], [], []
        for i in range(runs):
            modelLR = LR(inputs, labels, 1, True, retModel=True)
            modelNB = NB(inputs, labels, 1, 1, True)
            modelDT = DT(inputs, labels, 1, False, retModel=True)
            modelRF = RF(inputs, labels, 1, False, retModel=True)
            modelSV = SV(inputs, labels, 1, True, retModel=True)
            lst1.append(modelLR.predict(j))
            lst2.append(modelNB.predictor(j))
            lst3.append(modelDT.predict(j))
            lst4.append(modelRF.predict(j))
            lst5.append(modelSV.predict(j))
        store[0] = np.mean(lst1)
        store[1] = np.mean(lst2)
        store[2] = np.mean(lst3)
        store[3] = np.mean(lst4)
        store[4] = np.mean(lst5)
        
        return store
                