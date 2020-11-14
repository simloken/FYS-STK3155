from functions import FrankeFunction, X_Mat, MSE, R2, SGD, OLS, RIDGE, tcn
from functions import FFNN, LR, TensorFlow
from deprecated.deprecated import FFNN_old
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


np.random.seed(2)

digits = datasets.load_digits()

inputs = digits.images
labels = digits.target

n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
"""
    A function for comparing SGD, Ridge, and OLS.
    
    learn : float or array-like
            The initial learning rate
    epochs : integer or array-like
            Total amount of epochs
    itrs : integer or array-like
            Total iterations per epoch
    batchSize : integer or array-like
            Datapoints in a given batch
    o : integer
            Polynomial order of our design matrix
    Please note that only one input can be a array at a time.
"""
def soloSGD(learn,epochs,itrs,batchSize,o=10):
    np.random.seed(2)
    N = 1000
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    z = FrankeFunction(x,y)
    k = 0; l = 0; m = 0
    alpha = 0.9
    lst = [learn, epochs, itrs, batchSize]
    for i in lst:
        if hasattr(i, "__len__") == True:
            l = k
            array = i
            m += 1
        k += 1
        if m > 1:
            raise ValueError('Only one input can be an array-like')     
    
    if m != 0:
        if l == 0:
            error = np.zeros((o,len(learn)))
        if l == 1:
            error = np.zeros((o,len(epochs)))
        if l == 2:
            error = np.zeros((o,len(itrs)))
        if l == 3:
            error = np.zeros((o,len(batchSize)))
        for i in range(o):
            X = X_Mat(x,y,o)
            j = 0
            for k in array:
                    if l == 0:
                        variable = 'Learning Rate'
                        B = SGD(X,z,k,epochs,itrs,alpha,batchSize)
                        zpred = np.dot(X,B)
                        error[i][j] = MSE(z, zpred)
                    elif l == 1:
                        variable = 'Epochs'
                        B = SGD(X,z,learn,k,itrs,alpha,batchSize)
                        zpred = np.dot(X,B)
                        error[i][j] = MSE(z, zpred)
                    elif l == 2:
                        variable = 'Iterations'
                        B = SGD(X,z,learn,epochs,k,alpha,batchSize)
                        zpred = np.dot(X,B)
                        error[i][j] = MSE(z, zpred)
                    elif l == 3:
                        variable = 'Batch Size'
                        B = SGD(X,z,learn,epochs,itrs,alpha,k)
                        zpred = np.dot(X,B)
                        error[i][j] = MSE(z, zpred)
                        
                    j += 1
            
        plt.title('Mean-Squared-Error as a function of Polynomial Order\n with %s from %g to %g' %(variable, array[0], array[-1]))
        for i in range(len(error[1])):
            plt.plot(range(o), error[:,i], label='%s = %g' %(variable, array[i]))
            plt.xlabel('Polynomial Order')
            plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.show()
        for i in range(len(error[1])):
            print('MSE:', array[i], np.mean(error[:][i]))
    else:
        error = []
        for i in range(o):
            X = X_Mat(x,y,o)
            B = SGD(X,z,learn,epochs,itrs,alpha,batchSize)
            zpred = np.dot(X,B)
            error.append(MSE(z,zpred))
        plt.plot(range(o), error)
        plt.xlabel('Polynomial Order')
        plt.ylabel('MSE')
        plt.title('MSE of our Custom Stochastic Gradient Descent model on its own')
        plt.show()
"""
    A function for comparing SGD, Ridge, and OLS.
    
    learn : float
            The initial learning rate
    epochs : integer
            Total amount of epochs
    itrs : integer
            Total iterations per epoch
    alpha : float
            Tuning parameter
    batchSize : integer
            Datapoints in a given batch
    o : integer
            Polynomial order of our design matrix
"""    
def compareLight(learn,epochs,itrs,alpha,batchSize,o=10):
    np.random.seed(2)
    N = 1000
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    z = FrankeFunction(x,y)
    MSE1, MSE2, MSE3, MSE4 = [], [], [], []
    for i in range(o):
        X = X_Mat(x,y,o)
        B3 = SGD(X,z,learn,epochs,itrs,alpha,batchSize) #for some reason, OLS and RIDGE act weird
        zpred3 = np.dot(X,B3) #if this happens below them. Why I do not know.
        
        B = OLS(X, z)
        zpred = np.dot(X, B)
    
        B2 = RIDGE(X, z)
        zpred2 = np.dot(X,B2)
        
        sgdreg = SGDRegressor(max_iter = 3000, penalty='l2', eta0=0.05, learning_rate='adaptive', alpha=0.00001, loss='epsilon_insensitive', fit_intercept=False)
        sgdreg.fit(X,z)
        B4 = sgdreg.coef_
        zpred4 = np.dot(X,B4)
        
        
        MSE1.append(MSE(z,zpred))
        MSE2.append(MSE(z,zpred2))
        MSE3.append(MSE(z,zpred3))
        MSE4.append(MSE(z,zpred4)) 
        
    plt.plot(range(o), MSE1, label='SciKit OLS')
    plt.plot(range(o), MSE2, label='SciKit Ridge')
    plt.plot(range(o), MSE3, label='SGD')
    plt.plot(range(o), MSE4, label='SciKit SGD')
    plt.legend()
    plt.title('Mean-Squared Errors given polynomial order for different methods')
    plt.xlabel('Polynomial Order')
    plt.ylabel('MSE')
    plt.show()
"""
    Solves the MNIST system using a neural network
    
    neurons : integer
            Neurons for hidden layer
    aFunc : string
            Activation function
            Must be {'sm', 'relu', 'lrelu', 'sig', 'elu', 'sp'}
    inputs : array-like
            Our in-data, design matrix like
    labels : array-like
            the data we want to fit inputs onto
    eta_vals : array-like
            learning rate values to loop through
    lmbd_vals : array-like
            lambda parameter
    plotting : boolean
            dictates whether or not a plot should be made.
"""
def MNISTsolver(
        neurons,
        aFunc = 'sig',
        inputs=inputs,
        labels=labels,
        eta_vals = np.logspace(-5, 1, 7),
        lmbd_vals = np.logspace(-5, 1, 7),
        plotting=False
        ):
    np.random.seed(2)
    X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2)
    z_train_OH= tcn(z_train)
    
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    scores = np.zeros((DNN_numpy.shape))
    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = FFNN(X_train, z_train_OH, activation=aFunc, eta=eta, lmbd=lmbd, n_h_neurons=neurons)
            dnn.train()
            
            DNN_numpy[i][j] = dnn
            
            test_predict = dnn.predict(X_test)
            
            print("Learning rate  = ", eta)
            print("Lambda = ", lmbd)
            print("Accuracy score on test set: ", accuracy_score(z_test, test_predict))
            print()
            scores[i][j] = accuracy_score(z_test, test_predict)
    if plotting == True:
        plt.figure()
        for i in range(len(scores)):
            plt.loglog(lmbd_vals, scores[i][:], label=r'$\eta = {}$'.format(eta_vals[i]))
        plt.xlim([lmbd_vals[0], lmbd_vals[-1]+1000])
        plt.legend(loc='upper right')
        plt.title(r'Accuracy for all different $\eta$ with varying $\lambda$')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Accuracy [%]')
        plt.show()
"""
    The Logistic Regression function.
    Call from the console
    
    Type : string {'gd', 'sgd', 'sgdmb'}
            The solver type
    aFunc : string {'sm', 'relu', 'lrelu', 'sig', 'sp', 'elu', 'tanh'}
            The activation function
    penalty : string {'l1', 'l2'}
            The penalty function. Doesn't actually do anything, but left here
            for 'postarity's sake'.
    designOrder : int or array-like
            The polynomial degree of our design matrix X
    iters : int or array-like
            Number of iterations in each epoch
    epochs : int or array-like
            Total epochs over each set of iterations
    alpha : float
            Parameter value
    kappa : float
            Sharpness parameter value
    batchSize : int or array-like
            Data size when using SGD with minibatches.
    learn : float or array-like
            The initial learning rate of our model.
    plotting : boolean
            returns a plot if plotting = True.
            You should only use this is one of the inputs is an array-like
    bestFound : boolean
            Automatically assigns the best possible values (that I've found)
            and activation function for each method, and returns them.
            
    Only compatible with one array-like at a time, that is, you can't
    have both learn and iters be an array at the same time.
    
    Returns the MSE and R2 score of whatever was calculated
"""
def LogisticRegression(
    Type,
    aFunc = 'sm',
    penalty = 'l2',
    designOrder = 5,
    iters = 600,
    epochs = 200,
    alpha = 0.9,
    kappa = 1,
    batchSize = 50,
    learn = 0.001,
    plotting = False,
    bestFound = False):
    
    np.random.seed(2)
    k = 0; l = 0; m = 0
    vari = False
    lst = [designOrder, iters, batchSize, learn, epochs]
    for i in lst:
        if hasattr(i, "__len__") == True:
            vari = True
            l = k
            array = i
            m += 1
        k += 1
        if m > 1:
            raise ValueError('Only one input can be an array-like')     
    if bestFound == True: #will use the best parameters found by me whilst coding and testing
        print('Fitting with best possible parameters\n') #note that these can probably be optimized further, these are just some examples of good results
        if Type.lower() == 'gd': #gives an MSE of about ~ 0.02
            designOrder = 5
            aFunc = 'relu'
            penalty = 'l2'
            iters = 800
            epochs = 500
            alpha = 0.6
            learn = 1
            bestArray = [iters, epochs, alpha, learn]
            bestFunc = 'Rectified Linear Unit'
        elif Type.lower() == 'sgdmb': #gives an MSE of about ~ 0.07
            designOrder = 5
            aFunc = 'elu'
            penalty = 'l2'
            iters = 400
            alpha = 0.8
            epochs = 200
            batchSize = 50
            learn = 0.0005
            bestArray = [iters, epochs, alpha, batchSize, learn]
            bestFunc = 'Exponential Linear Unit'
        else: #assumes normal SGD w/o mini batches, MSE of about ~ 0.11
            designOrder = 5
            aFunc = 'sp'
            penalty = 'l2'
            iters = 600
            epochs = 200
            alpha = 0.7
            learn = 0.006
            kappa = 2.3
            bestArray = [iters, epochs, alpha, kappa, learn]
            bestFunc = 'Softplus'
    N = 1000
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    z = FrankeFunction(x,y)  
    ploter = plt.plot
    if vari == True:
        Blst = []
        for i in array:
                if l == 0:
                    variable = 'Polynomial Order'
                    X = X_Mat(x,y,i)
                    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
                    model = LR(X, z, Type, aFunc, iters, epochs, penalty, alpha, k, batchSize)
                    model.fitter(X_train, z_train, learn)
                    Blst.append(model.B)
                elif l == 1:
                    variable = 'Iterations'
                    X = X_Mat(x,y,designOrder)
                    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
                    model = LR(X, z, Type, aFunc, i, epochs, penalty, alpha, k, batchSize)
                    model.fitter(X_train, z_train, learn)
                    Blst.append(model.B)
                elif l == 2:
                    variable = 'Batch Size'
                    X = X_Mat(x,y,designOrder)
                    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
                    model = LR(X, z, Type, aFunc, iters, epochs, penalty, k, alpha, i)
                    model.fitter(X_train, z_train, learn)
                    Blst.append(model.B)
                elif l == 3:
                    variable = 'Learning Rate'
                    X = X_Mat(x,y,designOrder)
                    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
                    model = LR(X, z, Type, aFunc, iters, epochs, penalty, alpha, k, batchSize)
                    model.fitter(X_train, z_train, i)
                    Blst.append(model.B)
                    ploter = plt.semilogx
                elif l == 4:
                    variable = 'Epochs'
                    X = X_Mat(x,y,designOrder)
                    X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
                    model = LR(X, z, Type, aFunc, iters, i, penalty, alpha, k, batchSize)
                    model.fitter(X_train, z_train, learn)
                    Blst.append(model.B)
        msestore = []
        for i in range(len(array)):
            zpred4 = np.dot(X,Blst[i])
            print('\n%s: %g\nMSE: %g' %(variable, array[i], MSE(z,zpred4)))
            msestore.append(MSE(z,zpred4))
        if plotting == True:
            plt.title('Mean-Squared-Error as a function of %s\nfrom %g to %g' %(variable, array[0], array[-1]))
            ploter(array, msestore)
            plt.xlabel('%s' %(variable))
            plt.ylabel('MSE')
            plt.show()
    else:
        X = X_Mat(x,y,designOrder)
        X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
        model = LR(X, z, Type, aFunc, iters, epochs, penalty, alpha, kappa, batchSize)
        model.fitter(X_train, z_train, learn)
        B4 = model.B
        zpred4 = np.dot(X,B4)
        print('MSE: %g  ||  R2: %g' %(MSE(z,zpred4), R2(z, zpred4)))
        if bestFound == True:
            if Type.lower() == 'sgdmb':
                print('\nActivation Function: %s\n%s\nUsing the variables:\nIterations: %g\nEpochs: %g\nAlpha: %g\nBatch Size: %g\nLearning Rate: %g' 
                      %(bestFunc,'='*45, bestArray[0],bestArray[1], bestArray[2], bestArray[3], bestArray[4]))
            elif Type.lower() == 'sgd':
                print('Activation Function: %s\n%s\nUsing the variables:\nIterations: %g\nEpochs: %g\nAlpha Parameter: %g\nSharpness Parameter: %g\nLearning Rate: %g' 
                      %(bestFunc,'='*30, bestArray[0],bestArray[1], bestArray[2], bestArray[3], bestArray[4]))
            else:
                print('Activation Function: %s\n%s\nUsing the variables:\nIterations: %g\nEpochs: %g\nAlpha: %g\nLearning Rate: %g' 
                      %(bestFunc,'='*45, bestArray[0],bestArray[1], bestArray[2], bestArray[3]))
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
    out : integer
            The categories with wish to sort our results into.
            In the case of the MNIST data, this is 10, as we sort into
            0,1,2,...,8,9.
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
    bestFound : boolean
            Is False by default
            Ignores all other inputs and applies the best possible parameters
            or values. Additionally flags one_return as True, giving only
            one returned accuracy value.
    plotting : boolean
            Is False by default
            Should be True if and only if learns and lmbds are both array-likes.
            If True, returns a plot showing the accuracy for the different
            lmbds and learns values given.  
    reg : boolean
            Is False by default,
            if true, then returns treats it as a regression case, not classification.
            
            PS! This was a last minute edition, and as such, is not very flexible.
            Apologize in advance! It works with learning rate only! Although it is very
            sensitive and should only be in the range -10^4 to 1.
"""
def TF(inputs=inputs,
       labels=labels,
       Type = 'sgd',
       nlayers = [70,50,30],
       alayers = ['sig', 'sig', 'sig'],
       outa = 'sm',
       pens = ['l2', 'l2', 'l2'],
       o=5,
       out=10,
       epochs = 100,
       batchSize = 100,
       learns=np.logspace(-5,1,7),
       lmbds=np.logspace(-5,1,7),
       one_return=False,
       bestFound=False,
       plotting=False,
       reg=False):
    np.random.seed(2)
    if outa.lower() == 'sm':
        outa = 'softmax'
    elif outa.lower() == 'sig':
        outa = 'sigmoid'
    
    if bestFound==True:
        Type = 'sgd'
        nlayers = [80,50,30,20]
        alayers = ['sig','sig','sig','sig']
        outa = 'softmax'
        pens = ['l2','l2','l2','l2']
        out = 10
        epochs = 100
        batchSize = 100
        learns = 1
        lmbds = 1e-4
        one_return=True
    if reg == True:
        N = 1000
        x = np.sort(np.random.uniform(0,1,N))
        y = np.sort(np.random.uniform(0,1,N))
        labels = FrankeFunction(x,y)        
        inputs = X_Mat(x,y,o)
    model = TensorFlow(Type, nlayers, alayers, outa, pens, out, reg)
    scores = model.fitter(inputs, labels, epochs, batchSize, learns, lmbds, one_return)
    if one_return==False and plotting==True and reg==False:
        plt.figure()
        for i in range(scores.shape[0]):
            plt.loglog(lmbds, scores[i][:], label=r'$\eta = {}$'.format(learns[i]))
        plt.legend()
        plt.title(r"""Accuracy for all different $\eta$ with varying $\lambda$
using the {} method""".format(model.Func))
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Accuracy [%]')
        plt.show()
    elif one_return==False and plotting==True and reg==True:
        plt.figure()
        plt.semilogx(learns, scores)
        plt.title('The R2 Score of a FFNN Regression case with a variable Learning Rate')
        plt.xlabel('Learning rate')
        plt.ylabel('R2 Score')
        plt.show()

"""
This was supposed to be the FFNN Regression part, however I could never get
it to work properly, and it would often return NaN and/or just not work at all.
Will look at it sometime prior to project 3, hopefully.
Left here for posterities sake, but not actually used.

def FFNNRegression(o, learn, iters, batchSize):       
    N = 1000
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))
    z = FrankeFunction(x,y)        
    X = X_Mat(x,y,o)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    model = FFNN_old(X_train,z_train,iters ,learn, batchSize)
    
    model.SGD(model.model, X_train, z_train)
    error = []
    for i, x in enumerate(X_test):
        _, prob = model.Forward(x, model.model)
        error = MSE(z,np.dot(X,prob))
    
    print(error)
"""