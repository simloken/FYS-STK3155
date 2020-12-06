import numpy as np
from sklearn.model_selection import train_test_split
"""
    The manual non-sklearn version of Logistic Regression.
    Deprecated because I got very poor results, and with a bunch of exams
    coming up I didn't have time to do too much troubleshooting for this project.
    If its important, documentation for this class can be found in the functions.py
    script from Project 2.
"""
class LogisticRegression:  
    def __init__(self, X, z, Type, activation, iters, epochs, penalty, alpha, k, batchSize):
        self.Type = Type.lower()
        self.activation = activation.lower()
        
        self.iters = iters
        self.epochs = epochs
        
        self.penalty = penalty
        
        self.alpha = alpha
        self.k = k
        self.p = 0
        self.batchSize = batchSize
        self.batch = False
        
        if self.Type.lower() in ['gd', 'sgd', 'sgdmb']:
            if Type == 'gd':
                self.method = LogisticRegression.GD
            elif Type == 'sgd':
                self.method = LogisticRegression.SGD
            elif Type == 'sgdmb':
                self.method = LogisticRegression.SGD
                self.batch = True
        else:
            print('Desired method not recognized or supported, defaulting to Stochastic Gradient Descent')
            self.method = LogisticRegression.SGD
        
        if self.penalty.lower() in ['l1', 'l2']:
            if penalty.lower() == 'l1':
                self.pFunc = LogisticRegression.L1
            elif penalty.lower() == 'l2':
                self.pFunc = LogisticRegression.L2
        else:
            print('Desired penalty not recognized or supported, defaulting to L2')
            self.pFunc = LogisticRegression.L2
        
        if self.activation.lower() in ['sm', 'relu', 'lrelu', 'sig', 'sp', 'elu', 'tanh']:
            if activation.lower() == 'sm':
                self.aFunc = LogisticRegression.Softmax
            elif activation.lower() == 'relu':
                self.aFunc = LogisticRegression.RELU
            elif activation.lower() == 'lrelu':
                self.aFunc = LogisticRegression.leaky
            elif activation.lower() == 'sig':
                self.aFunc = LogisticRegression.sigmoid
            elif activation.lower() == 'sp':
                self.aFunc = LogisticRegression.Softplus
            elif activation.lower() == 'elu':
                self.aFunc = LogisticRegression.ELU
            elif activation.lower() == 'tanh':
                self.aFunc = LogisticRegression.tanh
        else:
            print('Desired activation function not recognized or supported, defaulting to Softmax')
            self.aFunc = LogisticRegression.Softmax
            
            """
    The Gradient Descent method
    Takes data X and z, in addition to an empty B array.
    Takes also an initial learning rate and iters, epochs
    Returns the Beta array that returned the smallest loss from our loss
    function
    """
    def GD(self, X, z, B, learn, iters, epochs):
        store_cost = 1e6
        learnparam = [1,20]
        best_B = 0
        for epoch in range(epochs):
            learn *= LogisticRegression.dynamicLearn(epoch, learnparam)
            for i in range(iters):
                old_B = B
                
                B = self.stepper(X, z, B, learn)
                
                if self.p != 0:
                    B += self.p*old_B
            
            cost = self.costfunc(X,z,B)
            if cost < store_cost:
                best_B = B
                store_cost = cost
                    
            
            
        if hasattr(best_B, '__len__') == True:
            return best_B
        else:
            return B
    
    """
    The Stochastic Gradient Descent method
    Takes data X and z, in addition to an empty B array.
    Takes also an initial learning rate and iters, epochs
    Additionally has a boolean batch, batch = True if and only
    if we're using minibatches, aka. Type = 'sgdmb'
    Returns the best B array with the least loss.
    """
    def SGD(self, X, z, B, learn, iters, epochs, batch=False):
        store_cost = 1e6 #some random high number which will get replaced first epoch
        learnparam = [1,20] #learning params to customize the learning rate decay
        best_B = 0
        for epoch in range(epochs):
            learn = LogisticRegression.dynamicLearn(epoch, learnparam)
            B = np.zeros(X.shape[1])
            B[0] = 1 #initializing one element 
            np.random.shuffle(X) #can be shuffled for each iteration as well if you
            np.random.shuffle(z) #want truly random arrays, however this causes excessive
            for i in range(iters):#runtime bloat
                ridx = np.random.randint(len(X)-self.batchSize)
                
                        
                if self.batch==True:
                    Xi = X[ridx:ridx+self.batchSize]
                    zi = z[ridx:ridx+self.batchSize]
                    B = self.stepper(Xi,zi, B, learn)
                else:
                    B = self.stepper(X[ridx], z[ridx], B, learn)   
            cost = self.costfunc(X,z,B)
            
            if cost < store_cost:
                best_B = B
                store_cost = cost
           
        if hasattr(best_B, '__len__') == True:
            return best_B
        else:
            return B
            
    """
    Our stepping function. Find the gradient, then applies that onto our
    B values.
    """
    def stepper(self, X, z, B, learn):
        gradient = self.newGrad(X, z, B)
        B -= gradient*learn
        return B
            
    """
    Our loss function for classification, Cross Entropy.
    """
    def costfunc(self, X, z, B):
        p = np.dot(X, B)
        loss = - np.sum(z*p - np.log(1+np.exp(p)))
        loss += (0.5*self.alpha*np.dot(B,B))
        return loss
    
    """
    This finds our new gradient for B and some shuffled data X and z.
    """    
    def newGrad(self, X, z, B):
        gradient = np.dot(X.T, (self.aFunc(self, (X @ B))) - z)
        gradient += self.alpha*B
        
        return gradient
    """
    This is just a function to call after declaring an object
    Essentially this is what you'll call if you want to solve 
    a classification problem.
    """    
    def fitter(self, X, z, learn):
        
        
        self.B = np.zeros(X.shape[1])
        self.B[0] = 1
        self.B = self.method(self,X, z, self.B, learn, self.iters, self.epochs)
    """
    Function to be called when calculating new learning rates.
    Gives a new fraction which you can multiply with the current learning rate
    """    
    def dynamicLearn(x, param):
        return param[0]/(x+param[1])
    
    """
    The softmax activation function
    """ 
    def Softmax(self, x):
        return np.exp(-x)/np.sum(np.exp(-x))
    
    """
    The RELU activation function
    """    
    def RELU(self, x):
        if self.Type == 'sgd':
            if x < 0:
                x = 0
        else:
            x[x < 0] = 0
        return x
    
    """
    The leaky RELU activation function
    """    
    def leaky(self, x):
        if self.Type == 'sgd':
            if x > 0:
                return x
            elif x <= 0:
                return 0.01*x
        else:
            hold = []
            for i in x:
                if i > 0:
                    hold.append(i)
            
                elif i >= 0:
                    hold.append(0.01*i)       
        return hold
    """
    The softplus activation function
    """    
    def Softplus(self, x):
        return(np.log(1+np.exp(self.k*x))/self.k)
        
    """
    The ELU activation function
    """    
    def ELU(self,x):
        if self.Type == 'sgd':
            if x < 0:
                return self.alpha*(np.exp(x)-1)
            elif x >= 0:
                return x
        else:
            hold = []
            for i in x:
                if i < 0:
                    hold.append(self.alpha*(np.exp(i)-1))
            
                elif i >= 0:
                    hold.append(i)
        return hold
    
    """
    The sigmoid activation function
    """    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    """
    The Hyperbolic Tangent activation function
    
    """
    def tanh(self, x):
        return np.tanh(x)
    """
    The L1 penalty function
    This is never called anywhere as I couln't get it to work properly
    but it's left here because why not.
    """        
    def L1(x):
        return np.linalg.norm(x)
    """
    The L2 penalty function
    Same as above. Never called, but still here.
    """    
    def L2(x):
        return 0.5*np.dot(x,x)
"""
    The old handler for the manual Logistic Regression.
    Also deprecated.
"""    
def LR(
    inputs,
    labels,
    Type,
    aFunc = 'elu',
    penalty = 'l2',
    iters = 600,
    epochs = 200,
    alpha = 0.9,
    kappa = 1,
    batchSize = 50,
    learn = 0.001,
    plotting = False,
    bestFound = False):
    
    model = LogisticRegression(inputs, labels, Type, aFunc, iters,
                               epochs, penalty, alpha, kappa, batchSize)
    
    X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2)
    
    model.fitter(X_train, z_train, learn)
    Bs = model.B
    probas = model.aFunc(model, (X_test @ Bs))
    result = np.zeros(len(probas))
    j = 0
    for i in probas:
        if i >= 0.5:
            result[j] = 1
        elif i < 0.5:
            result[j] = 0
        j += 1
    acc = np.sum(result)/len(result)
    
    return acc, probas