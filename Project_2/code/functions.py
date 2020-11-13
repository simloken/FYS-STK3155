import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.model_selection import train_test_split
"""
    The Feed Forward Neural Network class.
    This handles all the classification problems, namely the MNIST data.

    X_Data : array-like
            The design matrix of our problem
    Y_data : array-like
            The accompanying data we wish to fit onto our design matrix
    activation: string {'sm', 'relu', 'lrelu', 'sig', 'elu', 'sp'}
            A string dictating which activation function to use.
            Compatible with Softmax, RELU, Leaky RELU, Sigmoid, ELU and Softplus.
    n_h_neurons: int
            Integer telling the network how many hidden neurons to have on each layer
    cats: int
            Integer telling the network how many different categories it'll have
            to sort the data into, in our case this is 0, 1, 2,...8,9.
    epochs: int
            Integer telling us how many times to "loop over the loop".
            Ideally you would then only return the values for the Epoch with the
            least loss/cost.
    batch_size: int
            Dictates the size of our batches of data.
            Related to the SDG w/ mini batches method
    eta: float
            The learning rate of our network. Think of this as a timestep.
    lmbd: float
            A parameter
    alpha: float
            A parameter
            

"""
class FFNN:
    def __init__(
            self,
            X_data,
            Y_data,
            activation,
            n_h_neurons=50,
            cats=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            alpha=0.9):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_h_neurons = n_h_neurons
        self.cats = cats

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.alpha = alpha
        
        self.activation = activation
        if self.activation in ['sm', 'relu', 'sig', 'sp', 'tanh']:
            if activation.lower() == 'sm':
                self.aFunc = FFNN.Softmax
            elif activation.lower() == 'relu':
                self.aFunc = FFNN.RELU
            elif activation.lower() == 'sp':
                self.aFunc = FFNN.Softplus
            elif activation.lower() == 'sig':
                self.aFunc = FFNN.sigmoid
            elif activation.lower() == 'tanh':
                self.aFunc = FFNN.tanh
        else:
            print('Desired activation function not recognized or supported, defaulting to Softmax')
            self.aFunc = FFNN.sigmoid
            
            
        self.create_biases_and_weights()
    """
    Creates biases as a function of our supplied n_values
    """
    def create_biases_and_weights(self):
        self.h_weights = np.random.randn(self.n_features, self.n_h_neurons)
        self.h_bias = np.zeros(self.n_h_neurons) + 0.01

        self.o_weights = np.random.randn(self.n_h_neurons, self.cats)
        self.o_bias = np.zeros(self.cats) + 0.01
    """
    Creates probabilities based on supplied data
    """
    def forward(self):
        self.z_h = np.matmul(self.X_data, self.h_weights) + self.h_bias
        self.a_h = self.aFunc(self, self.z_h)

        self.z_o = np.matmul(self.a_h, self.o_weights) + self.o_bias

        exp_term = np.exp(self.z_o)
        self.prob = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    """
    Creates probabilities based on supplied data
    """
    def forward_out(self, X):
        z_h = np.matmul(X, self.h_weights) + self.h_bias
        a_h = self.aFunc(self, z_h)

        z_o = np.matmul(a_h, self.o_weights) + self.o_bias
        
        exp_term = np.exp(z_o)
        prob = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return prob
    """
    Adjusts bias and weights accordingly to supplied data
    """
    def backprop(self):
        error_o = self.prob - self.Y_data
        error_h = np.matmul(error_o, self.o_weights.T) * self.a_h * (1 - self.a_h)

        self.o_weights_grad = np.matmul(self.a_h.T, error_o)
        self.o_bias_grad = np.sum(error_o, axis=0)

        self.h_weights_grad = np.matmul(self.X_data.T, error_h)
        self.h_bias_grad = np.sum(error_h, axis=0)

        if self.lmbd > 0.0:
            self.o_weights_grad += self.lmbd * self.o_weights
            self.h_weights_grad += self.lmbd * self.h_weights

        self.o_weights -= self.eta * self.o_weights_grad
        self.o_bias -= self.eta * self.o_bias_grad
        self.h_weights -= self.eta * self.h_weights_grad
        self.h_bias -= self.eta * self.h_bias_grad
    """
    Predicts probabilities based on supplied data 
    """
    def predict(self, X):
        prob = self.forward_out(X)
        return np.argmax(prob, axis=1)
    """
    Predicts probabilities based on supplied data 
    """
    def predict_prob(self, X):
        prob = self.forward_out(X)
        return prob
    """
    Trains the model over a series of iterations and epochs.
    Used the SDG with minibatches method. Then applies those new datapoints
    onto the forward and backpropagation functions.
    """
    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.forward()
                self.backprop()
    """
    The softmax activation function
    """
    def Softmax(self, x):
        return np.exp(-x)/np.sum(np.exp(-x))
    """
    The sigmoid activation function
    """
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    """
    The RELU activation function
    """
    def RELU(self, x):
        x[x < 0] = 0
        return x
    """
    The softplus activation function
    """
    def Softplus(self, x):
        return(np.log(1+np.exp(x)))
        
    """
    The Hyperbolic Tangent activation function
    
    """
    def tanh(self, x):
        return np.tanh(x)
"""
    The TensorFlow Class.
    This uses tf.keras.
    Type : string
            The solver to use
            Must be {'sgd', 'adagrad', 'adam', 'nadam'}
    neuronsLayer : array-like or int
            The amount of neurons on each layer our neural network.
    activations : array-like or string
            The activation function on each layer of our neural network
            Must be {'sig', 'sm', 'sp', 'relu', 'elu'} or an array-like with
            any these elements
    outAct : string
            The final activation function on our output layer.
            Must be {'sig', 'sm', 'sp', 'relu', 'elu'}
    penalties : array-like or string
            The penalty to be applied on each layer of our neural network
            Must be {'l1', 'l2', 'l1l2'} or an array-like with any of these
            elements
    out : int
            Number of categories for the neural network to sort into.
            Think of it sort of like the "length" of the array we wish to sort
            our data into.
            In our case, when using the MNIST data, we'd be looking to sort
            into 10 categories, 0,1,2,...,8,9.
    
    This class is fully modular, that is, it is compatible with any number of
    layers, but be forewarned that (naturally) neuronsLayer, activations
    and penalties must be the same length.
"""    
class TensorFlow:
    def __init__(
            self,
            Type,
            
            neuronsLayer,    
            activations,
            outAct,
            penalties,
            
            out):
        self.neuronsLayer = neuronsLayer
        self.Type = Type
        self.activations = activations
        self.outAct = outAct
        self.penalties = penalties
        self.out = out
        self.multi = False
        if Type.lower() == 'sgd':
            self.solve = optimizers.SGD
            self.Func = 'SGD with Momentum'
        elif Type.lower() == 'adagrad':
            self.solve = optimizers.Adagrad
            self.Func = 'Adagrad'
        elif Type.lower() == 'adam':
            self.solve = optimizers.Adam
            self.Func = 'Adam'
        elif Type.lower() == 'nadam':
            self.solve = optimizers.Nadam
            self.Func = 'Nadam'
        else:
            raise ValueError("""Solving function not recognized, please use one of the following:
                        ['sgd', 'adagrad', 'adam', 'nadam']""")
        
        if isinstance(self.activations, str) == True and isinstance(self.penalties, str) == True and isinstance(self.neuronsLayer, int) == True:
            if self.activations.lower() == 'sig':
                self.aFuncs = 'sigmoid'
            elif self.activations.lower() ==  'sm':
                self.aFuncs = 'softmax'
            if self.penalties.lower() == 'l2':
                self.pFuncs = regularizers.l2
            elif self.penalties.lower() ==  'l1':
                self.pFuncs = regularizers.l1
            pass
        elif (len(self.neuronsLayer) != len(self.activations) or 
              len(self.neuronsLayer) != len(self.penalties) or
              len(self.activations) != len(self.penalties)):
            raise ValueError('The array-likes of neurons per layer, activation per layer, and \npenalty per layer must be same length')
        elif len(self.neuronsLayer) == len(self.activations) == len(self.penalties):
            self.multi = True
            self.aFuncs = []; self.pFuncs = []
            for i in self.activations:
                if i.lower() == 'sig':
                    self.aFuncs.append('sigmoid')
                elif i.lower() ==  'sm':
                    self.aFuncs.append('softmax')
                elif i.lower() == 'sp':
                    self.aFuncs.append('softplus')
                elif i.lower() == 'relu':
                    self.aFuncs.append('relu')
                elif i.lower() == 'elu':
                    self.aFuncs.append('elu')
                else:
                    raise ValueError("""Activation function not recognized, please use one of the following:
                        ['sig', 'sm', 'sp', 'relu', 'elu']""")
            for i in self.penalties:
                if i.lower() == 'l2':
                    self.pFuncs.append(regularizers.l2)
                elif i.lower() == 'l1':
                    self.pFuncs.append(regularizers.l1)
                elif i.lower() == 'l1l2':
                    self.pFuncs.append(regularizers.L1L2)
                else:
                    raise ValueError("""Penalty function not recognized, please use one of the following:
                        ['l1', 'l2', 'l1l2']""")
    """
    This function handles all the generating the generating of the network.
    It essentially takes only the learning rate and lambda parameter for a given
    "run", and uses that.
    It then returns a model which we can then fit onto our training data.
    """                
    def form_neural_network(self, learn, lmbd):
        model = Sequential()
        for i in range(len(self.neuronsLayer)):
            model.add(Dense(self.neuronsLayer[i], activation=self.aFuncs[i], kernel_regularizer=self.pFuncs[i](lmbd)))
        
        model.add(Dense(self.out, activation=self.outAct))
        
        opter = self.solve(lr=learn)
        
        model.compile(loss='categorical_crossentropy', optimizer=opter, metrics=['accuracy', 'mse'])
        
        return model
        
    """
    This function handles fitting our data.
    Takes an X and z input, in addition to other important information like
    epoch and batchsize
    Is compatible with learns and lmbds being both array-likes and floats,
    but if they are floats then the one_return boolean must be True.
    If learns and lmbds are array-likes, then it additionally returns all the
    accuracy score as an array. Used for plotting
    """
    def fitter(self, X, z, epochs, batchSize, learns, lmbds, one_return):
        z = to_categorical(z)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        if one_return == False:
            if hasattr(learns, '__len__') == False or hasattr(lmbds, '__len__') == False:
                raise ValueError('learns and lmbds must both be array-like')
            storage = np.zeros((len(learns), len(lmbds)), dtype=object)
            storescore = np.zeros((storage.shape))
            for i, learn in enumerate(learns):
                for j, lmbd in enumerate(lmbds):
                    
                    Network = self.form_neural_network(learn, lmbd)
                    Network.fit(X_train, z_train, epochs=epochs, batch_size=batchSize, verbose=0)
                    scores = Network.evaluate(X_test, z_test)
                    
                    storage[i][j] = Network
                    
                    print("Learning rate = ", learn)
                    print("Lambda = ", lmbd)
                    print("Test accuracy: %.3f" % scores[1])
                    print()
                    storescore[i][j] = scores[1]
            return storescore
        else:
            if hasattr(learns, '__len__') == True or hasattr(lmbds, '__len__') == True:
                raise ValueError('learns or lmbds cannot be array-likes')
            Network = self.form_neural_network(learns, lmbds)
            Network.fit(X_train, z_train, epochs=epochs, batch_size=batchSize, verbose=0)
            scores = Network.evaluate(X_test, z_test)
            storage = Network
            
            print("Learning rate = ", learns)
            print("Lambda = ", lmbds)
            print("Test accuracy: %.3f" % scores[1])
            print()
            
"""
    The Logistic Regression class
    This handles all regression problems, mainly when looking at our design
    matrix from Project 1 and the Franke data.
    
    X : array-like
            The design matrix we wish to fit onto our data
    z : array-like
            The data we wish to fit our design matrix onto
    Type : string {'gd', 'sgd', 'sgdmb'}
            The type or rather, the method used to solve a given problem.
            Compatible with three different solutions/solvers:
                Gradient Descent
                Stochastic Gradient Descent
                Stochastic Gradient Descent with Mini Batches
    activation : string {'sm', 'relu', 'lrelu', 'sig', 'sp', 'elu', 'tanh'}
            The activation function we wish to use.
            Compatible with the following:
                Softmax
                RELU
                Leaky RELU
                Sigmoid
                Softplus
                ELU
    iters : int
            The amount of iterations to run in each epoch
    epochs : int
            The amount of loops to run over all iterations.
            Ideally, we compare our observed values for each epoch, and
            extract only the best one
    penalty : string {'l1', 'l2'}
            The type of penalty to use
            Doesn't actually do anything because it kept causing overflow
            issues and/or ruining data. The functions will still be assigned
            but they are never called.
    alpha : float
            A parameter
      k   : float
            A sharpness parameter used only when using the Softplus activation
            function. k = 1 returns a non-sharp version.
    batchSize : int
            The data chunk which we wish to run our data over for a given
            iteration
"""    
class LR:  
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
                self.method = LR.GD
            elif Type == 'sgd':
                self.method = LR.SGD
            elif Type == 'sgdmb':
                self.method = LR.SGD
                self.batch = True
        else:
            print('Desired method not recognized or supported, defaulting to Stochastic Gradient Descent')
            self.method = LR.SGD
        
        if self.penalty.lower() in ['l1', 'l2']:
            if penalty.lower() == 'l1':
                self.pFunc = LR.L1
            elif penalty.lower() == 'l2':
                self.pFunc = LR.L2
        else:
            print('Desired penalty not recognized or supported, defaulting to L2')
            self.pFunc = LR.L2
        
        if self.activation.lower() in ['sm', 'relu', 'lrelu', 'sig', 'sp', 'elu', 'tanh']:
            if activation.lower() == 'sm':
                self.aFunc = LR.Softmax
            elif activation.lower() == 'relu':
                self.aFunc = LR.RELU
            elif activation.lower() == 'lrelu':
                self.aFunc = LR.leaky
            elif activation.lower() == 'sig':
                self.aFunc = LR.sigmoid
            elif activation.lower() == 'sp':
                self.aFunc = LR.Softplus
            elif activation.lower() == 'elu':
                self.aFunc = LR.ELU
            elif activation.lower() == 'tanh':
                self.aFunc = LR.tanh
        else:
            print('Desired activation function not recognized or supported, defaulting to Softmax')
            self.aFunc = LR.Softmax
            
            """
    The Gradient Descent method
    Takes data X and z, in addition to an empty B array.
    Takes also an initial learning rate and iters, epochs
    Returns the Beta array that returned the smallest loss from our loss
    function
    """
    def GD(self, X, z, B, learn, iters, epochs):
        store_cost = 1e6
        learnparam = [1,10]
        best_B = 0
        for epoch in range(epochs):
            learn *= LR.dynamicLearn(epoch, learnparam)
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
        learnparam = [1,10] #learning params to customize the learning rate decay
        for epoch in range(epochs):
            learn *= LR.dynamicLearn(epoch, learnparam)
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
    Our loss function for regression, MSE.
    """
    def costfunc(self, X, z, B):
        P = np.dot(X,B)
        return MSE(z,P)
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
    a regression problem.
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
    Takes two arrays x and y and returns a (len(x) x len(y))
    array.
"""    
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4)
"""
    Takes two array x and y and a polynomial order n.
    Returns a design matrix
"""
def X_Mat(x,y,n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    lenh = int((n+1)*(n+2)/2)                                                              
    X = np.ones([N,lenh])

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
"""
    Scales our data.
"""
def scaler(x,y):
    scaler = StandardScaler()
    scaler.fit(x)
    xS = scaler.transform(x)
    xSS = scaler.transform(y)
    return xS, xSS
"""
    Ordinary Least Squares using Scikit-learn.
"""
def OLS(X,y):
    reg = LinearRegression(fit_intercept = False)
    reg.fit(X,np.ravel(y))
    return reg.coef_

def RIDGE(X,y):
    reg = Ridge(fit_intercept=False)
    reg.fit(X,np.ravel(y))
    return reg.coef_

"""
    The "simple" version of SGD for use in a.
"""
def SGD(X, z, learn, epochs, itrs, alpha, batch):
    
    best_B = 0
    store_cost = 1e6
    for e in range(epochs):
        B = np.zeros(X.shape[1])
        B[0] = 1 #initializing one element 
        learn *= 1/(10+e)
        np.random.shuffle(X)
        np.random.shuffle(z)
        for i in range(itrs):
            ridx = np.random.randint(len(X)-batch)
            
            xi = X[ridx:ridx+batch]
            zi = z[ridx:ridx+batch]
            
            aFunc = 1/(1+np.exp(-xi @ B))
            
            grad = np.dot(xi.T, aFunc-zi)
            B -= learn*grad
            

        p = np.dot(X, B)
        cost = MSE(z,p)
        if cost < store_cost:
            best_B = B
            store_cost = cost  
    if hasattr(best_B, '__len__') == True:
        return best_B
    else:
        return B
"""
    The mean squared error between a true array y and a predictory array yt
"""    
def MSE(y,yt):
    return np.sum((y-yt)**2)/np.size(yt)

"""
    The R2 score between a true array y and a predictory array yt
"""  
def R2(y,yt):
    return 1 - np.sum((y-yt) ** 2) / np.sum((y - np.mean(y)) ** 2)

"""
    Returns a onehot vector of a given input.
    Used to make our MNIST data compatible with
    the FFNN class.
"""  
def tcn(integer_vector):
    n_inputs = len(integer_vector)
    cats = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, cats))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector