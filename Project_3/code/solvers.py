from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC as SVC
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
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
            In our case, this is 2 (spam/not spam)
    retModel : boolean
            Tells the solver whether or not to return the model instead of values.
            Useful when we want to cast predictions on other data.
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
            out,
            retModel=False):
        self.neuronsLayer = neuronsLayer
        self.Type = Type
        self.activations = activations
        self.outAct = outAct
        self.penalties = penalties
        self.out = out
        self.multi = False
        self.retModel = retModel
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
        
        model.compile(loss='categorical_crossentropy', optimizer=opter, metrics=['accuracy'])
        
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
        lst = []
        if one_return == False:
            if hasattr(learns, '__len__') == False or hasattr(lmbds, '__len__') == False:
                raise ValueError('learns and lmbds must both be array-like')
            storage = np.zeros((len(learns), len(lmbds)), dtype=object)
            storescore = np.zeros((storage.shape))
            for i, learn in enumerate(learns):
                for j, lmbd in enumerate(lmbds):
                    
                    Network = self.form_neural_network(learn, lmbd)
                    Network.fit(X_train, z_train, epochs=epochs, batch_size=batchSize, verbose=0)
                    if self.retModel == True:
                        lst.append(Network)
                    scores = Network.evaluate(X_test, z_test, verbose=0)
                    
                    storage[i][j] = Network
                    
                    print("Learning rate = ", learn)
                    print("Lambda = ", lmbd)
                    print("Test accuracy: %.3f" % scores[1])
                    print()
                    storescore[i][j] = scores[1]
            if self.retModel == True:
                return lst
            return storescore
        else:
            if hasattr(learns, '__len__') == True or hasattr(lmbds, '__len__') == True:
                raise ValueError('learns or lmbds cannot be array-likes')
            Network = self.form_neural_network(learns, lmbds)
            Network.fit(X_train, z_train, epochs=epochs, batch_size=batchSize, verbose=0)
            if self.retModel == True:
                return Network
            scores = Network.evaluate(X_test, z_test, verbose=0)
            storage = Network
            
            print("Learning rate = ", learns)
            print("Lambda = ", lmbds)
            print("Test accuracy: %.3f" % scores[1])
            print()

"""
    The LogisticRegression class
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
"""
class LogisticRegression:
    
    def __init__(self,
                 inputs,
                 labels,
                 scaled,
                 retModel=False
                ):
        self.inputs = inputs
        self.labels = labels
        self.scaled = scaled
        self.retModel = retModel
        if self.scaled not in [True, False]:
            raise ValueError('scaled must be a boolean')
            
    def fitter(self):
        X_train, X_test, z_train, z_test = train_test_split(self.inputs, self.labels, test_size=0.2)
        model = LR(max_iter=500)
        if self.scaled == True:
            scale = StandardScaler()
            scale.fit(X_train)
            trainScaled = scale.transform(X_train)
            testScaled = scale.transform(X_test)
            model.fit(trainScaled, z_train)
            if self.retModel == True:
                return model
            
            return model.score(testScaled, z_test)
        else:
            if self.retModel == True:
                model.fit(X_train, z_train)
                return model
            model.fit(X_train, z_train)
            return model.score(X_test, z_test)
        
    
"""
    This class is called by the Multinomial class below and contains both the
    prediction casting and the accuracy.
"""    
    
class NaiveBayes:
    #Using sci-kit learns NaiveBayes as reference, however slimmed down to just include Multinomial (which is most common for spam filtering)
    #For reference, https://github.com/scikit-learn/scikit-learn/blob/a1860144aa2083277ba354b0cc46f9eb4acf0db0/sklearn/naive_bayes.py
    
    def predictor(self, X):
        
        return self.cats[np.argmax(self.loglike(X), axis=1)]
    
    def accuracy(self, X, z):
        
        pred = self.predictor(X)
        accuracy_score = 0
        for i in range(len(pred)):
            if pred[i] == z[i]:
                accuracy_score += 1
                
        return accuracy_score/len(pred)
    
"""
    The Multinomial Naive Bayes class
    
    alpha : float
        a parameter. must be greater than (or equal to) 0.
"""    
class Multinomial(NaiveBayes):
    
    def __init__(self,
                 alpha):
        if alpha <= 0:
            raise ValueError('Alpha parameter cannot be less than or equal to 0')
            
        self.alpha = alpha
    
    def fitter(self, X, z):
        
        cats, counts = np.unique(z, return_counts=True)
        self.cats = cats
        self.prevlogs = np.log([counts]) - np.log(len(z))
        
        
        self.lst = []
        for i in range(len(self.cats)):
            choice = X[z == i, :]
            
            count = np.sum(choice, axis=0) + self.alpha
            countsum = np.sum(count) + 2*self.alpha
        
            self.lst.append(np.log(count) - np.log(countsum.reshape(-1,1)))
        
        self.lst = np.vstack(self.lst)
    #log likelihood        
    def loglike(self, X):
        
        return self.prevlogs + np.dot(X, self.lst.T)
"""
    The DecisionTree class
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
"""    
class DecisionTree():
    
    def __init__(self,
             inputs,
             labels,
             scaled,
             retModel=False):
        
        self.inputs = inputs
        self.labels = labels
        self.scaled = scaled
        self.retModel = retModel
        if self.scaled not in [True, False]:
            raise ValueError('scaled must be a boolean')
        
    def fitter(self):
        X_train, X_test, z_train, z_test = train_test_split(self.inputs, self.labels, test_size=0.2)
        model = DecisionTreeClassifier(criterion='entropy', max_features='log2')
        if self.scaled == True:
            scale = StandardScaler()
            scale.fit(X_train)
            trainScaled = scale.transform(X_train)
            testScaled = scale.transform(X_test)
            model.fit(trainScaled, z_train)
            if self.retModel == True:
                return model
            
            return model.score(testScaled, z_test)
        else:
            if self.retModel == True:
                model.fit(X_train, z_train)
                return model
            
            model.fit(X_train, z_train)
            return model.score(X_test, z_test)
"""
    The RandomForest class
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
"""    
class RandomForest():
    
    def __init__(self,
             inputs,
             labels,
             scaled,
             retModel=False):
        
        self.inputs = inputs
        self.labels = labels
        self.scaled = scaled
        self.retModel = retModel
        if self.scaled not in [True, False]:
            raise ValueError('scaled must be a boolean')
    def fitter(self):
        X_train, X_test, z_train, z_test = train_test_split(self.inputs, self.labels, test_size=0.2)
        model = RandomForestClassifier()
        if self.scaled == True:
            scale = StandardScaler()
            scale.fit(X_train)
            trainScaled = scale.transform(X_train)
            testScaled = scale.transform(X_test)
            model.fit(trainScaled, z_train)
            if self.retModel == True:
                return model
            
            return model.score(testScaled, z_test)
        else:
            if self.retModel == True:
                model.fit(X_train, z_train)
                return model
            
            model.fit(X_train, z_train)
            return model.score(X_test, z_test)
        
"""
    The SupportVector class
    
    inputs : array-like
        a set of inputs which can then be categorized to some label
    labels : array-like
        the aforementioned labels
    scaled : boolean
        whether or not we wish to scale our data
    retModel : boolean
        whether or not to return the model instead of just a set of values.
"""        
class SupportVector():
    
    def __init__(self,
             inputs,
             labels,
             scaled,
             retModel=False):
        
        self.inputs = inputs
        self.labels = labels
        self.scaled = scaled
        self.retModel = retModel
        if self.scaled not in [True, False]:
            raise ValueError('scaled must be a boolean')
        
        
    def fitter(self):
        X_train, X_test, z_train, z_test = train_test_split(self.inputs, self.labels, test_size=0.2)
        model = SVC(C=1000, dual=False)
        if self.scaled == True:
            scale = StandardScaler()
            scale.fit(X_train)
            trainScaled = scale.transform(X_train)
            testScaled = scale.transform(X_test)
            model.fit(trainScaled, z_train)
            if self.retModel == True:
                return model
            
            return model.score(testScaled, z_test)
        else:
            model.fit(X_train, z_train)
            if self.retModel == True:
                return model
            
            return model.score(X_test, z_test)
        