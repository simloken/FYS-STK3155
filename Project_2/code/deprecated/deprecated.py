import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import random  
 
class FFNN_old:
    def __init__(self, X, z, iters, learn, batchSize):
        
        
        self.X = X
        self.z = z
        self.iters = iters
        self.learn = learn
        self.batchSize = batchSize
        self.model = dict(
                W1 = np.random.randn(self.X.shape[-1], 100),
                W2 = np.random.randn(100, self.X.shape[-1]))
    
        
        
    def Softmax(x):
        
        return np.exp(x) / np.exp(x).sum()
    
    def Sigmoid(x):
        
        return 1/(1+np.exp(-x))
    
    def Forward(self, x, model):
        
        h = x @ self.model['W1']
        h[h < 0] = 0 
        proba = FFNN_old.Softmax(h @ self.model['W2'])
        return h, proba
    
    def accuracy(zpred,zreal):
        accu = np.zeros(len(zpred))
        for i in range(len(zpred)):
            if zpred[i] == zreal[i]:
                accu[i] = 1
            else:
                accu[i] = 0
        return accu.sum()/len(zpred)
    
    def Backprop(self, model, x, h, error):
        
        dW2 = h.T @ error
        
        dh = error @ self.model['W2'].T
        
        dh[h <= 0] = 0
        
        dW1 = x.T @ dh
        
        return dict(W1=dW1, W2=dW2)
    def SGD(self, model, X_train, z_train):
        random.shuffle(X_train); random.shuffle(z_train)
        for iter in range(self.iters):
            ridx = np.random.randint(len(X_train)-self.batchSize)
            
            for i in range(X_train.shape[0]):
                Xbatch = X_train[ridx:ridx+self.batchSize]
                zbatch = z_train[ridx:ridx+self.batchSize]
                model = self.step(model, Xbatch, zbatch)
                
        return model
    
    def step(self, model, Xbatch, zbatch):
        gradient = self.miniGrad(model, Xbatch, zbatch)
        model = model.copy()
        
        for layer in gradient:
            
            model[layer] += self.learn * gradient[layer]
            
        return model
          
    
    
    def miniGrad(self, model, Xbatch, ybatch):
        xlst = []; hlst = []; errlst = []
        
        for x, cls_idx in zip(Xbatch,ybatch):
            h, zpred = self.Forward(x,model)
            
            yreal = np.zeros(self.X.shape[1])
            yreal[int(cls_idx)] = 1
            
            err = yreal - zpred
            
            xlst.append(x)
            hlst.append(h)
            errlst.append(err)
            
            
        return self.Backprop(model, np.array(xlst), np.array(hlst), np.array(errlst)) 
    
    
"""
nruns = 10

X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2)
# Create placeholder to accumulate prediction accuracy
accs = np.zeros(nruns)

for k in range(nruns):
    # Reset model
    model = FFNN(inputs,labels,100,1e-3,50,'cla')

    # Train the model
    model_results = model.SGD(model.model, X_train, z_train)

    y_pred = np.zeros_like(z_test)

    for i, x in enumerate(X_test):
        # Predict the distribution of label
        _, prob = model.Forward(x, model.model)
        # Get label by picking the most probable one
        y = np.argmax(prob)
        y_pred[i] = y

    # Compare the predictions with the true labels and take the percentage
    accs[k] = (y_pred == z_test).sum() / z_test.size

print('Mean accuracy: %g' %(accs.mean()))
"""