from main import TF, LogisticRegression, MNISTsolver, compareLight, soloSGD
import numpy as np
from sklearn import datasets
from functions import X_Mat, FrankeFunction
#usually the function don't need an input, but just in case.
digits = datasets.load_digits()
inputs = digits.images
labels = digits.target
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
N = 1000
x = np.sort(np.random.uniform(0,1,N))
y = np.sort(np.random.uniform(0,1,N))
z = FrankeFunction(x,y)
X = X_Mat(x,y,5)

#An example of a decent result of our custom SGD in relation to SciKit methods
#compareLight(0.6, 300, 600, 0.9, 100)

#An example of our custom SGD alone with a variable learning rate (and polynomial order)
#soloSGD(np.logspace(-5,1,7),200,200,50,o=10)
#An example of our custom SGD alone with a variable epoch number (and polynomial order)
#soloSGD(0.6, [20,50,100,200],600 ,50 , o=10)
#An example of our custom SGD alone with a variable iteration count (and polynomial order)
#soloSGD(0.6,200,[100,200,300,500],50,o=10)
#An example of our custom SGD alone with a variable batch count (and polynomial order)
#soloSGD(0.6,200,600,[10,30,50,70,100],o=10)
#Solo run using the "best" variables from functions above
#soloSGD(10,100,100,70,o=10)

#FFNN case of Regression for a variable learning rate from 10^-4 to 1.
#TF(inputs=X, labels=z, learns=np.logspace(-4,0,5), reg=True, plotting=True)

#FFNN case of Classification for one layer, 100 neurons (with plotting)
#MNISTsolver(100, plotting=True)

#Tensorflow case of Classification for one layer, 100 neurons (with plotting)
#TF(nlayers=[100],alayers=['sig'],pens=['l2'], plotting=True)