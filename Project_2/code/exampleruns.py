from main import TF, LogisticRegression, MNISTsolver, compareLight, soloSGD
import numpy as np


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