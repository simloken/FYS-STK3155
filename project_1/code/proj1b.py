# -*- coding: utf-8 -*-
from functions import FrankeFunction, OLS, X_Mat, Franke3D, sciOLS, MSE, scaler, variancebias, crossval
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

np.random.seed(80085)
N = 30
nboots = 100
o = 5
amp = 0.2
x = np.sort(np.random.uniform(0,1,N))
y = np.sort(np.random.uniform(0,1,N))
z = FrankeFunction(x,y)
z += np.random.normal(0,1, x.shape)*amp
X = X_Mat(x,y,o)
#b1 = sciOLS(X,z)
#b2 = OLS(X,z)

#zpred = X_test @ b2
#ztilde = X_train @ b2
"""
print(zpred,zmesh)
fig = plt.figure()
heat = plt.imshow(zmesh)
fig.colorbar(heat, shrink=0.5, aspect=5)
plt.show()
"""
vari, bias, error, Type = variancebias('lasso',z,N,nboots=nboots,order=o)

plt.plot(range(o),vari, label='Variance')
plt.plot(range(o),bias, label='Bias')
plt.plot(range(o),error, label='Error')
plt.title('Bias-Variance Trade off for the %s method. \n N = %i, nboots = %i, noise amplified by %.1f' %(Type.upper(),N,nboots, amp))
plt.xlabel('Polynomial Order')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.figure()
