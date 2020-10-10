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
N = 40
nboots = 100
o = 5
k = 5
amp = 1
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
MSE1 = crossval(x,y,z,k,o,'ols')
MSE2, lambdas = crossval(x,y,z,k,o,'ridge')
MSE2 = np.mean(MSE2, axis=1)
MSE3 = crossval(x,y,z,k,o,'lasso')
plt.plot(range(o),MSE1, label='MSE for OLS')
plt.plot(range(o),MSE3, label='MSE for LASSO')
plt.title('MSE for OLS and LASSO using Cross Validation')
plt.xlabel('Polynomial Order')
plt.ylabel('MSE')
plt.legend()
plt.show()
plt.figure()
plt.plot(np.log10(lambdas),(MSE2), label='MSE for Ridge manually')
#plt.plot(np.log10(lambdas),(MSE22), label='MSE for Ridge using SciKitLearn')
plt.legend()
plt.xlabel('Log10(lambdas)')
plt.ylabel('MSE')
plt.title('MSE for Ridge using Cross Validation')
plt.show()