# -*- coding: utf-8 -*-
from functions import FrankeFunction, OLS, X_Mat, Franke3D, scaler, MSE, R2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
fig = plt.figure()
ax = fig.gca(projection='3d')
np.random.seed(80085)
N = 500
o = 7
x = np.sort(np.random.uniform(0,1,N))
y = np.sort(np.random.uniform(0,1,N))
xm, ym = np.meshgrid(x,y)
z = FrankeFunction(xm,ym)
#z =+ np.random.normal(size = (N,N))
xr = np.linspace(0,1,N)
yr = np.linspace(0,1,N)
xm, ym = np.meshgrid(x,y)
def y_t(Xr,B):
    return Xr @ B

X = X_Mat(xm,ym,o)
Xr = X_Mat(xm,ym,o)
B, ztilde, zpred = OLS(X,z)
surf = ax.plot_surface(xm, ym, (y_t(Xr,B)).reshape((N,N)),cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
#Franke3D(x,y)

z = np.ravel(z)
X_train,X_test,z_Train,z_Test = train_test_split(X,z,test_size=0.2)
X_train,X_test = scaler(X_train,X_test)
MSE_train = MSE(z_Train,ztilde)
R2_train = R2(z_Train,ztilde)
MSE_test = MSE(z_Test,zpred)
R2_test = R2(z_Test,zpred)
vB = np.diag(np.linalg.inv(X_train.T @ X_train))
conf = 1.96*np.sqrt(vB)
print('Beta:', B)
print('Confidence:', conf)
print('Training MSE:', MSE_train)
print('Training R2:', R2_train)
print('Testing MSE:', MSE_test)
print('Testing R2:', R2_test)