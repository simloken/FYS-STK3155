import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
from functions import X_Mat, crossval, terrainPlot
from sklearn.model_selection import train_test_split
# Load the terrain                                                                                  
terrain = imread('SRTM_data_Norway_1.tif')

# just fixing a set of points
N = 1000
o = 5 # polynomial order  
k = 5                                                                          
terrain = terrain[:N,:N]
z = np.ravel(terrain)
# Creates mesh of image pixels                                                                      
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
# Note the use of meshgrid
X = X_Mat(x_mesh, y_mesh,o)
X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)
MSE, ldas = crossval(x,y,z,5,5,'Ridge')  
MSE = np.mean(MSE, axis=1)   
plt.plot(np.log10(ldas),MSE)
plt.xlabel('Log10(lambdas)')
plt.ylabel('MSE')
plt.title('MSE for Ridge using Cross Validation')
plt.show()
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#does not work sadly, I can't think of any way to spread a zpred over a mesh, nor get NxN values for zpred to begin with.
terrainPlot(z,k)