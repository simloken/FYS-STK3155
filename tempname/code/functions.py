# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
"""
Takes two arrays x and y and returns the Franke Function given those two arrays
"""
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return (term1 + term2 + term3 + term4)
"""
Plots a default Franke Function over x,y
"""
def Franke3D(x,y):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    xm, ym = np.meshgrid(x,y)

    z = FrankeFunction(xm, ym)
    # Plot the surface.
    surf = ax.plot_surface(xm,ym,z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    plt.show()
    return x,y,z
"""
Performs an OLS operation given a design matrix X and a dataset y.
Returns beta, in addition to ytilde and the prediciton of y
"""
def OLS(X,y):
    if len(y.shape) > 1:
        y = np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    B = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    scaler(X_train, X_test)
    ytilde = X_train @ B
    ypred = X_test  @ B
    return B, ytilde, ypred
"""
The Scikitlearn method for OLS
"""
def sciOLS(X,y):
    reg = LinearRegression(fit_intercept = False)
    reg.fit(X,np.ravel(y))
    return reg.coef_
"""
Takes the desired method, in addition to the dataset and some length N
Additionally, it also takes amount of bootstraps and the polynomial order.
Returns three arrays containing the variance, the bias and the MSE.
Additionally returns which method was used, for better plotting/easier to keep
track of which values belong to which statistical method.

PS.Feels like something is wrong, the outputs look a bit weird but it could
just be my inexperience with this subject (?). Reader beware (if it's wrong)!
"""
from numba import jit
@jit
def variancebias(Type,z,N,nboots=50, order=5):
    x = np.sort(np.random.uniform(0,1,N))
    y = np.sort(np.random.uniform(0,1,N))

    Type = Type.lower()
    if Type not in ['ols', 'ridge', 'lasso']:
        raise ValueError('Not accepted method. Try OLS, Ridge or Lasso')
        
    if Type == 'ols' or Type == 'lasso' or Type =='ridge':
        variL = np.zeros(order)
        biasL = np.zeros(order)
        error = np.zeros(order)
        for degree in range(order):
            if Type =='ols':
                model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
            elif Type == 'lasso':
                model = make_pipeline(PolynomialFeatures(degree=degree), Lasso(fit_intercept=False))
            elif Type == 'ridge':
                model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(fit_intercept=False))
            X = X_Mat(x,y,degree)
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
            X_train, X_test = scaler(X_train, X_test)
            zpred = np.empty((y_test.shape[0], nboots))
            for i in range(nboots):
                print('ProgressTracker: ',degree,i) #calculations take time, this is a very simple way of tracking how far along you are
                X_, z_ = resample(X_train, y_train)
                zpred[:, i] = model.fit(X_, z_).predict(X_test).ravel() 
            
            y_test = y_test.reshape(len(y_test),1)
            error[degree] = np.mean( np.mean((y_test - zpred)**2, axis=1, keepdims=True) )
            variL[degree] = np.mean( np.var(zpred, axis=1, keepdims=True) )
            biasL[degree] = np.mean( (y_test - np.mean(zpred, axis=1, keepdims=True))**2 )
    return variL, biasL, error, Type

    
        
    
"""
Takes two randomized arrays x and y to span our design matrix X.
Additionally takes a dataset z and a kfold value k. o is polynomial order
Type is the desired method.
Returns an estimated MSE for each order of polynomial as an array.
Should be plotted against 
"""         
def crossval(x,y,z,k,o,Type, RealDat=False):
    kfold = KFold(n_splits=k)

    poly = PolynomialFeatures(degree=o)
    X = X_Mat(x,y,o)
    Type = Type.lower()
    if Type not in ['ols', 'ridge', 'lasso']:
        raise ValueError('Not accepted method. Try OLS, Ridge or Lasso')
        
    if Type == 'ols':
        model = LinearRegression()
        estimated_mse_sklearn = np.zeros(o)
        
    elif Type == 'ridge':
        nlambdas = 500
        scoresKfold = np.zeros((nlambdas, k))
        lambdas = np.logspace(-5, 3, nlambdas)
        estimated_mse_sklearn = np.zeros(nlambdas)
        #scoresSK = np.zeros(nlambdas)
        i = 0
        sneed = []
        for lmb in lambdas:
            j = 0
            model = Ridge(alpha=lmb)
            for train_inds, test_inds in kfold.split(X):
                x_train = x[train_inds]
                y_train = y[train_inds]
                x_test = x[test_inds]
                y_test = y[test_inds]
                X_train =  poly.fit_transform(x_train[:,np.newaxis])
                model.fit(X_train, y_train[:,np.newaxis])
                
                X_test = poly.fit_transform(x_test[:,np.newaxis])
                ypred = model.predict(X_test)
                scoresKfold[i,j] = np.sum((ypred-y_test[:,np.newaxis])**2)/np.size(ypred)
                j += 1
                sneed.append(ypred)
            i += 1
        """
        i = 0
        #scikit solution, return scoresSK, although it is (pretty much) identical to the manual solution 
        for lmb in lambdas:
            ridge = Ridge(alpha = lmb)
        
            X = poly.fit_transform(x[:, np.newaxis])
            estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)
            scoresSK[i] = np.mean(-estimated_mse_folds)

            i += 1
        """
        if RealDat == True:
            return scoresKfold, lambdas, sneed
        else:
            return scoresKfold, lambdas
    elif Type == 'lasso':
        model = LassoCV(cv=k)
        estimated_mse_sklearn = np.zeros(o)
        
    if Type == 'ols' or 'lasso':
        for polydegree in range(1, o):
            for degree in range(polydegree):
                X = X_Mat(x,y,degree)
            estimated_mse_folds = cross_val_score(model, X, z, scoring='neg_mean_squared_error', cv=kfold)
            estimated_mse_sklearn[polydegree] = np.mean(-estimated_mse_folds)  
        
        return estimated_mse_sklearn
"""
Takes a value for the "true" y and y_tilde and returns the bias
"""
def bias(y,yt):
    return np.sum((y - np.mean(yt))**2)/np.size(yt)
"""
Takes a value for the "true" y and y_tilde and returns the variance
"""
def vari(yt):
    return np.sum((yt - np.mean(yt))**2)/np.size(yt)

"""
Creates our design matrix given arrays x,y and polynomial order n
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
Finds mean square error given the "true" y and y_tilde
"""
def MSE(y,yt):
    n = np.size(yt)
    return np.sum((y-yt)**2)/n
"""
Finds R2 score given the "true" y and y_tilde
"""
def R2(y,yt):
    return 1 - np.mean((y-yt) ** 2) / np.mean((y - np.mean(y)) ** 2)
"""
Scales arrays x and y. Use on ex. X_train and X_test
"""
def scaler(x,y):
    scaler = StandardScaler()
    scaler.fit(x)
    xS = scaler.transform(x)
    xSS = scaler.transform(y)
    return xS, xSS


"""
Plots real terrain using ridge method. Does not work for the other regression methods
Used Cross Validation
Does not work!!!
"""
def terrainPlot(z,k,N=100, Type='ridge'):
    if Type not in ['ols', 'ridge', 'lasso']:
        raise ValueError('Not accepted method. Try OLS, Ridge or Lasso')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(0,1,N)
    y = x
    o = 5
    hold, hold2, zpred = crossval(x,y,z,k,o,Type, RealDat=True)
    zpred = np.array(zpred).ravel()
    xm,ym = np.meshgrid(x,y)
    ax.plot_surface(xm,ym,zpred.reshape((N,N)), cmap=cm.coolwarm,linewidth=0, antialiased=False)