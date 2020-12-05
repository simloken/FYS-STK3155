from functions import generateSpambaseData, string2cats
from handlers import TF, LR, NB, DT, RF, SV, testAgainstStrings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt



def predictString(runs):
    
    inputs, labels = generateSpambaseData(shuffle=True)
    
    file = open('../data/strings.txt', 'r')
    f = file.readlines()
    file.close()
    
    strings = np.zeros((len(f), 57))
    j = 0
    for i in f:
        strings[j] = string2cats(i)
        j += 1
    
    control = [1, 1, 1, 1, 0, 0, 0]
    
    store = testAgainstStrings(strings, inputs, labels, runs).astype(float)
    L = np.shape(store)[1]
    printstr = L*"{:.1f} "
    print('\nCTRL:', printstr.format(*control),'\n')
    names = ["LR", "NB", "DT", "RF", "SV"]
    for i in range(len(store)):
        
        print("{}:  ".format(names[i]), printstr.format(*(store[i][:])))


    #this does not work for some reason, 
    #so I cannot compare with the predictions cast 
    #by our Tensorflow neural network
    
    """
    modelList = TF(inputs, labels, retModel=True)
    for i in modelList:
        print(i.evaluate(strings, np.asarray(stringsLabel)))
    """


def accuracies(runs, tensor=False, adjust=False):
    
    inputs, labels = generateSpambaseData(shuffle=True)
    
    lr, _ = LR(inputs,labels, runs, True)
    
    nb, _ = NB(inputs, labels, 1, runs)
    
    dt, _ = DT(inputs,labels, runs, True)
    
    rf, _ = RF(inputs,labels, runs, True)
    
    sv, _ = SV(inputs,labels, runs, False)
    
    names = ["LR", "NB", "DT", "RF", "SV"]
    lst = [lr, nb, dt, rf, sv]
    
    for i in range(len(lst)):
        print('%s yieled a mean accuracy of %g over %i runs' %(names[i], lst[i], runs))
    
    learns, lmbds = 0.0005, 0.0005
    k = 0
    if tensor == True:
        while k == 0:
            if adjust == True:
                learns = float(input('Please enter a learning rate:\n'))
                lmbds = float(input('Please enter a lambda parameter:\n'))
                print('')
            TF(inputs, labels, learns=learns, lmbds=lmbds, one_return=True)
            if k == 0:
                yesno = str(input('Would you like to re-adjust values? [y/n]\n'))
                if yesno.lower() == 'y':
                    continue
                elif yesno.lower() == 'n':
                    k = 1
            if adjust == False:
                k = 1