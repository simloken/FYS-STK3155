from functions import generateSpambaseData, string2cats
from handlers import TF, LR, NB, DT, RF, SV, testAgainstStrings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def predictString(runs, spam=None, start=None, stop=None):
    
    inputs, labels = generateSpambaseData(shuffle=True)
    
    file = open('../data/strings.txt', 'r')
    f = file.readlines()
    file.close()
    
    strings = np.zeros((len(f), 57))
    j = 0
    for i in f:
        strings[j] = string2cats(i)
        j += 1
    
    if spam == True:
        start = 0; stop = 5
    elif spam == False:
        start = 5; stop = 10
    control = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    if spam != None:
        strings = strings[start:stop]
        control = control[start:stop]
    if len(strings) != len(control):
        raise ValueError('Input Strings and control labels must be same length')
    
    if start != None and stop != None:
        strings = strings[start:stop]
        control = control[start:stop]
    elif start != None and stop == None:
        strings = strings[start:10]
        control = control[start:10]
    elif start == None and stop != None:
        strings = strings[0:stop]
        control = control[0:stop]
    
    store = testAgainstStrings(strings, inputs, labels, runs).astype(float)
    TFmodel = TF(inputs, labels, learns = 0.001, lmbds = 0.001, one_return=True, retModel=True)
    TFout = TFmodel.predict(strings)
    TFout = np.round(TFout, decimals=1)
    
    #what follows is an awful way of extracting predictions from TF, apologize in advance
    predictions = np.argmax(TFout, axis=1) #get prediction column
    TFpreds = np.zeros(len(TFout))
    for i in range(len(TFout)):
        TFpreds[i] = TFout[i,predictions[i]] #extract highest prediction
    
    finalarr = np.zeros(len(TFout))
    j = 0
    for i in predictions:
        if i == 0:
            finalarr[j] = abs(TFpreds[j] - 1) #convert to the usual
        elif i == 1:
            finalarr[j] = TFpreds[j]
    
        j += 1
    #horror ends here
    
    store = np.vstack([finalarr, store])
    L = np.shape(store)[1]
    printstr = L*"{:.1f} "
    print('\nCTRL:', printstr.format(*control),'\n')
    
    names = ["TF", "LR", "NB", "DT", "RF", "SV"]
    for i in range(len(names)):
        
        print("{}:  ".format(names[i]), printstr.format(*(store[i][:])))


    store = np.vstack([control, store])
    namesAll = ["CTRL", "TF", "LR", "NB", "DT", "RF", "SV"]
    d2h = pd.DataFrame(store, index=namesAll)
    
    fig = sns.heatmap(data=d2h, annot=True, cmap='autumn')
    plt.yticks(rotation=0)
    plt.xticks([])
    bottom, top = fig.get_ylim()
    fig.set_ylim(bottom + 0.5, top - 0.5)
    fig.set_title('Alternative heatmap')
    plt.show()


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
            if k == 0 and adjust==True:
                yesno = str(input('Would you like to re-adjust values? [y/n]\n'))
                if yesno.lower() == 'y':
                    continue
                elif yesno.lower() == 'n':
                    k = 1
            if adjust == False:
                k = 1