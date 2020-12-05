import pandas
import numpy as np

def generateSpambaseData(shuffle=False):
    
    dataFile = pandas.read_csv('../data/spam.csv')
    
    holdAll = np.zeros((np.shape(dataFile)))
    for i in range(len(dataFile)):
        holdAll[i][:] = dataFile.iloc[i]
        
        
    if shuffle == True:
        np.random.shuffle(holdAll)
    
    inputs = np.zeros((len(holdAll), np.shape(holdAll)[1]))
    labels = np.zeros((np.shape(holdAll)[0]))
    
    for i in range(np.shape(holdAll)[0]):
        labels[i] = holdAll[i][57]
        
    j = 0
    for i in holdAll:
        inputs[j] = i[:]
        j += 1
        
    if shuffle == True:        
        x = np.random.randint(0, np.shape(holdAll)[0]) #random sample to
        y = np.random.randint(0, np.shape(holdAll)[1]-1)# ensure that shuffling
        assert inputs[x,y] == holdAll[x,y] #has not messed up data
        assert labels[x] == holdAll[x, np.shape(holdAll)[1]-1] #should never fail
        
    inputs = np.transpose(inputs) #remove the labeling
    inputs = np.delete(inputs, len(inputs)-1, 0) #from our 
    inputs = np.transpose(inputs) #input array
    return inputs, labels

def string2cats(inputString, tol=0.75):
    #based on the attribute information given at http://archive.ics.uci.edu/ml/datasets/Spambase/
    #takes a given string and returns it as if it was part of our dataset
    
    words = ["make", "address", "all", "3d", "our", "over", "remove", "internet",
             "order", "mail", "recieve", "will", "people", "report", "addresses",
             "free", "business", "email", "you", "credit", "your", "font", "000",
             "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet",
             "857", "data", "415", "85", "technology", "1999", "parts", "pm",
             "direct", "cs", "meeting", "original", "project", "re", "edu", 
             "table", "conference"]
    
    woc = np.zeros(len(words))
    
    k = 0
    inputs = inputString.split(" ")
    for i in words: #checks for words
        for j in inputs:
            j = ''.join(c for c in j if c.isalpha()) #removes special characters from word
            if i.lower() == j.lower():
                woc[k] += 1
        if woc[k] == woc[k-1]: #check for sneaky modifications
            z = 0 #if a word matches a flagged word within a tolerance, then it will treat it as such
            for ii, jj in zip(i.lower(), j.lower()): #for example, rec1eve to fake out recieve would've
                    if ii == jj: #been caught here.
                        z += 1
            if z/len(j) >= tol:
                woc[k] += 1
        
        if woc[k] == woc[k-1]: #check for weird sneaky word extensions
            ii = "".join(set(i)) #shortens a word to include no duplicates, then checks it against a no duplicates flagged word
            jj = "".join(set(j)) #for example, mooney to fake out money would've been caught here
            if ii.lower() == jj.lower(): #lowers the "resolution" of a word and risks false positives but
                woc[k] += 1 #shouldn't be too common
                
        k += 1
    
    woc = 100*woc/len(inputs)
    
    specialchars = [";", "(", "[", "!", "$", "#"]
                                
    soc = np.zeros(len(specialchars))
     
    k = 0                           
    for i in specialchars: #checks for special characters
        for j in inputString:
            if i == j:
                soc[k] += 1
        k += 1
        
    soc = 100*soc/len(inputString)
    
    uppers = np.zeros(3)
    
    k = 0
    average = []
    
    for i in inputString: #checks for uppercase letters
        if i.isupper() and i.isalpha():
            k += 1
        elif i.islower() and i.isalpha():
            if k != 0:
                average.append(k)
            
            k = 0
    if k != 0:
        average.append(k)
            
    if not average: #if there are no uppercase letters
        uppers[0] = 0
        uppers[1] = 0  
        uppers[2] = 0
    else:
        uppers[0] = np.mean(average)    
        uppers[1] = max(average)    
        uppers[2] = sum(1 for i in inputString if i.isupper())
    
    output = np.hstack((woc, soc, uppers)).ravel()
    return output
    
"""
    The mean squared error between a true array y and a predictory array yt
"""    
def MSE(y,yt):
    return np.sum((y-yt)**2)/np.size(yt)

"""
    The R2 score between a true array y and a predictory array yt
"""  
def R2(y,yt):
    return 1 - np.sum((y-yt) ** 2) / np.sum((y - np.mean(y)) ** 2) 


def logsummer(arr, axis=None):
    #not actually used, from an old iteration of MN Naive Bayes
    arr = np.asarray(arr)
    
    if axis is None:
        arr = arr.ravel()
    else:
        arr = np.rollaxis(arr, axis)
        
    arrmax = arr.max(axis=0)
    out = np.log(sum(np.exp(arr - arrmax), axis=0))
    out += arrmax
    
    return out

def findSVParams(inputs, labels):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    
    param_grid = {'C': [1, 10, 100, 1000],
                'gamma': [1, 0.5, 0.01, 0.001, 0.0001],
                'kernel': ['rbf']}
    X_train, X_test, z_train, z_test = train_test_split(inputs, labels, test_size=0.2)
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
    grid.fit(X_train, z_train)