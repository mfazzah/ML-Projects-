import math
import numpy as np
import os 
import matplotlib.pyplot as plt

"""
    returns the length/line wordCount of a file 
"""
def getwordCount(fname):
    wordCount = 0 
    
    with open(os.path.join(os.getcwd(), fname), 'r') as f:
        for line in f:  
            wordCount += 1 
    return wordCount 


"""
Initializing some global variables 
"""
#get lengths of relevant files, will form matrix dimensions later on
wordCountClass = getwordCount("newsgrouplabels.map")
wordCountVocab = getwordCount("vocabulary.txt")
wordCountTest = getwordCount("test.label")
wordCountTrain = getwordCount("train.label")

#holds total numbers of words for each class 
totalOccurence = [0] * wordCountClass

vocab = []
testData = []
testLabel = []
trainData = []
trainLabel = []
labels = []

mapMatrix = np.zeros((wordCountClass, wordCountVocab))
probability = np.zeros((wordCountClass, wordCountVocab))


"""
Function parses and cleans the relevant data of additional whitespace and populates
lists with it 
"""

def parseFiles():
    with open(os.getcwd() + "/vocabulary.txt") as file:
        for i, line in enumerate(file):
            vocab.append(line.strip())
            
    with open(os.getcwd() + "/test.data.txt") as file:
        for line in file:
            line = line.strip()
            newline = line.split(' ')
            testData.append(newline)
            
    with open(os.getcwd() + "/test.label") as file:
        for i, line in enumerate(file):
            testLabel.append(line.strip())


    with open(os.getcwd() + "/train.label") as file:
        for i, line in enumerate(file):
            trainLabel.append(line.strip())
            totalOccurence[int(line)-1]+=1

    with open(os.getcwd() + "/train.data.txt") as file:
        for line in file:
            line = line.strip()
            newline = line.split(' ')
            trainData.append(newline)
    with open(os.getcwd() + "/newsgrouplabels.map") as file:
        for line in file:
            labels.append(line.strip())
            

"""
Function calculates the number of times a word appears in every document in the 
training set (train.data). The value pertaining to each classifier is held in each
element of the mapMatrix. The total amount of words across every document is returned
"""
def configureMAP(alpha):
    totalWordwordCount = [0]*(len(vocab)+1)
    for i, line in enumerate(trainData):
        line.append(str(trainLabel[int(line[0])-1]))
        mapMatrix[int(line[3])-1][int(line[1])-1]+=int(line[2])
    
    totalEachClass = [sum(mapMatrix[i]) for i in range(20)]
    return totalEachClass

"""
The function finds the prior probability of a class k--in this case, the
training set. Given N documents in total in a training set, the prior 
probability of a class k may be as the relative frequence of documents of class k 
        P-hat(Y_k) = N_k / N
"""

def getMLE(totalEachClass):
    
    probabilityEachClass = [float(k / wordCountTrain) for k in totalEachClass ]
    
    return probabilityEachClass

def classify(probabilityEachClass, alpha):
    """
    Given a classifier, calculate the possbility of each word
        P(X_i | Y_k) = (sum of X_i for class k) + alpha+1 / 
                        (total words in class k) * ((alpha+1) * all vocab words)
    """
    for k in range(20):
        for i, j in enumerate(vocab):
            numerator = mapMatrix[k][i] + (alpha+1)
            denominator = totalEachClass[k] + ((alpha+1) * wordCountVocab)
            probability[k,i] = float(numerator) / float(denominator)
    
    """
    Classify test data
        Y_n = argmax(log(P(Y_k)) + sum(X_n) * log(P(X_n | Y_k)))
            where sum(X_n) is the sum of the words in test data for a respective class
            and P(X_n | Y_k) is the proability of each word in X_n given a classifier
    """
    
    temp = []
    checkDoc = int(testData[0][0])
    #used to check label that were matched correctly (or not) against the test data
    #if check[i] == testLabel[i], then it was matched correctly
    check = []
    classArray = []
    
    for el in testData:
        if int(el[0]) == checkDoc:
            temp.append(el[1:])
        else:
            for k in range(20):
                #log(P(Y_k))
                logProbabilityEachClass = math.log(probabilityEachClass[k])
                
                wordCount = 0
                #sum(X_n) * log(P(X_n | Y_k)
                for i in temp:
                    wordCount += (float(i[1])) * (math.log(probability[k, int(i[0])-1]))
                                 
                classArray.append(logProbabilityEachClass + wordCount)
                
            #argmax of each Y_n
            check.append(classArray.index(max(v for v in classArray))+1)
            
            #set variables for next iteration
            checkDoc = int(el[0])
            temp=[]
            classArray = []
   
    check.append('20')
    return check, testLabel
    

if __name__=='__main__':
    parseFiles()
    totalEachClass = configureMAP(1/wordCountVocab)
    probabilityEachClass = getMLE(totalEachClass)
    
    yPred, yTest = classify(probabilityEachClass, 1/wordCountVocab)
    
    print(type(yPred))
    
    
    totalWords = len(yPred)
    
    totalMatches = 0
    for i in range(len(yPred)):
        if int(yPred[i]) == int(yTest[i]):
            totalMatches += 1
    
    #confusion matrix
    yPred = [int(i) for i in yPred]
    yTest= [int(i) for i in yTest]

    #formatting plot 
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(yTest, yPred)
    
 
    sns.heatmap(cm,annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (alpha = 1/|V|)')
    plt.ylabel('Actual')
    
    #Show results 
    print("\n\nOverall testing accuracy (alpha = 1/|V|): {}%".format(totalMatches / totalWords * 100))
    print(cm)
    plt.show()
    
    #test for different values of alpha 
    print('testing for different values of alpha')
    accuracy = []
    alphas = [.00001, .00002, .00004, .00008, .0001, .0002, .0004, .0008, .001, .002, .004, .008,.01,.02, .04, .08, .1,.2,.4,.8,0.9, 0.99]
    for a in alphas:
         print("iteration: ", a)
         totalEachClass = configureMAP(a)
         probabilityEachClass = getMLE(totalEachClass)
         yPred, yTest = classify(probabilityEachClass, a)
         
         totalWords = len(yPred)
    
         totalMatches = 0
         for i in range(len(yPred)):
             if int(yPred[i]) == int(yTest[i]):
                 totalMatches += 1
         acc = totalMatches/totalWords * 100
         print('accuracy: ', acc)
         accuracy.append(acc)
         
        
    #graph of alphas vs percent accuracy  
    plt.subplot(111)
    plt.semilogx(alphas, accuracy)
    plt.title("Alpha vs Percent accuracy")
    plt.xlabel('Alpha')
    plt.ylabel('Percent accuracy')
    plt.grid(True)
    plt.show()
        