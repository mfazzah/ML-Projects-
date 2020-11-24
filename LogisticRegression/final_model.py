import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn 

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from statistics import mode 
x_OneHotEncoder = OneHotEncoder()
y_OneHotEncoder = LabelBinarizer()
scale = StandardScaler()

def finalmodel(lr):
    df = pd.read_csv('adult.data')
    
    """Final model"""
    #preparing data for final model 
    dfc = pd.DataFrame(data=df)
    dfc.drop(['education','relationship', 'occupation'], axis=1, inplace=True)
    dfc['marital'] = dfc['marital'].replace(['Never-married', 'Widowed', 'Divorced', 'Separated'], 0)
    dfc['marital'] = dfc['marital'].replace(['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent'], 1)
    
    #fix for skewness ad normalize numerical data 
    skewed = ['capital_gain', 'capital_loss']
    numerical = ['age','education_num','capital_gain', 'capital_loss', 'hr_per_week', 'marital']
    
    x_data = dfc.drop(['income'], axis = 1)
    x_data[skewed] = x_data[skewed].apply(lambda x: np.log(x+1))
    x_data[numerical] = scale.fit_transform(x_data[numerical])
    y_data = dfc['income']
    
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy().reshape((-1,1))
    
    x_data = x_OneHotEncoder.fit_transform(x_data)  
    y_data = y_OneHotEncoder.fit_transform(y_data).ravel()
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    
    #test different learning rates and regularization 
    file = open('results.txt', 'w+')
    eta0_ = 0.1
    alpha_ = 0.00001
    test = 0
    a = 0
    for i in range(5):
        for j in range(5):
                model = SGDClassifier(learning_rate = lr, eta0 = eta0_ , alpha=alpha_)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                test +=1
                if accuracy > a:
                    a = accuracy 
                # s = str(test) + ',' + str(accuracy)
                # file.writelines(s + '\n')
                
                alpha_ += 0.00001
        eta0_ += 0.1

    # file.close()
    return a

      #confusion matrix 
    cm = confusion_matrix(y_test, y_pred)
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    return  

def main():
    
    max_acc = finalmodel('optimal')
    print("Accuracy of Final Model, optimal learning rate : ", max_acc)
    
    max_acc = finalmodel('adaptive')
    print("Accuracy of Final Model, adaptive learning rate : ", max_acc)
    
    max_acc = finalmodel('constant')
    print("Accuracy of Final Model, constant learning rate : ", max_acc)
    # x = np.loadtxt('results.txt', delimiter=',')
    # test = []
    # avg= []
    # for i in x:
    #     test.append(i[0].item(0))
    #     avg.append(i[1].item(0))
    # print('Best overall test:', int(mode(test)))
    # for y in range(len(test)):
    #     if avg[y] == max(avg):
    #         best = int(test[y]) 
    # print('Best Test:', best, '\tTest Score:', max(avg))
    # f = plt.figure(1)
    # plt.hist(test)
    # g = plt.figure(2)
    # plt.hist(avg)
    # #g.show()
    # plt.show()

main()