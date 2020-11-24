import pandas as pd 
import numpy as np 
import graphviz
from sklearn import tree 
from sklearn.metrics import classification_report, accuracy_score

filepath = 'C:/Users/Mira/python/CSCI4520_HW1/'
#read in raw data 
train = pd.read_csv(filepath +'train.csv')
test = pd.read_csv(filepath + 'test.csv')

#correct column name from category to buy
train = train.rename(columns = {'Category' : 'Buy'})
test = test.rename(columns = {'Category' : 'Buy'})

#change yes/no string values in buy column to 1/0
train.replace(' Yes', np.int(1), inplace=True)
train.replace(' No', np.int(0), inplace=True)
test.replace('Yes', np.int(1), inplace=True)
test.replace('No', np.int(0), inplace=True)

#separate independent and dependent variable columns 
#'buy' is dependent var, 'type' and 'price' are independent 
train_buy = train[train.columns[-1]]
test_buy = test[test.columns[-1]]

train_buy = train_buy.to_frame()
test_buy = test_buy.to_frame()


#transform categorical variables to numerical variables via dummy variables 
train_type = pd.get_dummies(train.Type)
train_price = pd.get_dummies(train.Price)
test_type = pd.get_dummies(test.Type)
test_price = pd.get_dummies(test.Price)


#concatenate new dataframes as necessary 
train_data = pd.concat([train_type, train_price], axis=1)
test_data = pd.concat([test_type, test_price], axis=1)



#decision tree 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_buy)

#predict the label for test dataset 
test_pred = clf.predict(test_data)

temp = pd.concat([train_type, train_price, train_buy], axis=1)
print("Training set as numerical variables: " )
print(temp)
print("\nPrediction for test dataset, where 1 represents CD bought:")
print(test_pred)

#evaluation of the model 
print("\nAccuracy score: ", accuracy_score(test_buy, test_pred), "\n")
print("Classification report: ")
print(classification_report(test_buy, test_pred))

#plot decision tree 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=train_data.columns.values)
graph = graphviz.Source(dot_data)
graph.render("hw1")


