import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn 

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MinMaxScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
x_OneHotEncoder = OneHotEncoder()
y_OneHotEncoder = LabelBinarizer()
scale = MinMaxScaler()

df = pd.read_csv('adult.data')

#no missing attributes found 

#split columns based on necessary preprocessing needed 
skewed = ['capital_gain', 'capital_loss']
numerical = ['age','education_num', 'fnlwgt', 'capital_gain', 'capital_loss', 'hr_per_week']
categorical = ['type_employer', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex', 'country']

raw_x_data = df.drop(['income'], axis = 1)
y_data = df['income']
    
#perform log transform on skewed variables
x_log = pd.DataFrame(data = raw_x_data)
x_log[skewed] = raw_x_data[skewed].apply(lambda x: np.log(x+1))

#normalize numerical features on the interval [0,1]
x_log_scaled = pd.DataFrame(data = x_log)
x_log_scaled[numerical] = scale.fit_transform(x_log[numerical])


#transform categorical features
x_data = x_log_scaled.to_numpy()
y_data = y_data.to_numpy().reshape((-1,1))

x_data = x_OneHotEncoder.fit_transform(x_data)  
y_data = y_OneHotEncoder.fit_transform(y_data).ravel()

#training model 
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

model = SGDClassifier()
model.fit(x_train, y_train)     

#calculate training and validation scores 
training_score = model.score(x_train, y_train)
validation_score = model.score(x_test, y_test)
print(f'\nTraining Score: {training_score}')
print(f'Validation Score: {validation_score}')

#calculate accuracy 
y_pred = model.predict(x_test)

print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))


#confusion matrix 
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')

"""
Data visualization to help choose best features 
"""
df = pd.read_csv('adult.data')
dfc = df.copy()

#set income to numeric 
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
dfc['income'] = df['income']

#marital status 
fix, axe = plt.subplots(2, 1,  figsize=(30,10))
fig = sn.barplot(x='marital', y='income', data=df, ax=axe[0], order=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
fig.set_xlabel("Marital Status")
fig.set_ylabel("Proability of Income >50k")

#set marital status to numeric where 1 "married" and 0 is "single"
dfc['marital'] = dfc['marital'].replace(['Never-married', 'Widowed', 'Divorced', 'Separated'], 0)
dfc['marital'] = dfc['marital'].replace(['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent'], 1)

fig = sn.barplot(x='marital', y='income', data=dfc, ax=axe[1], order=[1,0])
fig.set_xlabel("Marital Status")
fig.set_ylabel("Proability of Income >50k")

#correlation matrix of all numeric data (including income and marital)
cols = ['age', 'education_num', 'fnlwgt', 'capital_gain', 'capital_loss', 'hr_per_week', 'marital', 'income']
fig, axe = plt.subplots(figsize=(20,20))
sn.heatmap(dfc[cols].corr(), annot=True)

"""Final model"""
#preparing data for final model 
dfc = pd.DataFrame(data=df)
dfc.drop(['sex','type_employer', 'education', 'race', 'relationship', 'occupation', 'country' ], axis=1, inplace=True)
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

model = SGDClassifier()
model.fit(x_train, y_train) 


#calculate accuracy 
y_pred = model.predict(x_test)

print('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))

#confusion matrix 
cm = confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')

