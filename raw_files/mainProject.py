#import main packages   ---this is the same code as the notebook

import pandas as pd
import warnings
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score,  mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import XGBRegressor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import preprocessing
warnings.filterwarnings('ignore')

#import main csv files
print('preparing data..')
traindata = pd.read_csv('fraudTrain.csv')
testdata = pd.read_csv('fraudTest.csv')

# converting Time to dateTime class object
traindata['Time'] = pd.to_datetime(traindata['Time'])
testdata['Time'] = pd.to_datetime(testdata['Time'])

# adding month , day and year to the dataset

traindata['Day'] = traindata['Time'].dt.day
traindata['Month'] = traindata['Time'].dt.month
traindata['Year'] = traindata['Time'].dt.year


testdata['Day'] = testdata['Time'].dt.day
testdata['Month'] = testdata['Time'].dt.month
testdata['Year'] = testdata['Time'].dt.year


# adding hour and minute to the dataset
traindata['Hour'] = traindata['Time'].dt.hour
traindata['Minute'] = traindata['Time'].dt.minute

testdata['Hour'] = testdata['Time'].dt.hour
testdata['Minute'] = testdata['Time'].dt.minute

# dropping nan raws
traindata.dropna(inplace = True)


print('encodeing data..')

#Encoding object columns in traindata
dummies = pd.get_dummies(traindata[['merchant','category']])
newdata = pd.concat([traindata, dummies], axis = 1)
x_train = newdata.drop(columns=['ID','Time','firstName','lastName','trans_num','is_fraud','merchant','category'])
y_train = newdata['is_fraud']

#Encoding object columns in testdata
dummies = pd.get_dummies(testdata[['merchant','category']])
newdata_test = pd.concat([testdata, dummies], axis = 1)
x_test = newdata_test.drop(columns=['ID','Time','firstName','lastName','trans_num','is_fraud','merchant','category'])
y_test = newdata_test['is_fraud']

print('resampling data..')
rus = RandomUnderSampler(sampling_strategy = 1)
x_train, y_train = rus.fit_resample(x_train,y_train)

x_test = x_test.astype(float)
x_train = x_train.astype(float)

#model training
print('training model..')
# 21 11
model = XGBClassifier(objective='reg:squarederror', random_state=1)
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
print('training done.')


#accuracy
print('calculating model accuracy....')


trainScore = accuracy_score(y_train, y_train_pred)
print('precision: %.2f' % precision_score(y_train, y_train_pred))
print('recall: %.2f' % recall_score(y_train, y_train_pred))
print('f1_score: %.2f' % f1_score(y_train, y_train_pred))
print(f'the train accuracy is {trainScore*100}%')
print(f'the train score is {trainScore}')
print(f'the mean squared error is {mean_squared_error(y_train, y_train_pred)}')
print('train Confusion Matrix:\n', confusion_matrix(y_train, y_train_pred))
print('---------------------------------------')

testScore = accuracy_score(y_test, y_test_pred)
print('precision: %.2f' % precision_score(y_test, y_test_pred))
print('recall: %.2f' % recall_score(y_test, y_test_pred))
print('f1_score: %.2f' % f1_score(y_test, y_test_pred))
print(f'the test accuracy is {testScore*100}%')
print(f'the mean squared error is {mean_squared_error(y_test, y_test_pred)}')
print(f'the test score is {testScore}')
print('test Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
print('---------------------------------------')



# trainScore = accuracy_score(y_train,np.round(abs(y_train_pred)))
# print('precision: %.2f' % precision_score(y_train, np.round(abs(y_train_pred))))
# print('recall: %.2f' % recall_score(y_train, np.round(abs(y_train_pred))))
# print('f1_score: %.2f' % f1_score(y_train, np.round(abs(y_train_pred))))
# print(f'the train accuracy is {trainScore*100}%')
# print(f'the train score is {trainScore}')
# print(f'the mean squared error is {mean_squared_error(y_train, y_train_pred)}')
# print('train Confusion Matrix:\n', confusion_matrix(y_train, np.round(abs(y_train_pred))))
# print('---------------------------------------')
#
# testScore = accuracy_score(y_test,np.round(abs(y_test_pred)))
# print('precision: %.2f' % precision_score(y_test, np.round(abs(y_test_pred))))
# print('recall: %.2f' % recall_score(y_test, np.round(abs(y_test_pred))))
# print('f1_score: %.2f' % f1_score(y_test, np.round(abs(y_test_pred))))
# print(f'the test accuracy is {testScore*100}%')
# print(f'the mean squared error is {mean_squared_error(y_test, y_test_pred)}')
# print(f'the test score is {testScore}')
# print('test Confusion Matrix:\n', confusion_matrix(y_test, np.round(abs(y_test_pred))))
# print('---------------------------------------')


