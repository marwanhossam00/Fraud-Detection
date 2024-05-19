import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#########################################
data = pd.read_csv('fraudTrain.csv')
datatest = pd.read_csv('fraudTest.csv')

data.dropna(axis=0, how='any',inplace=True)
datatest.dropna(axis=0, how='any',inplace=True)

#########################################
# converting Time to dateTime class object
data['Time'] = pd.to_datetime(data['Time'])
datatest['Time'] = pd.to_datetime(datatest['Time'])

# adding month , day and year to the dataset

data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year


datatest['Day'] = datatest['Time'].dt.day
datatest['Month'] = datatest['Time'].dt.month
datatest['Year'] = datatest['Time'].dt.year


# adding hour and minute to the dataset
data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute

datatest['Hour'] = datatest['Time'].dt.hour
datatest['Minute'] = datatest['Time'].dt.minute

#########################################
dummies = pd.get_dummies(data[['merchant','category']])
newdata = pd.concat([data, dummies], axis = 1)
x = newdata.drop(columns=['ID','Time','firstName','lastName','trans_num','is_fraud','merchant','category'])
y = newdata['is_fraud']
#########################################
dummies2 = pd.get_dummies(datatest[['merchant','category']])
datatest = pd.concat([datatest, dummies2], axis = 1)
x3 = datatest.drop(columns=['ID','Time','firstName','lastName','trans_num','is_fraud','merchant','category'])
y3 = datatest['is_fraud']
#########################################
ss = 0.05
rus = RandomUnderSampler(sampling_strategy=ss)
x1, y1 = rus.fit_resample(x,y)
#########################################
model = RandomForestClassifier()
model.fit(x1,y1)
y_pred = model.predict(x1)
yy = model.predict(x3)
#########################################
accuracy = accuracy_score(y1, y_pred)
precision = precision_score(y1, y_pred)
recall = recall_score(y1, y_pred)
f1 = f1_score(y1, y_pred)
TP = np.sum(np.logical_and(y_pred == 1, y1 == 1))
TN = np.sum(np.logical_and(y_pred == 0, y1 == 0))
FP = np.sum(np.logical_and(y_pred == 1, y1 == 0))
FN = np.sum(np.logical_and(y_pred == 0, y1 == 1))
#########################################
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("#########################################")
#########################################
accuracy2 = accuracy_score(y3, yy)
precision2 = precision_score(y3, yy)
recall2 = recall_score(y3, yy)
f12 = f1_score(y3, yy)
#########################################
TP2 = np.sum(np.logical_and(yy == 1, y3 == 1))
TN2 = np.sum(np.logical_and(yy == 0, y3 == 0))
FP2 = np.sum(np.logical_and(yy == 1, y3 == 0))
FN2 = np.sum(np.logical_and(yy == 0, y3 == 1))
#########################################
print("Accuracy2:", accuracy2)
print("Precision2:", precision2)
print("Recall2:", recall2)
print("F1 Score2:", f12)
print("TP2:", TP2)
print("TN2:", TN2)
print("FP2:", FP2)
print("FN2:", FN2)

columns_after_encoding = x1.columns.tolist()

with open('randomForest_columns.pkl', 'wb') as file:
    pickle.dump(columns_after_encoding, file)

with open('trained_randomForest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print('model saved.')


# Without time (ss = 1, ts = 0.2):
# Accuracy: 0.94750656167979
# Precision: 0.9517966695880806
# F1 Score: 0.9476439790575916
# TP: 1086
# TN: 1080
# FP: 55
# FN: 65
# #########################################
# Accuracy2: 0.9528268783323947
# Precision2: 0.07088354845610782
# Recall2: 0.9268065268065268
# F1 Score2: 0.13169487595641086
# TP2: 1988
# TN2: 527516
# FP2: 26058
# FN2: 157
# With time (ss = 1, ts = 0.2)
# Accuracy: 0.9601924759405074
# Precision: 0.9717813051146384
# Recall: 0.9491817398794143
# F1 Score: 0.9603485838779956
# TP: 1102
# TN: 1093
# FP: 32
# FN: 59
# #########################################
# Accuracy2: 0.9743935334224672
# Precision2: 0.12415251601666978
# Recall2: 0.9305361305361305
# F1 Score2: 0.2190758423883218
# TP2: 1996
# TN2: 539493
# FP2: 14081
# FN2: 149

