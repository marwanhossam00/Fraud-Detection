import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#########################################
data = pd.read_csv('fraudTrain.csv')
datatest = pd.read_csv('fraudTest.csv')
#########################################
dummies = pd.get_dummies(data[['merchant','category']])
newdata = pd.concat([data, dummies], axis = 1)
x = newdata.drop(columns=['ID','Time','Card Number','firstName','lastName','trans_num','is_fraud','merchant','category'])
y = newdata['is_fraud']
#########################################
dummies2 = pd.get_dummies(datatest[['merchant','category']])
datatest = pd.concat([datatest, dummies2], axis = 1)
x3 = datatest.drop(columns=['ID','Time','Card Number','firstName','lastName','trans_num','is_fraud','merchant','category'])
y3 = datatest['is_fraud']
#########################################
ss = 1
ts = 0.2
rus = RandomUnderSampler(sampling_strategy=ss)
xnew, ynew = rus.fit_resample(x,y)
x1, x2, y1, y2 = train_test_split(xnew,ynew,test_size=ts)
#########################################
model = RandomForestClassifier()
model.fit(x1,y1)
y_pred = model.predict(x2)
yy = model.predict(x3)
#########################################
accuracy = accuracy_score(y2, y_pred)
precision = precision_score(y2, y_pred)
recall = recall_score(y2, y_pred)
f1 = f1_score(y2, y_pred)
TP = np.sum(np.logical_and(y_pred == 1, y2 == 1))
TN = np.sum(np.logical_and(y_pred == 0, y2 == 0))
FP = np.sum(np.logical_and(y_pred == 1, y2 == 0))
FN = np.sum(np.logical_and(y_pred == 0, y2 == 1))
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
# Test 1(ss = 0.05, ts = 0.2):
# Accuracy: 0.9823296520108356
# Precision: 0.8087971274685817
# Recall: 0.810251798561151
# F1 Score: 0.8095238095238095
# TP: 901
# TN: 22670
# FP: 213
# FN: 211
# #########################################
# Accuracy2: 0.9896476456626461
# Precision2: 0.24176925279129688
# Recall2: 0.7874125874125875    
# F1 Score2: 0.3699485269959479  
# TP2: 1689
# TN2: 548277
# FP2: 5297
# FN2: 456
# #########################################
# Test 2 (ss = 1, ts = 0.002):
# Accuracy: 0.9565217391304348
# Precision: 1.0
# Recall: 0.9473684210526315
# F1 Score: 0.972972972972973
# TP: 18
# TN: 4
# FP: 0
# FN: 1
# #########################################
# Accuracy2: 0.9420408515814648
# Precision2: 0.058816623620568206
# Recall2: 0.9342657342657342
# F1 Score2: 0.11066626170030648
# TP2: 2004
# TN2: 521506
# FP2: 32068
# FN2: 141
# #########################################
# Test 3 (ss = 1, ts = 0.2):
# Accuracy: 0.9352580927384077
# Precision: 0.9292576419213974
# Recall: 0.9407603890362511
# F1 Score: 0.9349736379613357
# TP: 1064
# TN: 1074
# FP: 81
# FN: 67
# #########################################
# Accuracy2: 0.9394280202764347
# Precision2: 0.05651243949116289
# Recall2: 0.9361305361305361
# F1 Score2: 0.10659022745972344
# TP2: 2008
# TN2: 520050
# FP2: 33524
# FN2: 137