#import main packages   ---this is the same code as the notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler # for scaling data
warnings.filterwarnings('ignore')


#import main csv files
data = pd.read_csv('fraudTrain.csv')



print(data.head())


# feature engineering

# converting Time to dateTime class object
data['Time'] = pd.to_datetime(data['Time'])

print(data['Time'].head())
print(data['Time'].tail())

print(data['Time'].describe())

# adding mont and day and year to the dataset

data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year

print(data.head())

# adding hour and minute to the dataset
data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute

print(data.head())


# no need for the time column, it has been substituted with day, month, year and time
data = data.drop(['Time'], axis=1)

print(data.head())


#the number of unique value
print('----------------')
print(data['is_fraud'].nunique())
print(data['is_fraud'].unique())


# data cleansing
print(pd.value_counts(data['is_fraud']))
print(data.shape)

#count the number of nulls
print(data.isnull().sum())

# dropping nan raws
data.dropna(inplace = True)

#count the number of nulls
print('nulls:', data.isnull().sum())

#Encoding object columns
label_encoding=preprocessing.LabelEncoder()
cols=['merchant', 'category', 'firstName', 'lastName', 'trans_num']
for i in cols:
    data[i]=label_encoding.fit_transform(data[i])

# check if any data is duplicated
print(data.duplicated().sum())


#check outlires
cols = data.columns
for i in cols:
    q1 = np.percentile(data[i], 25)
    q3 = np.percentile(data[i], 75)
    norm_range = (q3 - q1) * 1.5

    lower_outliers = data[data[i] < (q1 - norm_range)] # lower ouliers
    upper_outliers = data[data[i] > (q3 + norm_range)] # upper outliers


    outliers = len(lower_outliers) + len(upper_outliers) #total number of outliers

    print(f"number of outliers in column {i} is {outliers}")

    # Replace outliers
    data[i] = np.where(data[i] < (q1 - norm_range), q1 - norm_range, data[i])
    data[i] = np.where(data[i] > (q3 + norm_range), q3 + norm_range, data[i])


x = data.drop(columns=['is_fraud'])
y = data['is_fraud']
print('Class distribution in y:')
print(y.value_counts())
print(data['is_fraud'].nunique())

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y, test_size=0.1, random_state = 21, stratify=y)
print(len(x_train),len(y_train),len(x_test),len(y_test))

print('Class distribution in y_train:')
print(y_train.value_counts())
print('Class distribution in y_test:')
print(y_test.value_counts())

#logistic regression model training

#scaling the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print('the trainig after scaling shape:', x_train_scaled.shape)
# training the model
log_reg = LogisticRegression(random_state = 0).fit(x_train_scaled, y_train)
prediction = log_reg.predict(x_train_scaled)

#accuracy
trainAccuracy = log_reg.score(x_train_scaled,y_train)
testAccuracy = log_reg.score(x_test_scaled,y_test)

print(f'the train score is {log_reg.score(x_train_scaled, y_train)}')
print(f'the train accuracy is {trainAccuracy}%')
print('---------------------------------------')
print(f'the test accuracy is {testAccuracy}%')
print(f'the test score is {log_reg.score(x_test_scaled, y_test)}')