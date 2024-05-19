import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Load the trained model and column names
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    columns_after_encoding = pickle.load(file)

def preprocess_input_data(data):
    # Apply the same one-hot encoding
    dummies = pd.get_dummies(data[['merchant', 'category']])
    data = pd.concat([data, dummies], axis=1)

    # Drop original categorical columns
    data = data.drop(columns=['Time', 'firstName', 'lastName', 'trans_num', 'merchant', 'category'])

    # Add missing columns with zeros
    for col in columns_after_encoding:
        if col not in data.columns:
            data[col] = 0

    # Ensure the order of columns is consistent
    data = data[columns_after_encoding]
    data = data.astype('float64')

    return data

def get_user_input():
    # Collect user input
    transaction_time_str = input("Enter transaction time (YYYY-MM-DD HH:MM:SS): ")
    transaction_time = pd.to_datetime(transaction_time_str)
    data = {
        'Time': [pd.to_datetime(transaction_time_str)],
        'Card Number': [float(input("Enter card number: "))],
        'merchant': [input("Enter merchant: ")],
        'category': [input("Enter category: ")],
        'amount': [float(input("Enter amount: "))],
        'firstName': [input("Enter first name: ")],
        'lastName': [input("Enter last name: ")],
        'trans_num': [input("Enter transaction number: ")],
        'Day': [float(transaction_time.day)],
        'Month': [float(transaction_time.month)],
        'Year': [float(transaction_time.year)],
        'Hour': [float(transaction_time.hour)],
        'Minute': [float(transaction_time.minute)]
    }

    input_data = pd.DataFrame(data)

    # Convert 'Time' to datetime if necessary
    input_data['Time'] = pd.to_datetime(input_data['Time'])
    pd.set_option('display.max_columns', None)
    print(input_data.head())

    return input_data

if __name__ == "__main__":
    input_data = get_user_input()

    # Preprocess the input data
    processed_data = preprocess_input_data(input_data)
   # processed_data = processed_data.astype('float64')

    # Predict using the model
    prediction = model.predict(processed_data)
    output = np.round(abs(prediction[0]))  # Since it's a regression model, round the output

    print(f'Fraud prediction: {output}')
