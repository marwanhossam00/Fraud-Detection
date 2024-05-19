import tkinter as tk
from datetime import datetime

import pandas as pd
import numpy as np
from tkinter import messagebox
import pickle

def preprocess_input_RF_data(data, columns_after_encoding):
    # Apply the same one-hot encoding
    dummies = pd.get_dummies(data[['merchant', 'category']])

    data['Time'] = pd.to_datetime(data['Time'])

    data['Year'] = data['Time'].dt.year
    data['Month'] = data['Time'].dt.month
    data['Day'] = data['Time'].dt.day
    data['Hour'] = data['Time'].dt.hour
    data['Minute'] = data['Time'].dt.minute

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


def preprocess_user_input(data, cat_targ_enc, merchant_targ_enc, columns_after_encoding):
    # Apply target encoding for 'category' and 'merchant'
    data['category_target_enc'] = data['category'].map(cat_targ_enc)
    data['merchant_target_enc'] = data['merchant'].map(merchant_targ_enc)

    # Extract time features
    data['Time'] = pd.to_datetime(data['Time'])
    data['Time:year'] = data['Time'].dt.year
    data['Time:month'] = data['Time'].dt.month
    data['Time:day'] = data['Time'].dt.day
    data['Time:hour'] = data['Time'].dt.hour

    # Drop unnecessary columns
    data = data.drop(columns=['category', 'merchant', 'Time'])

    # Log transform the 'amount' feature
    data['log_amount'] = np.log(data['amount'])
    data = data.drop(columns=['amount'])

    # Ensure all necessary columns are present
    for col in columns_after_encoding:
        if col not in data.columns:
            data[col] = 0

    # Ensure the order of columns is consistent
    data = data[columns_after_encoding]
    print('done with data preprocessing')
    return data


def xgBoostFunc(input_data):
    print('in xgBoostFunc')
    # Load the XGBoost model
    with open('xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
    with open('cat_targ_enc.pkl', 'rb') as file:
        cat_targ_enc = pickle.load(file)
    with open('merchant_targ_enc.pkl', 'rb') as file:
        merchant_targ_enc = pickle.load(file)
    with open('columns_after_encoding.pkl', 'rb') as file:
        columns_after_encoding = pickle.load(file)
    print('in xgBoostFunc2')

    input_data = input_data.drop(columns=['firstName', 'lastName', 'trans_num'])

    processed_data = preprocess_user_input(input_data, cat_targ_enc, merchant_targ_enc, columns_after_encoding )
    print('f2')
    # Perform prediction using the loaded model---- reshaping the data
    prediction = xgb_model.predict(processed_data)
    print(prediction)
    return prediction

def randomForest(input_data):
    # Load the trained model and column names
    with open('trained_randomForest_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('randomForest_columns.pkl', 'rb') as file:
        columns_after_encoding = pickle.load(file)

    # Perform prediction using the loaded model
    processed_data = preprocess_input_RF_data(input_data, columns_after_encoding)
    result = model.predict(processed_data)

    return result


def logisticRFunc(inputs):
    # Load the Logistic Regression model
    with open("logisticR.pkl", "rb") as f:
        model = pickle.load(f)
    # Perform prediction using the loaded model
    result = model.predict(inputs)
    return result


def check_fraud():
    # Retrieve user inputs
    input_data = {
        'Time': [time_entry.get()],
        'Card Number': [float(card_entry.get())],
        'merchant': [merchant_entry.get()],
        'category': [category_entry.get()],
        'amount': [float(amount_entry.get())],
        'firstName': [first_name_entry.get()],
        'lastName': [last_name_entry.get()],
        'trans_num': [transaction_entry.get()]
    }

    #input_df['Time'] = pd.to_datetime(input_df['Time'])
    input_df = pd.DataFrame(input_data)

    # Choose algorithm based on selected option
    selected_algo = algo_var.get()
    if selected_algo == "xgBoostAlgo":
        result = xgBoostFunc(input_df)
    elif selected_algo == "RandomForestAlgo":
        result = randomForest(input_df)
    elif selected_algo == "LogisticRAlgo":
        result = logisticRFunc(input_df)
    else:
        result = "Error: Algorithm not selected"

    # Display result
    print('prediction : ', result[0])
    if result[0] == 1:
        message = "Fraud!"
    elif result[0] == 0:
        message = "Not Fraud"
    else:
        message = "Invalid result"
    messagebox.showinfo("Fraud Detection Result", message)

# Create the main window
root = tk.Tk()
root.title("Fraud Detection")
root.geometry("400x400")  # Set window size to 400x400 pixels

# Create input fields with labels
fields = [
    ("Time:", 0),
    ("Card Number:", 1),
    ("Merchant:", 2),
    ("Category:", 3),
    ("Amount:", 4),
    ("First Name:", 5),
    ("Last Name:", 6),
    ("Transaction Number:", 7)
]

for field, row in fields:
    label = tk.Label(root, text=field, padx=10, pady=5)
    label.grid(row=row, column=0, sticky=tk.W)

    entry = tk.Entry(root)
    entry.grid(row=row, column=1)
    if field == "Card Number:" or field == "Transaction Number:":
        entry.config(show="*")

    if field == "Time:":
        time_entry = entry
    elif field == "Card Number:":
        card_entry = entry
    elif field == "Merchant:":
        merchant_entry = entry
    elif field == "Category:":
        category_entry = entry
    elif field == "Amount:":
        amount_entry = entry
    elif field == "First Name:":
        first_name_entry = entry
    elif field == "Last Name:":
        last_name_entry = entry
    elif field == "Transaction Number:":
        transaction_entry = entry

# Create dropdown menu for algorithm selection
algo_var = tk.StringVar(root)
algo_var.set("xgBoostAlgo")  # default value
algo_label = tk.Label(root, text="Choose Algorithm:", padx=10, pady=5, width=30)
algo_label.grid(row=8, column=0, sticky=tk.W)

algo_menu = tk.OptionMenu(root, algo_var, "xgBoostAlgo", "LogisticRAlgo", "RandomForestAlgo")
algo_menu.grid(row=8, column=1)

# Create "Check" button
check_button = tk.Button(root, text="Check", command=check_fraud)
check_button.grid(row=9, column=0, columnspan=2, pady=10)

root.mainloop()