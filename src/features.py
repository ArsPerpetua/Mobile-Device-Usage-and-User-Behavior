import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE

# Adding project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def preprocess_data(data):
    # Fill missing values
    numerical_features = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
    ]
    for column in numerical_features:
        data[column].fillna(data[column].mean(), inplace=True)
    for column in ["Device Model", "Operating System", "Gender"]:
        data[column].fillna(data[column].mode()[0], inplace=True)

    # Encoding categorical variables
    le = LabelEncoder()
    data["Operating System"] = le.fit_transform(data["Operating System"])
    data["Gender"] = le.fit_transform(data["Gender"])
    data = pd.get_dummies(data, columns=["Device Model"], drop_first=True)

    return data


def split_data(data):
    # Define the columns you want to keep
    features_to_keep = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 
                         'Battery Drain (mAh/day)', 'Number of Apps Installed', 
                         'Data Usage (MB/day)', 'Age', 'Operating System', 'Gender']
    
    # Ensure only relevant columns are used
    X = data[features_to_keep]
    y = data['User Behavior Class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    features = X.columns
    return X_train, X_test, y_train, y_test, features, scaler

