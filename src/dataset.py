import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    print("Info dataset:")
    print(data.info())
    print("\nStatistik deskriptif:")
    print(data.describe())
    return data
