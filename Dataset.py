import pandas as pd
import numpy as np

def load_data():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    df = pd.read_csv(url)
    return df

    # Read the local file instead!
    #df = pd.read_csv("ai4i2020.csv")
    return df

if __name__ == "__main__":
    df = load_data()
    print("Quick look at the data:")
    print(df.head())