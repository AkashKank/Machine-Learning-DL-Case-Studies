import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def HeadBrainPredictor(data_path):
    # Step 1 : Load the Data
    data = pd.read_csv(data_path,index_col=0)
    print("Size of Data Set : ",len(data))
    print(data)

    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values

    X = X.reshape(-1,1)

    n = len(X)

    reg  = LinearRegression()

    reg = reg.fit(X,Y)

    Predictions = reg.predict(X)

    r2 = reg.score(X,Y)

    print(r2)

def main():
    print("Supervised Machine Learning")

    print("Linear Regression on Head and Brain size data set")

    HeadBrainPredictor("HeadBrain.csv")

if __name__=="__main__":
    main()