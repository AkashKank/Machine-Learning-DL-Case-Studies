import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot

def AdvertisingPredictor(data_path):
    data = pd.read_csv(data_path,index_col=0)
    print(data)

    print("Size of Actual Dataset : ",len(data))

    Feature_names = ['TV','radio','newspaper']
    print("Names of Features : ",Feature_names)

    X = data[Feature_names]
    print(X)

    y = data.sales

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/2)

    print("Size of Training Dataset",len(X_train))

    print("Size of Testing Dataset",len(X_test))

    linreg = LinearRegression()

    linreg.fit(X_train,y_train)

    predictions = linreg.predict(X_test)

    print("Testing Set")
    print(X_test)

    print("Result of Testing : ")
    print(predictions)

    print("Mean Square")
    print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))


def main():
    print("Machine Learning Case Study using Advertising Dataset")

    AdvertisingPredictor("Advertising.csv")

if __name__=="__main__":
    main()