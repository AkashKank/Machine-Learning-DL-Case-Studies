# Consider Below characteristics of Machine Learning Application : 
# Classifier :         K Nearest Neighbour
# DataSet:             Play Predictor Dataset
# Features:            Whether, Temprature
# Lables:              Yes,No
# Training Dataset:    30 Entries
# Testing Dataset:     1 Entry


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    #Step 1 : Load data
    data = pd.read_csv(data_path,index_col=0)

    print("Size of Actual dataset",len(data))

    #step 2 : Clean,Prepare and Manipulate data
    feature_names = ['Whether','Temperature']

    print("Names of Features",feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers.
    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    # Converting string lablels into numbers
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print(temp_encoded)

    # combining weather and temp into single listof tuples
    features = list(zip(weather_encoded,temp_encoded)) 

    # Step 3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(features,label)

    # Step 4 : Test Data
    predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild
    print(predicted)


def main():
    print("---- CaseStudy of Play Predictor Application by Akash Kank ----")

    print("Machine Learning Application")

    print("Play Predictor application using K Nearest Knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__=="__main__":
    main()