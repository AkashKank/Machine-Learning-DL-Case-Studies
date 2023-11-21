###################################################################################
# Required Python Packages
###################################################################################
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

###################################################################################
# File Paths
###################################################################################
INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"

###################################################################################
# Headers
###################################################################################
HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]

###################################################################################
# Function Name : read_Data
# Description : Read the data into pandas dataframe
# Input : path of CSV file
# Output : Gives the Data
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def read_data(path):
    data = pd.read_csv(path)
    return data

###################################################################################
# Function Name : get_headers
# Description : dataset headers
# Input : dataset
# Output : Returns the header
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def get_headers(dataset):
    return dataset.columns.values
   
###################################################################################
# Function Name : add_headers
# Description : add headers to the dataset
# Input : dataset
# Output : Updated Dataset
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

###################################################################################
# Function Name : data_file_to_csv
# Input : Nothing
# Output : Write the Data to csv
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def data_file_to_csv():
    #Headers
    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]
    # Load The data into pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add the headers to the loaded dataset
    dataset = add_headers(dataset,headers)
    # Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File Saved...!")

###################################################################################
# Function Name : split_dataset
# Description : split the dataset with the train_percentage
# Input : dataset with related information
# Output : Dataset After splitting
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def split_dataset(dataset, train_percentage, feature_headers, target_headers):
    # Split the dataset into train and test dataset
    train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],dataset[target_headers],train_size = train_percentage)
    return train_x,test_x,train_y,test_y

###################################################################################
# Function Name : handel_missing_values
# Description : filter missing values from  the dataset
# Input : dataset with missing values
# Output : Dataset remocking by missing values
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def handel_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header]!=missing_label]

###################################################################################
# Function Name : random_forest_classifier
# Description : To train the random forest classifier with features and target data
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def random_forest_classifier(features,target):
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf

###################################################################################
# Function Name : dataset_statistics
# Description : Basic statistics of the dataset
# Input : Dataset
# Output : Description of Dataset
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def dataset_statistics(dataset):
    print(dataset.describe())

###################################################################################
# Function Name : main
# Description : main function where execution gets start
# Author : Akash Atul Kank
# Date : 29/10/2023
###################################################################################
def main():
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH,index_col=0)
    # Get the basic statistics of loaded dataset
    dataset_statistics(dataset)

    # Filter missing values 
    dataset = handel_missing_values(dataset,HEADERS[6],'?')
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])

    # train and test dataset size details
    print("Train_x Shape :: ",train_x)
    print("Train_y Shape :: ",train_y)
    print("Test_X Shape :: ",test_x)
    print("Test_Y Shape :: ",test_y)

    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x,train_y)
    print("Trained model :: ",trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0,205):
        print("Actual outcome :: {} and predicted outcome :: {}".format(list(test_y)[i], predictions[i]))

    print("Train Accuracy :: ",accuracy_score(train_y,trained_model.predict(train_x)))
    print("Test Accuracy :: ",accuracy_score(test_y,predictions))
    print("Cofusion matrix :: ",confusion_matrix(test_y,predictions))

    ###################################################################################
    # Application starter
    ###################################################################################

if __name__=="__main__":
    main()