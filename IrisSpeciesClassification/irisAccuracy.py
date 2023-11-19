# Consider below characteristics of Machine Learning Application : 

# Classifier:       Decision Tree & K Nearest Neighbour
# DataSet:          Iris DataSet
# Features:         Sepal Width, Sepal Length, Petal Width, Petal Length
# Lables:           Versicolor, Sentosa, Virginica
# Training DataSet: 75 Entries
# Testing DataSet:  75 Entries

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def CalculateAccuracyDecisionTree():
    iris = load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=0.5)

    classifier = tree.DecisionTreeClassifier()

    classifier.fit(data_train,target_train)

    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)

    return Accuracy

def CalculateAccuracyKNeighbor():
    iris =load_iris()

    data = iris.data
    target = iris.target

    data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=0.5)

    classifier = KNeighborsClassifier(n_neighbors=9)

    classifier.fit(data_train,target_train)

    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)

    return Accuracy

def main():
    Accuracy = CalculateAccuracyDecisionTree()
    print("Accuracy of classification algorithm with Decision Tree classifier is",Accuracy*100,"%")

    Accuracy = CalculateAccuracyKNeighbor()
    print("Accuracy of classification algorithm with K Neighbor classifier is",Accuracy*100,"%")

if __name__=="__main__":
    main()