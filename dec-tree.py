import pandas as pd
from sklearn import datasets
import numpy as np
import graphviz 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import tree
from sklearn.datasets import load_iris

# Get iris data from somewhere
iris = datasets.load_iris()

# Our trained machine learning model is actually a class.
# Thus, we can create multiple models trained with different sets
class NearestNeighborModel:
    # Given an array, uses our algorithm to predict the result of each element
    def predict(self, test_arr):
        return [0] * len(test_arr)
class NearestNeighbor:

    # init function should take in parameters (I.E. n_neighbors for NN algorithm)
    def __init__(self):
        return

    # Gives array of input and output to help train the model
    def fit(self, data_train, target_train):
        return NearestNeighborModel()


#################################################
#   Preprocessing Data
#################################################
def pp_iris(iris):
    data1 = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                         columns = iris['feature_names'] + ['target'])

    return data1


#################################################
# USE THESE VALUES TO SHOWCASE DATA
# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)
#################################################

# Splits the data randomly
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=91)

# Comment and Uncomment to switch to between algorhithms.
#classifier = GaussianNB()
#classifier = NearestNeighbor()
classifier = tree.DecisionTreeClassifier()

# Calls the function to train the data then creates predictions
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# Useful for more accurately tuning any deviation (Uncomment helpful for debugging)
print(targets_predicted)
print(target_test)

# Print percentage correctly guessed
error = 1.0 - np.mean( target_test != targets_predicted )
print(error)



iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

def visualize_graph(clf, name):
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(name) 

visualize_graph(clf, "iris")