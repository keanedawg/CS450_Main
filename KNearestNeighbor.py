from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Get iris data from somewhere
iris = datasets.load_iris()

# Our trained machine learning model is actually a class.
# Thus, we can create multiple models trained with different sets
class NearestNeighborModel:
    # Given an array, uses our algorithm to predict the result of each element
    def predict(self, test_arr): # ASSUME K=3 FOR NOW, DELETE LATER
        # Calculates the result mapping every single value
        result = list(map(lambda v: self.getNearestNeighbors(v), test_arr))
        return result

    # returns nearest neighbor classification
    # k is number of neighbors, v is value to find neighbors for 
    def getNearestNeighbors(self, v):  
        # Get all the distances from the value V 
        dists = list(map(lambda x: np.linalg.norm(x-v), self.id))
        # Now sort them (note we sort both the dist data and the target data)
        sot = [x for _, x in sorted(zip(dists, self.ot))] # sot = sorted output target
        ck = sot[:self.k] # ck = closest k neighbors
        # This is a funky way of getting the mode
        mode = max(set(ck), key=ck.count)
        return mode


class NearestNeighbor:

    def __init__(self, n_neighbors):
        self.k = n_neighbors
        return

    # Gives array of input and output to help train the model
    def fit(self, data_train, target_train):
        sn = NearestNeighborModel()
        sn.id = data_train # id = input data
        sn.ot = target_train # ot = output target
        sn.k = self.k  
        return sn

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
    iris.data, iris.target, test_size=0.3, random_state=39)

# Comment and Uncomment to switch between various implementations
#classifier = GaussianNB()
#classifier = KNeighborsClassifier(n_neighbors=1)
classifier = NearestNeighbor(n_neighbors=3)

# Calls the function to train the data then creates predictions
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# Useful for more accurately tuning any deviation (Uncomment helpful for debugging)
#print(targets_predicted)
#print(target_test)

# Print percentage correctly guessed
error = 1.0 - np.mean( target_test != targets_predicted )
print(error)