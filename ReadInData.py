import pandas
cars = pandas.read_csv("Data/cars.csv")

def preprocess_cars_data(car_data):
    car_data["buying"] = car_data["buying"].astype('category')
    car_data["maint"] = car_data["maint"].astype('category')
    car_data["doors"] = car_data["doors"].astype('category')
    car_data["persons"] = car_data["persons"].astype('category')
    car_data["lug_boot"] = car_data["lug_boot"].astype('category')
    car_data["distr"] = car_data["distr"].astype('category')
    car_data["safety"] = car_data["safety"].astype('category')
    car_data["buying"] = car_data["buying"].cat.codes
    car_data["maint"] = car_data["maint"].cat.codes
    car_data["doors"] = car_data["doors"].cat.codes
    car_data["persons"] = car_data["persons"].cat.codes
    car_data["lug_boot"] = car_data["lug_boot"].cat.codes
    car_data["distr"] = car_data["distr"].cat.codes
    car_data["safety"] = car_data["safety"].cat.codes
    return (car_data.values, car_data["safety"].values )

foo = preprocess_cars_data(cars)
data = foo[0]
target = foo[1]
print(data)

# Last weeks code
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier

# Choose Dataset
myData = datasets.load_iris()
#myData = foo

#print(dir(myData))
#print(myData.target)

# Nearest Neighbor Model used to reach requirements
# This is a naive approach to nearest neighbor
class NearestNeighborModel:
    # Given an array, uses our algorithm to predict the result of each element
    def predict(self, test_arr): 
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
        ck = sot[:self.k] # ck = closest k neighbors (represented as indices)
        # This is a funky way of getting the mode
        mode = max(set(ck), key=ck.count)
        return mode

# This gets the naive nearestNeighbor model
class NearestNeighbor:

    def __init__(self, n_neighbors):
        self.k = n_neighbors
        return

    # Gives array of input and output to help train the model
    def fit(self, data_train, target_train):
        sn = NearestNeighborModel() # sn = self NearestNeighborModel
        sn.id = data_train # id = input data
        sn.ot = target_train # ot = output target
        sn.k = self.k  
        return sn


# Nearest Neighbor Model used to go above and beyond
# This uses SKLearn's KDTree and not my own
class KDTreeNearestNeighborModel:
    # Given an array, uses our algorithm to predict the result of each element
    def predict(self, test_arr):
        # initialize the KDTree with our data
        tr = KDTree(self.id) # tr = tree
        # sends test_arr through KDTree
        nd, ni = tr.query(test_arr, k=self.k) # nd = nearest-distance, ni = nearest indice
        # Transforms list of list of nearest neighbors to list of list of respective classifications
        c = list(map(lambda x: list(map(lambda y: self.ot[y], x)), ni)) # c = classifications
        # Gets the Mode (most common type) of each list of respective classifications
        c = list(map(lambda ck: max(set(ck), key=ck.count), c))
        return c

# This gets the KD implemented nearestNeighbor model
class KDTreeNearestNeighbor:
    # init function should take in parameters (I.E. n_neighbors for NN algorithm)
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        return

    # Gives array of input and output to help train the model
    def fit(self, data_train, target_train):
        sn = KDTreeNearestNeighborModel()
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
    data, target, test_size=0.3, random_state=55)

# Comment and Uncomment to switch between various implementations
#classifier = GaussianNB() # Just a reference point, not really a nearestNeighbor algorithm
#classifier = KNeighborsClassifier(n_neighbors=2)
#classifier = NearestNeighbor(n_neighbors=2)
classifier = KDTreeNearestNeighbor(n_neighbors=2)

# Calls the function to train the data then creates predictions
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# Useful for more accurately tuning any deviation (Uncomment helpful for debugging)
#print(targets_predicted)
#print(target_test)

# Print percentage correctly guessed
error = 1.0 - np.mean( target_test != targets_predicted )
print(error)