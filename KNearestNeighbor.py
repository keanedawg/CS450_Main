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
    def predict(self, test_arr):
        print("OMG THIS IS A TEST TO SHOW I GOT THIS CODE TO RUN!!!")
        return [0] * len(test_arr)
class NearestNeighbor:
    # Gives array of input and output to help train the model
    def fit(self, data_train, target_train):
        sn = NearestNeighborModel()
        sn.data = data_train
        sn.target_train = target_train
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
classifier = NearestNeighbor()

# Calls the function to train the data then creates predictions
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# Useful for more accurately tuning any deviation (Uncomment helpful for debugging)
#print(targets_predicted)
#print(target_test)

# Print percentage correctly guessed
error = 1.0 - np.mean( target_test != targets_predicted )
print(error)