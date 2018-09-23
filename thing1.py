from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print(iris.data)

# Show the target values (in numeric format) of each instance
print(iris.target)

# Show the actual target names that correspond to each number
print(iris.target_names)


#, y = np.arange(10).reshape((5, 2)), range(5)
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

print(data_train)
print(data_test)
print(target_train)
print(target_test)

classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

print(model)