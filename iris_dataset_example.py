from sklearn import datasets
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

# The dataset contains a set of 150 records under five attributes -
# petal length, petal width, sepal length, sepal width and species

iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
print(digits.target)
print(iris.data)
print(iris.target)
print(iris.target_names)

iris_data = iris.data #variable for array of the data
iris_target = iris.target #variable for array of the labels

iris_test_ids = np.random.permutation(len(iris_data)) #randomly splitting the data set

#splitting and leaving last 15 entries for testing, rest for training
iris_train_one = iris_data[iris_test_ids[:-15]]
iris_test_one = iris_data[iris_test_ids[-15:]]
iris_train_two = iris_target[iris_test_ids[:-15]]
iris_test_two = iris_target[iris_test_ids[-15:]]
iris_classify = tree.DecisionTreeClassifier()#using the decision tree for classification
iris_classify.fit(iris_train_one, iris_train_two) #training or fitting the classifier using the training set
iris_predict = iris_classify.predict(iris_test_one) #making predictions on the test dataset

print("Labels Predicted")
print(iris_predict) #labels predicted (flower species)
print("Actual Labels")
print (iris_test_two) #actual labels
print("Accuracy")
print (accuracy_score(iris_predict, iris_test_two)*100) #accuracy metric
