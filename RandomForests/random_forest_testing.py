# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

from sklearn_porter import Porter


# Set random seed
np.random.seed(0)


# load dataset
columns = ['dayOfWeek', 'hour', 'operationType', 'fileType', 'parentFolder', 'fileSize',
           'predessorFile1', 'predessorFile2', 'predessorFile3', 'predessorFile4', 'filename',
           'successorFile1', 'successorFile2', 'successorFile3', 'successorFile4']

# Create a dataframe with the four feature variables
dataframe = pd.read_csv("/home/anuradha/PycharmProjects/data/fyp/final/test-after-filtered.csv", header=None, names=columns)


# View the top 5 rows
# print(dataframe.head())

# Create a new column that for each row, generates a random number between 0 and 1, and
# if that value is less than or equal to .75, then sets the value of that cell as True
# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
# be used as the training data and some as the test data.
dataframe['is_train'] = np.random.uniform(0, 1, len(dataframe)) <= .9

# # View the top 5 rows
# # print(dataframe.head())
# # Create two new dataframes, one with the training rows, one with the test rows
train, test = dataframe[dataframe['is_train']==True], dataframe[dataframe['is_train']==False]



dataframe.drop(dataframe.columns[[0, 1,5]], axis=1, inplace=True)

# Create a list of the feature column's names
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 0:9]
Y = dataset[:, 8]
Y_train = np.asarray(Y, dtype= "|S6")
# View features


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=200, max_features= 'auto', n_jobs=-1,
                             random_state=0, verbose = 1, min_samples_leaf= 10
                             )

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(X,Y_train)

# Export:
porter = Porter(clf, language='java')
output = porter.export()
print(output)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
prdictions = clf.predict(X[:, 0:9])
print ("first ten predictions: ", prdictions[0:10])
print ("actual values : ",(Y[0:10]))

# View a list of the features and their importance scores
print(clf.feature_importances_)
pscore_train = metrics.accuracy_score(Y_train, prdictions)
print("Accuracy : ", pscore_train)

# View the predicted probabilities of the first 10 observations
# probs = clf.predict_proba(test.values[:, 0:9])[0:10]
# print ("probabilities: ", probs)


