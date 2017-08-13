import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# load dataset

# load dataset
columns = ['dayOfWeek' , 'formattedTime' , 'operationType' ,'fileType' ,'parentFolder' ,'fileSize',
           'predessorFile1','predessorFile2','predessorFile3','predessorFile4','filename',
           'successorFile1','successorFile2','successorFile3','successorFile4']
dataframe = pandas.read_csv("/home/anuradha/PycharmProjects/data/fyp/test2-08-12.csv", header = None, names = columns)
# dataset = dataframe.values
dataset = dataframe.values
scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

# print (dataframe.head())
#
# dataframe['operationType'].value_counts().plot(kind='bar', title='Training examples by file types');
# plt.show()

# split into input (X) and output (Y) variables
X = dataset[:,0:11]
Y = dataset[:,11]

# print (dataframe.shape)
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
# numpy.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
#
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=20, batch_size=5, verbose=2)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))



#
# X_train, X_test, y_train, y_test = train_test_split(X, Y)
# #
# # model = Sequential()
# # model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(1, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(1, kernel_initializer='normal'))
# # # Compile model
# # model.compile(loss='mean_squared_error', optimizer='adam')
# # model.fit(X_train, y_train, epochs=20, batch_size=5, verbose=2)
# #
# predictions  = estimators.__getattribute__('mlp').predict(X_test)
# print (predictions)
#
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))