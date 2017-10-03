import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import keras.layers.advanced_activations
import tensorflow as tf;
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib


MODEL_NAME = 'regression_model'

# load dataset

# load dataset
columns = ['dayOfWeek', 'hour', 'operationType', 'fileType', 'parentFolder', 'fileSize',
           'predessorFile1', 'predessorFile2', 'predessorFile3', 'predessorFile4', 'filename',
           'successorFile1', 'successorFile2', 'successorFile3', 'successorFile4']
dataframe = pandas.read_csv("/home/anuradha/PycharmProjects/data/fyp/test_v3.csv", header=None, names=columns)
# dataset = dataframe.values
dataset = dataframe.values
scaler = MinMaxScaler(feature_range=(0, 1))

# split into input (X) and output (Y) variables
X = dataset[:, 0:11]
Y = dataset[:, 11]

dataset = scaler.fit_transform(X)


# print (dataframe.head())
#
# dataframe['operationType'].value_counts().plot(kind='bar', title='Training examples by file types');
# plt.show()



# print (dataframe.shape)
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# define the model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation="relu"))
    model.add(Dense(1, kernel_initializer='normal', activation="relu"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
                         MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
                              False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
                              "save/restore_all", "save/Const:0", \
                              'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

    return


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
estimator = KerasRegressor(build_fn=larger_model, epochs=20, batch_size=50, verbose=2)


# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))



#
X_train, X_test, y_train, y_test = train_test_split(X, Y)
estimator.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
# #
# # model = Sequential()
# # model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(1, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(1, kernel_initializer='normal'))
# # # Compile model
# # model.compile(loss='mean_squared_error', optimizer='adam')
# # model.fit(X_train, y_train, epochs=20, batch_size=5, verbose=2)
# #

prediction = estimator.predict(X_test)
print (prediction)
prediction = prediction.astype(int)
# print (" test X shape" + X.shape)
# print (" predictions shape : " , prediction.shape)
# print(prediction[:10])
# print (y_test[:10])

print(K.get_session().graph.get_operations())
#
# print(K.get_session().graph.get_all_collection_keys())



# print(n.name) for n in tf.get_default_graph().as_graph_def().node

export_model(tf.train.Saver(), estimator, ["dense_1_input"], "dense_2/Relu")

# invert predictions
# trainY = scaler.inverse_transform(numpy.array(Y))
# print(trainPredict)

# predictions  = estimator.predict(X_test)
# print (predictions)
#
# print(accuracy_score(y_test,prediction))
# print(classification_report(y_test,predictions))
