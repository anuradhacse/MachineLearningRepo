import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import keras.layers.advanced_activations
import tensorflow as tf;
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib


MODEL_NAME = 'regression_model'

# load dataset
columns = ['dayOfWeek', 'hour', 'operationType', 'fileType', 'parentFolder', 'fileSize',
           'predessorFile1', 'predessorFile2', 'predessorFile3', 'predessorFile4', 'filename',
           'successorFile1', 'successorFile2', 'successorFile3', 'successorFile4']

dataframe = pandas.read_csv("/home/anuradha/PycharmProjects/data/fyp/final/test-after-filtered.csv", header=None, names=columns)
# dataset = dataframe.values
dataset = dataframe.values

dataframe.drop(dataframe.columns[[0, 1,5]], axis=1, inplace=True)

print(dataframe.describe())

# split into input (X) and output (Y) variables
X = dataset[:, 0:9]
Y = dataset[:, 8]

# fix random seed for reproducibility
seed = 7

# evaluate model with standardized dataset
numpy.random.seed(seed)

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

# normalizedX = preprocessing.scale(X)

# summarize transformed data
# numpy.set_printoptions(precision=3)
print(normalizedX[0:7,:])


# create model
model = Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation="relu", use_bias= True))
model.add(Dense(1, kernel_initializer='normal', activation="relu"))


model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(normalizedX, Y)

model.fit(X_train, y_train, batch_size=20, epochs=2000, verbose=2)
score = model.evaluate(X_test, y_test, batch_size=1, verbose=2)

print("score : ",score)
model.save("models/regression_model.h5")

prediction = model.predict(X_test)
print (prediction)
prediction = prediction.astype(int)


plt.show()

print(prediction[:10])
print (y_test[:10])


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







