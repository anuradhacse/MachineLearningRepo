from flask import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd
from flask import request
import numpy as np

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.data
     print (json_)
     arr = json_.split(',')
     arr_reshape = np.array(arr).reshape(1,9)
     df = pd.DataFrame(np.array(arr_reshape))
     # query = pd.get_dummies(query_df)
     prediction = clf.predict(df)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run(port=8081)
