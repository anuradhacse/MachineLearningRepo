from server import Flask, jsonify
from sklearn.externals import joblib
import pandas as pd

app = Flask(__name__)
@app.route('/predict', methods=['POST'])

def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = clf.predict(query)
     return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
     clf = joblib.load('model.pkl')
     app.run(port=8080)
