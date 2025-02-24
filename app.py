# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and test data
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

@app.route('/test', methods=['GET'])
def test():
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    return jsonify({'accuracy': accuracy})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)