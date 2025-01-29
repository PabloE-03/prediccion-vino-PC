from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

app = Flask(__name__)

def get_data():
  data = pd.read_csv('data.csv')
  y = data['quality']
  X = data.drop('quality', axis=1)
  return (X, y)

def create_models():
  classifier = KNeighborsClassifier(n_neighbors=3)
  regressor = KNeighborsRegressor(n_neighbors=3)
  
  classifier.fit(get_data()[0], get_data()[1])
  regressor.fit(get_data()[0], get_data()[1])
  
  pickle.dump(classifier, open('classifier.pkl', 'wb'))
  pickle.dump(regressor, open('regressor.pkl', 'wb'))

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/classifier')
def knn_classifier():
  params = request.get_json(force=True)
  classifier = pickle.load(open('classifier.pkl', 'rb'))
  prediction = classifier.predict(params)
  return jsonify(prediction)

@app.route('/regressor')
def knn_regressor():
  params = request.get_json(force=True)
  regressor = pickle.load(open('regressor.pkl', 'rb'))
  prediction = regressor.predict(params)
  return jsonify(prediction)


if __name__ == '__main__':
  app.run(
    host='localhost',
    port=5002,
    debug=True
  )