from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os.path as path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from prepross import preprocessing
from k_folds import k_folds

app = Flask(__name__)

def get_data():
  data = preprocessing()
  
  y = data['quality']
  X = data.drop('quality', axis=1)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  train = pd.concat([X_train, y_train], axis=1)
  
  return (train, X_test, y_test)

def create_models():
  data = get_data()
  
  hyperparams = k_folds(data[0])
  max_key = max(hyperparams, key=hyperparams.get)
  max_value = hyperparams[max_key]
  
  classifier = KNeighborsClassifier(n_neighbors=max_value, weights=max_key)
  regressor = KNeighborsRegressor(n_neighbors=max_value, weights=max_key)
  
  y = data[0]['quality'] 
  X = data[0].drop('quality', axis=1)
  
  classifier.fit(X, y)
  regressor.fit(X, y)
  
  pickle.dump(classifier, open('models/classifier.pkl', 'wb'))
  pickle.dump(regressor, open('models/regressor.pkl', 'wb'))
  
  print(f'Precisión del clasificador: {accuracy_score(data[2], classifier.predict(data[1]))*100}%')
  print(f'INFORME DE CLASIFICACION DEL CLASIFICADOR:\n {classification_report(data[2],classifier.predict(data[1]))}')
        
  print(f'Precisión del regresor: {accuracy_score(data[2], np.round(regressor.predict(data[1])))*100}%')
  print(f'INFORME DE CLASIFICACION DEL REGRESOR:\n {classification_report(data[2],np.round(regressor.predict(data[1])))}')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/classifier', methods=['POST'])
def knn_classifier():
  params = request.get_json(force=True)
  for keys in params:
    params[keys] = [params[keys]]
  data = pd.DataFrame(data=params)

  # escalado
  scaler = pickle.load(open('models/scaler.pkl',"rb"))
  data = scaler.transform(data)

  # prediccion
  classifier = pickle.load(open('models/classifier.pkl', 'rb'))
  prediction = classifier.predict(data)[0]
  return {'prediction': int(prediction)}

@app.route('/regressor', methods=['POST'])
def knn_regressor():
  params = request.get_json(force=True)
  for keys in params:
    params[keys] = [params[keys]]
  data = pd.DataFrame(data=params)

  # escalado
  scaler = pickle.load(open('models/scaler.pkl',"rb"))
  data = scaler.transform(data)

  # prediccion
  regressor = pickle.load(open('models/regressor.pkl', 'rb'))
  prediction = regressor.predict(data)[0]
  return {'prediction': int(prediction)}

if not path.exists('models/classifier.pkl') and not path.exists('models/regressor.pkl'):
  create_models()

if __name__ == '__main__':
  app.run(
    host='localhost',
    port=5002,
    debug=True
  )