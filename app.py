from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os.path as path
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from prepross import preprocessing
import numpy as np

app = Flask(__name__)

def get_data():
  data = pd.read_csv('data/data.csv')
  data = preprocessing(data)
  
  y = data['quality']
  X = data.drop('quality', axis=1)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  
  return (X, y)

def create_models():
  # la longitud del conjunto de datos de entrenamiento es 160 por lo que se partira en 4 partes
  SPLITS = 4
  # como la longitud del conjunto de datos de entreno es 160 40 vecinos se ajusta muy bien como maximo
  MAX_NEIGHBORS = 40
  cross_validation = KFold(n_splits=SPLITS,shuffle=True)
  hiperparam = {}
  # iterado de los metodos de peso para comprobar cual se ajusta mejor
  for i,weights in enumerate(["uniform","distance"]):
      scores = []
      # iterado de numero de vecinos para saber cual es el numero que mejor se ajusta
      for neighbor in range(1,MAX_NEIGHBORS):
          accuracy = []
          knn = KNeighborsClassifier(neighbor,weights=weights)
          # prueba del modelo del minimo de vecinos al maximo de vecinos
          for train_fold,test_fold in cross_validation.split(train):
              
              # seleccion aleatoria de datos de entrenamiento mediante indices
              r_train = train.iloc[train_fold]
              r_test = train.iloc[test_fold]

              # entrenamiento del modelo
              knn.fit(r_train.drop(columns=["label"]),r_train["label"])

              # prediccion del modelo usando los datos de entrenamiento 
              evaluation = knn.predict(r_test.drop(columns=["label"]))

              accuracy.append(accuracy_score(r_test["label"],evaluation))

          scores.append(np.mean(accuracy))
      
      # guardamos el valor máximo del peso para evaluarlo más adelante
      hiperparam[weights] = np.argmax(scores)+1
      

  
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

if not path.exists('classifier.pkl') and not path.exists('regressor.pkl'):
  create_models()

if __name__ == '__main__':
  app.run(
    host='localhost',
    port=5002,
    debug=True
  )