from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
import numpy as np

def k_folds(train):
  SPLITS = 4
  MAX_NEIGHBORS = 40
  cross_validation = KFold(n_splits=SPLITS,shuffle=True)
  hiperparam = {}
  # iterado de los metodos de peso para comprobar cual se ajusta mejor
  for i, weights in enumerate(["uniform","distance"]):
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
              knn.fit(r_train.drop(columns=["quality"]),r_train["quality"])

              # prediccion del modelo usando los datos de entrenamiento 
              evaluation = knn.predict(r_test.drop(columns=["quality"]))

              accuracy.append(accuracy_score(r_test["quality"],evaluation))

          scores.append(np.mean(accuracy))
      
      # guardamos el valor máximo del peso para evaluarlo más adelante
      hiperparam[weights] = np.argmax(scores)+1
      
  return hiperparam