import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

def preprocessing():
    data = pd.read_csv('./data/winequality-red.csv')
    
    # Eliminamos los campos que no son necesarios
    data = data.drop(columns=['pH','free sulfur dioxide', 'residual sugar'], axis=1)
    
    # Cambiamos los outliers por la mediana
    K = 1.5
    columns = ["fixed acidity", "volatile acidity", "chlorides", "density","sulphates","total sulfur dioxide","citric acid"]

    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        top_margin = Q3 + IQR * K
        low_margin = Q1 - IQR * K

        data[column] = data[column].apply(lambda x: data[column].median() if x < low_margin or x > top_margin else x)
    
    # Normalizamos los datos
    mm_scaler =  MinMaxScaler()
    mm_scaler.fit(data.drop(columns=["quality","density","chlorides"]))
    data_norm = mm_scaler.transform(data.drop(columns=["quality","density","chlorides"]))

    prueba = pd.DataFrame(data=data_norm,columns=data.drop(columns=["quality","density","chlorides"]).columns)
    data[prueba.columns] = data_norm
    
    # Arreglamos el desbalanceo en los datos mediante SMOTE
    rus = SMOTE()
    quality = data["quality"]

    caract_X, caract_Y = rus.fit_resample(data.drop(columns=["quality"]),quality)
    caract_X['quality'] = caract_Y
    data = caract_X  
    
    # Escalamos los datos mediante StandardScaler
    scaler = StandardScaler()
    scaler.fit(data.drop(columns=["quality"]))
    
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    
    data_norm = scaler.transform(data.drop(columns=["quality"]))
    
    y = data['quality']
    data = data.drop(columns=['quality'])
    data[data.columns] = data_norm
    data["quality"] = y
    return data