"""
Created on Sat May 10 17:35:52 2025
@author: Lorena Cujilema
"""
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------- Funciones ---------------------------------------
# Función para dividir los datos en entrenamiento y prueba
def split_data(X, y, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    print("Shape of the original boston data: ", X.shape)
    print("Shape of the boston train data = ", X_train.shape)
    print("Shape of the boston test data = ", X_test.shape)

    # return train_data, test_data 
    return X_train, X_test, y_train, y_test 
# ----------------------------------------------- Funciones ---------------------------------------
 
# Selección de características
boston_features = ['RM', 'ZN', 'PTRATIO', 'LSTAT'] # ['RM', 'ZN', 'LSTAT'] #
boston_labels = ['MEDV'] # ['medv'] # MEDV

# Cargar el dataset
boston = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
boston.columns = boston.columns.str.upper()
print(boston.columns)
X, y = boston[boston_features], boston[boston_labels]  

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = split_data(X, y) # Función split_data

# Definir características y etiquetas
boston_train_features = X_train[boston_features]
boston_train_labels = y_train # X_train[boston_labels]
 
# Entrenar el modelo
modelo = DecisionTreeRegressor(random_state=1)  

# Entrenar el modelo final con todos los datos de entrenamiento
modelo.fit(boston_train_features, boston_train_labels)
 
# Evaluar el modelo
y_pred = modelo.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Guardar el modelo en un archivo PKL
with open("modelo.pkl", "wb") as f:
     pickle.dump(modelo, f)

print("Modelo guardado en 'modelo.pkl'")


#print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
#Porque estás haciendo una regresión, y accuracy_score solo se usa para clasificación, no regresión.

#boston_test_features = X_test[boston_features]
#boston_test_labels = X_test[boston_labels]
