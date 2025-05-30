"""
Created on Sun May 11 16:25:55 2025
@author: Lorena Cujilema
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Función para mostrar la sección de inicio
def inicio():
    st.title("Bienvenido a la App de Predicción de Precios de Casas")
    st.markdown("""
    Esta aplicación predice el valor de una vivienda basado en algunas características 
    del conjunto de datos clásico de Boston Housing. Utiliza un modelo de regresión 
    entrenado con `scikit-learn` y visualizaciones interactivas.
    """)
    #imagen = Image.open("casa.jpg")
    imagen = Image.open("casa1.png")
    st.image(imagen, caption="Ejemplo de vivienda") # , use_column_width=True)
    
    st.write("Utiliza la opción 'Predicción' en el menú para probar el modelo con tus propias características.")


# Función para mostrar la sección "Acerca de"
# def acerca_de():
#     st.title("Acerca de")
#     st.markdown("""
#     **Aplicación creada por:** Adela Lorena Cujilema Tenenuela  
#     **Modelo:** Árbol de decisión (`DecisionTreeRegressor`)  
#     **Lenguaje:** Python  
#     **Framework:** Streamlit  
#     **Fuente de datos:** [BostonHousing Dataset](https://github.com/selva86/datasets/blob/master/BostonHousing.csv)  
#     """)
    
    
def acerca_de():
    st.title("Acerca de")
    st.write("Esta aplicación utiliza un modelo de Árbol de Decisión entrenado con el conjunto de datos **Boston Housing**.")
    
    image = Image.open("boston.jpg")
    st.image(image, caption="Boston Housing - Predicción de precios") # , use_column_width=True)

    st.write("El conjunto de datos contiene información sobre distintas características de viviendas en Boston y su precio medio (MEDV), expresado en miles de dólares.")

    # Crear una tabla con descripciones de las características usadas
    data = {
        'Característica': ['RM', 'ZN', 'PTRATIO', 'LSTAT'],
        'Descripción': [
            'Número promedio de habitaciones por vivienda',
            'Proporción de terrenos residenciales divididos en zonas para lotes grandes (> 25,000 pies²)',
            'Relación alumnos por maestro en cada ciudad',
            '% de población con bajo nivel socioeconómico'
        ],
        'Ejemplo de valores': ['4.5 - 8.7', '0.0 - 100.0', '12.6 - 22.0', '1.7 - 37.0']
    }

    st.table(data)

    st.write("""
    El modelo ha sido entrenado para **predecir el precio medio de una vivienda (MEDV)** a partir de estas características.

    Cuando uses la función de predicción, intenta ingresar valores realistas dentro de los rangos mostrados para obtener mejores resultados.

    ---
    **Autor:** Nalama1  
    **Repositorio:** [GitHub - proyecto](https://github.com/nalama1/proyectoStreamlit)  
    """)


    

# Función para mostrar la predicción
def prediccion():
    st.title("Predicción del precio de una casa")

    # Cargar modelo
    with open("modelo.pkl", "rb") as f:
        modelo = pickle.load(f)

    # Inputs
    st.subheader("Ingresa las características de la casa:")
    rm = st.number_input("RM: Promedio de habitaciones por vivienda", min_value=1.0, max_value=10.0, value=6.0)
    zn = st.number_input("ZN: Proporción de terrenos residenciales (>25,000 pies²)", min_value=0.0, max_value=100.0, value=12.5)
    ptratio = st.number_input("PTRATIO: Ratio alumno/maestro", min_value=10.0, max_value=30.0, value=18.0)
    lstat = st.number_input("LSTAT: % de población con estatus socioeconómico bajo", min_value=0.0, max_value=40.0, value=12.0)

    if st.button("Predecir precio"):
        entrada = np.array([[rm, zn, ptratio, lstat]])
        prediccion = modelo.predict(entrada)
        st.success(f"El precio estimado de la casa es: ${prediccion[0]*1000:,.2f}")

    # Histograma
    st.subheader("Distribución de precios en el dataset original")
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
    fig, ax = plt.subplots()
    ax.hist(df["medv"], bins=30, color="skyblue", edgecolor="black")
    ax.set_xlabel("Precio en miles de USD")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de precios MEDV")
    st.pyplot(fig)

# Menú principal
menu = st.sidebar.selectbox("Menú", ["Inicio", "Predicción", "Acerca de", "Salir"])

if menu == "Inicio":
    inicio()
elif menu == "Acerca de":
    acerca_de()
elif menu == "Predicción":
    prediccion()
elif menu == "Salir":
    st.stop()
