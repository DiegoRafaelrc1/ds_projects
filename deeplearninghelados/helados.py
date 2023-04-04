# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:55:03 2022

@author: rafae
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

df_helados= pd.read_csv("C:/Users/rafae/OneDrive/Documentos/Projects_programming/AI/deeplearninghelados/IceCreamData.csv")

#sns.scatterplot(df_helados,x="Temperature",y="Revenue")

x_train=df_helados["Temperature"]
y_train=df_helados["Revenue"]

#creando modelo
modelo=tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

modelo.summary()

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.5),loss="mean_squared_error")

#entrenamiento
epoch_his=(modelo.fit(x_train,y_train,epochs=1000))



plt.plot(epoch_his.history["loss"])
plt.title("Progreso de perdida helados")
plt.ylabel("Temperatura")
plt.xlabel("EPOCH")
plt.legend("training loss")

#prediccion
weights = modelo.get_weights()


#temp=35
#rev=modelo.predict([temp])
#print("La ganancia segun la red neuronal sera de ",rev," dolares")

#Grafico de predicción


plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,modelo.predict(x_train),color="blue")
plt.xlabel("Temperatura")
plt.ylabel("Dolares")
plt.title("Temperatura vs Ganancia")



