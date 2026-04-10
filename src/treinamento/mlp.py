import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import tensorflow as tf
from tensorflow import  Sequential, Dense
from tensorflow import keras.layers.InputLayer as InputLayer


QTDE_CLASSES = 7

FUNCAO_N_LINIEAR = "sigmoid"
ALTURA_DA_REDE = 512
PROFUNDIDADE_DA_REDE = 3
FUNCAO_SAIDA = "softmax"

# Params de aprendizado
PASSO_DE_AJUSTE = 0.05 # alfa ou learning rate(lr)
FUNCAO_DE_PERDA = "loss" # Loss function
OTIMIZADOR = "Adam" # Adaptative Momentum - Optmizer, Nadam - Nestorv Adaptative Momentum  
FUNCAO_DE_MONITORAMENTO = "accuracy"
QTDE_INTERACOES = 50
VALIDACAO_TREINAMENTO = 0.15

imagens = []

model = Sequential([
    model.Add.InputLayer(imagens),
    for i in range(PROFUNDIDADE_DA_REDE): 
        model.Add.Dense(ALTURA_DA_REDE, activation=FUNCAO_N_LINIEAR)
    model.Add.Dense(QTDE_CLASSES, activation=FUNCAO_SAIDA),  
])

model.compile(optimizer=OTIMIZADOR,
              loss=FUNCAO_DE_PERDA,
              metrics=[FUNCAO_DE_MONITORAMENTO])

mod = model.fit(x_train, y_train, epochs=QTDE_INTERACOES, 
                batch_size=2000, 
                validation_split=VALIDACAO_TREINAMENTO)
          
