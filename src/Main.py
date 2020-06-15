"""
    Classe responsavel por por controlar toda a execução do algoritmo Multi-layer Perceptron, tanto para sua fase
    treinamento, quanto para sua fase de teste
"""
from sklearn.neural_network import MLPClassifier #implementa o MLP
from sklearn.model_selection import train_test_split #define os conjuntos de treino e teste
from sklearn.preprocessing import StandardScaler #normalizacao
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #metricas
from sklearn.model_selection import GridSearchCV #pre-fit. otimizador dos hyperparametros do MLPClassifier

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")



##realiza a leitura do csv para pegar os dados para o MLP
train_df = pd.read_csv('../inputs/Part-1/caracteres-limpos.csv', header=None)
train_df = train_df.drop(labels=63, axis=1)
targets = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
]
targets = np.squeeze(targets)

## divide os data frames "train_df" e "targets" em conjunto de treino e conjunto de teste
train_df_x, train_df_test, targets_y, targets_test = train_test_split(train_df, targets, test_size= 7, stratify=targets)

###normalizacao
# scaler = StandardScaler()

###aplica o fit soment para os dados de treinamento
# scaler.fit(train_df_x)

###plica as transformacoes nos dados
# train_df_x = scaler.transform(train_df_x)
# train_df_test = scaler.transform(train_df_test)

###GridSearchCV
###define o MLP com a quantidade de iteracoes para o GrifSearch
mlp_model = MLPClassifier(
    max_iter=10000
)

# ###define os parametros que serao combinados pelo GridSearch
# parameter_space = {
#     'hidden_layer_sizes': [15, 20, 45, 63, 70],
#     'activation': ['relu', 'tanh', 'logistic'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.00001, 0.0001, 0.001],
#     'learning_rate': ['constant' ,'adaptive'],
#     'learning_rate_init': [0.25, 0.5, 0.6, 0.8, 0.7],
#     'tol': [0.00001, 0.000001, 0.0001, 0.001, 0.01, 0.0000001]
# }
#
# ##chamada que implementa o GridSearch
# clf = GridSearchCV(mlp_model, parameter_space, n_jobs=-1, cv=7)
#
# ##realiza o fit com para as combinacoes retornadas pelo GridSearch utilizando os parametros pre-definidos
# clf.fit(train_df_x, targets_y)
#
# ##log GridSearch
# print('Best estimator found:\n', clf.best_estimator_)
# print('Best parameters found:\n', clf.best_params_)
# print('Best index found:\n', clf.best_index_)
# print('Best score index found:\n', clf.best_score_)
# print('CV:\n', clf.cv)
# print('CV Result:\n', clf.cv_results_)
# print('Predict:\n', clf.predict)
# print('Error score:\n', clf.error_score)
# print('Param grid:\n', clf.param_grid)
# print('Multimetric:\n', clf.multimetric_)

##implementacao do MLP atribuindo os hyperparametros necessarios
mlp_model = MLPClassifier(hidden_layer_sizes=70,
                          max_iter=10000,
                          alpha=1e-05,
                          activation='logistic',
                          solver='sgd',
                          learning_rate='adaptive',
                          learning_rate_init=0.5,
                          tol=1e-07,
                          #shuffle=True,
                          verbose=True)

MLP_fit = mlp_model.fit(train_df_x, targets_y)

print('### PARAMETROS INICIAIS DA REDE ###\n--- NUMERO DE NEURONIOS ---\n')
print(f'Camada de Entrada:63\n')
print(f'Camada Escondida:{mlp_model.hidden_layer_sizes}\n')
print(f'Camada de Saida:{mlp_model.n_outputs_}\n')

print(f'--- PESOS ---\nCamada de Entrada:?{mlp_model.coefs_[0]}\n')
print(f'--- BIAS ---\nCamada de Entrada:?{mlp_model.intercepts_[0]}\n')

print('--- CONFIGURACOES DA REDE ---\n')
print(f'Funcao de Ativacao:{mlp_model.activation}\n')
print(f'Solver utilizado:{mlp_model.solver}\n')
print(f'Taxa de Aprendizado:{mlp_model.learning_rate}\n')
print(f'Taxa de Aprendizado Inicial:{mlp_model.learning_rate_init}\n')
print(f'Tol:{mlp_model.tol}\n')

predictions_proba = mlp_model.predict_proba(train_df_test)
predictions = mlp_model.predict(train_df_test)

print('### PARAMETROS FINAIS DA REDE ###\n')

###é uma lista de matrizes de peso, em que a matriz de peso no índice i representa os pesos entre a camada i e a camada i + 1.
print(f'--- PESOS ---\nCamada de Saida:{mlp_model.coefs_[1]}\n')

###é uma lista de vetores de bias, em que o vetor no índice i representa os valores de bias adicionados à camada i + 1.
print(f'--- BIAS ---\nCamada de Saida:{mlp_model.intercepts_[1]}\n')

print('### METRICAS ###')
print(f'--- ACURACIA ---\n{accuracy_score(targets_test, predictions)}\n')

##curva de erro x iteracao
print('--- ERRO X ITERACAO ---\nCurva do erro calculado em funcao da perda x iteracao.\n')
loss_curve = pd.DataFrame(mlp_model.loss_curve_)
graph = sns.relplot(ci=None, kind="line", data=loss_curve)
graph
#gg(loss_curve, aes(x='iterations', y='loss')) + gg.geom_line()

##matriz de confusao
print(f'--- MATRIZ DE CONFUSAO ---\n{confusion_matrix(targets_test.argmax(axis=1), predictions.argmax(axis=1))}\n')

##classificador
print(f'--- OUTRAS METRICAS DO CLASSIFICADOR ---\n{classification_report(targets_test.argmax(axis=1),predictions.argmax(axis=1))}\n')