from sklearn import metrics
from sklearn.neural_network import MLPClassifier #implementa o MLP
from sklearn.model_selection import train_test_split #seleciona os dados dividindo-os entre conjunto de treino e conjunto de teste
from sklearn.preprocessing import StandardScaler #pre processamento dos dados normalizando-os utilizando StandardScaler
from sklearn.metrics import classification_report, confusion_matrix #classificador e matriz de confusao
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

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
## o problema sepa ta no target

## divide os data frames "train_df" e "targets" em conjunto de treino e conjunto de teste
train_df_x, train_df_test, targets_y, targets_test = train_test_split(train_df, targets)

##normalizacao
# scaler = StandardScaler()

# aplica o fit soment para os dados de treinamento
# scaler.fit(train_df_x)

#aplica as transformacoes nos dados
# train_df_x = scaler.transform(train_df_x)
# train_df_test = scaler.transform(train_df_test)

##implementacao do MLP atribuindo o quantidade de neuronios presentes em cada camada escondida
# mlp_model = MLPClassifier(solver='sgd',
#                           learning_rate='constant',
#                           learning_rate_init=0.5,
#                           hidden_layer_sizes=(30,),
#                           random_state=1,
#                           max_iter=400,
#                           verbose=True)

# mlp_model = MLPClassifier(
#     max_iter=50000
# )

# parameter_space = {
#     'hidden_layer_sizes': [2, 5, 10, 20, 50, 63, 70, 100, (5, 5), (10, 10), (25, 25), (63, 63), (100, 100)],
#     'activation': ['tanh', 'logistic'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.00001, 0.0001, 0.001, 0.05],
#     'learning_rate': ['constant', 'adaptive'],
#     'learning_rate_init': [0.001, 0.05, 0.1, 0.25, 0.5, 0.6, 0.75, 0.8, 0.9, 0.99],
# }

# clf = GridSearchCV(mlp_model, parameter_space, n_jobs=-1, cv=7)
# clf.fit(train_df_x, targets_y)
#
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


mlp_model = MLPClassifier(hidden_layer_sizes=63,
                          max_iter=50000,
                          activation='tanh',
                          solver='sgd',
                          learning_rate='adaptive',
                          learning_rate_init=0.6,
                          shuffle=True,
                          verbose=True)

mlp_model.fit(train_df_x, targets_y)

predictions_proba = mlp_model.predict_proba(train_df_test)
predictions = mlp_model.predict(train_df_test)

accuracy = accuracy_score(targets_test, predictions)

##matriz de confusao
print(f'matriz de confusao:\n{confusion_matrix(targets_test.argmax(axis=1), predictions.argmax(axis=1))}')
#
# ##classificador
# print(f'classificador:\n{classification_report(targets_test.argmax(axis=1),predictions.argmax(axis=1))}')
#
# pesos = mlp_model.coefs_ # são os pesos

##é uma lista de matrizes de peso, em que a matriz de peso no índice i representa os pesos entre a camada i e a camada i + 1.
print(f'pesos:\n{mlp_model.coefs_}')
#
# é uma lista de vetores de bias, em que o vetor no índice i representa os valores de bias adicionados à camada i + 1.
# print(f'bias:\n{mlp_model.intercepts_}')

# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(train_df, targets[0])
# # targets deve ter 1 dimensão
# print(clf.predict(train_df))
# print(clf.predict_proba)

