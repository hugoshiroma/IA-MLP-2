from sklearn.neural_network import MLPClassifier
from sklearn import svm
import pandas as pd
import numpy as np

train_df = pd.read_csv('../inputs/Part-1/caracteres-limpos.csv', header=None)
train_df = train_df.drop(labels=63, axis=1)
targets = [
    [1, -1, -1, -1, -1, -1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, -1, -1, -1, -1, -1, 1],
    [1, -1, -1, -1, -1, -1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, -1, -1, -1, -1, -1, 1],
    [1, -1, -1, -1, -1, -1, -1],
    [-1, 1, -1, -1, -1, -1, -1],
    [-1, -1, 1, -1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1, 1, -1],
    [-1, -1, -1, -1, -1, -1, 1]
]
targets = np.squeeze(targets)
# o problema sepa ta no target

mlp_model = MLPClassifier(solver='sgd',
                          learning_rate='constant',
                          learning_rate_init=0.5,
                          hidden_layer_sizes=(30,),
                          random_state=1,
                          max_iter=400,
                          verbose=True)

mlp_model.fit(train_df, targets)
mlp_model.predict(train_df)
pesos = mlp_model.coefs_ # são os pesos


# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(train_df, targets[0])
# # targets deve ter 1 dimensão
# print(clf.predict(train_df))
# print(clf.predict_proba)

