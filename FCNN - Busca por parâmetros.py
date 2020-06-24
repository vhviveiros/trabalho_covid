# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import tensorflow
from keras.utils import np_utils
from tensorflow.keras.metrics import AUC, Precision, Recall
import scipy

scipy.test()


dataset = pd.read_csv('characteristics.csv')
X = dataset.iloc[:, 0:255].values
y = dataset.iloc[:, 255].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = np_utils.to_categorical(y)


def build_classifier(optimizer, activation, activationOutput):
    classifier = Sequential()
    classifier.add(Dense(units=200, activation=activation, input_dim=255))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=200, activation=activation))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=200, activation=activation))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=200, activation=activation))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=200, activation=activation))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=200, activation=activation))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(units=1, activation=activationOutput))
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [32, 16, 24],
              'epochs': [100, 250, 200, 500],
              'optimizer': ['adam'],
              'activation': ['relu'],
              'activationOutput': ['sigmoid']}

metricas = ['accuracy', 'roc_auc', 'precision', 'recall']

grid_search = GridSearchCV(estimator=classifier,
                           verbose=2,
                           param_grid=parameters,
                           n_jobs=None,
                           scoring=metricas,
                           refit='precision',
                           return_train_score=False,
                           cv=2)

grid_search = grid_search.fit(X_train, y_train)

#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
# %%


def print_results(results):
    list_view = list(filter(lambda x: str.startswith(x, 'split'), results))
    for key in list_view:
        print("\n\n%s:\n\n" % (key))
        print(grid_search.cv_results_[key])


print_results(grid_search.cv_results_)


# %%
