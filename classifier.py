from keras.wrappers.scikit_learn import KerasClassifier
from models import classifier_model
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
from utils import abs_path, check_folder
import tensorflow as tf


class Classifier:
    def __init__(self, file):
        self._read_file(file)

    def _read_file(self, file):
        # Read csv
        ctcs = pd.read_csv(file)
        entries = ctcs.iloc[:, 1:255].values
        results = ctcs.iloc[:, 255].values

        # Split into test and training
        X_train, X_test, self.y_train, self.y_test = train_test_split(
            entries, results, test_size=0.2, random_state=0)

        # Normalize data
        sc = StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)

    def validation(self, cv=10, batch_size=-1, epochs=-1):
        classifier = KerasClassifier(build_fn=classifier_model)

        parameters = {'batch_size': batch_size,
                      'epochs': epochs,
                      'optimizer': ['adam'],
                      'activation': ['relu'],
                      'activationOutput': ['sigmoid']}

        metrics = ['accuracy', 'roc_auc', 'precision', 'recall']

        grid_search = GridSearchCV(estimator=classifier,
                                   verbose=2,
                                   param_grid=parameters,
                                   n_jobs=None,
                                   scoring=metrics,
                                   refit='precision',
                                   return_train_score=False,
                                   cv=cv)

        grid_search = grid_search.fit(self.X_train, self.y_train)

    def fit(self, logs_folder, save_dir=None, batch_size=32, epochs=250):
        date_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        log_dir = logs_folder + date_time
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        model = classifier_model('adam', 'relu', 'sigmoid')
        history = model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=1, use_multiprocessing=True,
                            validation_data=(self.X_test, self.y_test), callbacks=[tensorboard_callback])
        if save_dir is not None:
            self._save_model(save_dir, model, date_time)

    def _save_model(self, save_dir, model, date_time):
        check_folder(save_dir)
        model.save(save_dir + 'save_' + date_time + '.h5')
