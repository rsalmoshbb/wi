#2222
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant,GlorotUniform
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


class LogisticToSequential:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.logistic_model = None
        self.sequential_model = None
        self.initial_weights = None
        self.history = None

    def train_logistic_regression(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.logistic_model = LogisticRegression(max_iter=1000)
        self.logistic_model.fit(X_train, y_train)
        self.extract_weights()

    def extract_weights(self):
        if self.logistic_model is None:
            raise ValueError("Logistic regression model not trained yet.")
        self.initial_weights = [self.logistic_model.intercept_.reshape(-1, 1), self.logistic_model.coef_.T]

    def create_sequential_model(self, output_units, activation='sigmoid', loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        if self.initial_weights is None:
            raise ValueError("Initial weights not extracted yet.")
        self.sequential_model = Sequential()
        self.sequential_model.add(Dense(output_units, activation=activation, input_dim=self.X.shape[1],
                                        kernel_initializer=Constant(self.initial_weights[1]),
                                        bias_initializer=Constant(self.initial_weights[0])))
        self.sequential_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    

    def create_sequential_modelXavier(self, output_units, activation='sigmoid', loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']):
        if self.initial_weights is None:
            raise ValueError("Initial weights not extracted yet.")
        
        self.sequential_model = Sequential()
        self.sequential_model.add(Dense(output_units, activation=activation, input_dim=self.X.shape[1],
                                        kernel_initializer=GlorotUniform(),
                                        bias_initializer='zeros'))
        
        self.sequential_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        


    
    def evaluate_sequential_model(self):
        if self.sequential_model is None:
            raise ValueError("Sequential model not created yet.")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.history = self.sequential_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        y_pred = (self.sequential_model.predict(X_test) > 0.5).astype(int).reshape(-1)
        cm = confusion_matrix(y_test, y_pred)
        return cm
    
    def plot_performance(self):
        if self.history is None:
            raise ValueError("Sequential model not trained yet.")
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        return plt.show()
