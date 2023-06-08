import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN
from tensorflow.keras.initializers import Constant,GlorotUniform,HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras import regularizers




class MyModel_Linear:
    def __init__(self):
        self.model = None
        self.model_poly = None
        self.poly = None
        self.NonLinear_model = None
        self.initial_weights = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.history_model_1=None
        self.history_model_2=None
        self.history_model_3=None

    def load_data(self, X, y, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self):
        self.logistic_model = LogisticRegression(max_iter=1000)
        self.logistic_model.fit(self.x_train, self.y_train)
        self.extract_weights()


    def extract_weights(self):
        self.initial_weights = [self.logistic_model.intercept_.reshape(-1, 1), self.logistic_model.coef_.T]


    def train_Hybrid(self):
        print("______________________________________________________")
        print("-----------------PROPOSED MODEL START-----------------")
        #self.get_initial_weights()
        self.NonLinear_model = Sequential()
        self.NonLinear_model.add(Dense(1, activation='sigmoid', input_dim=self.x_train.shape[1],
                                        kernel_initializer=Constant(self.initial_weights[1]),
                                        bias_initializer=Constant(self.initial_weights[0]),
                                        kernel_regularizer=regularizers.l2(0.001)))
        self.NonLinear_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        self.history_model_1=self.NonLinear_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50,verbose=0)
        
        # Print classification report
        y_pred = self.NonLinear_model.predict(self.x_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        print(classification_report(self.y_test, y_pred))
        # Print accuracy
        _, accuracy = self.NonLinear_model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Accuracy: %.2f%%' % (accuracy * 100))

        # Print minimum loss and epoch
        min_loss = min(self.history_model_1.history['val_loss'])
        min_loss_epoch = self.history_model_1.history['val_loss'].index(min_loss) + 1
        print('Minimum validation loss:', min_loss, 'at epoch', min_loss_epoch)
      

        print("----------------PROPOSED MODEL END--------------------")
        print("______________________________________________________")



    def train_Xavier(self):
        print("______________________________________________________")
        print("-----------------XAVIER MODEL START-------------------")
        #self.get_initial_weights()
        self.NonLinear_model = Sequential()
        self.NonLinear_model.add(Dense(1, activation='tanh', input_dim=self.x_train.shape[1],
                                        kernel_initializer=GlorotUniform(),
                                        bias_initializer='zeros'))
        self.NonLinear_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        self.history_model_2= self.NonLinear_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50,verbose=0)
          # Print classification report
        y_pred = self.NonLinear_model.predict(self.x_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        print(classification_report(self.y_test, y_pred))

        # Print accuracy
        _, accuracy = self.NonLinear_model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Accuracy: %.2f%%' % (accuracy * 100))

        # Print minimum loss and epoch
        min_loss = min(self.history_model_2.history['val_loss'])
        min_loss_epoch = self.history_model_2.history['val_loss'].index(min_loss) + 1
        print('Minimum validation loss:', min_loss, 'at epoch', min_loss_epoch)
        
       
        print("-------------------XAVIER MODEL END-------------------")
        print("______________________________________________________")


    def train_He(self):
        print("______________________________________________________")
        print("-----------------He MODEL START-------------------")
        #self.get_initial_weights()
        self.NonLinear_model = Sequential()
        self.NonLinear_model.add(Dense(1, activation='relu', input_dim=self.x_train.shape[1],
                                        kernel_initializer=HeUniform(),
                                        bias_initializer='zeros'))
        self.NonLinear_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        self.history_model_3= self.NonLinear_model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=50,verbose=0)
          # Print classification report
        y_pred = self.NonLinear_model.predict(self.x_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        print(classification_report(self.y_test, y_pred))

        # Print accuracy
        _, accuracy = self.NonLinear_model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Accuracy: %.2f%%' % (accuracy * 100))

        # Print minimum loss and epoch
        min_loss = min(self.history_model_3.history['val_loss'])
        min_loss_epoch = self.history_model_3.history['val_loss'].index(min_loss) + 1
        print('Minimum validation loss:', min_loss, 'at epoch', min_loss_epoch)
        
      
        print("-------------------He MODEL END-------------------")
        print("______________________________________________________")


    def PlotLoss(self):
        # Plot validation loss performance of both models
        plt.plot(self.history_model_1.history['val_loss'], label='Hybrid  Validation')
        plt.plot(self.history_model_2.history['val_loss'], label='Xavier Validation')
        plt.plot(self.history_model_3.history['val_loss'], label='He Validation')
        plt.title('Model Validation Loss')
        plt.ylabel('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylim(0, 7)  # set y-axis limits
        plt.legend(loc='upper right')

        # Save the plot as a high-resolution image
        plt.savefig('model_loss_comparison_make_moons.png', dpi=300)

        # Display the plot
        plt.show()
    
    def PlotAccuracy(self):
        # Plot validation accuracy performance of both models
        plt.plot(self.history_model_1.history['val_accuracy'], label='Hybrid Validation')
        plt.plot(self.history_model_2.history['val_accuracy'], label='Xavier Validation')
        plt.plot(self.history_model_3.history['val_accuracy'], label='He Validation')

        plt.title('Model Validation Accuracy')
        plt.ylabel('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)  # set y-axis limits
        plt.legend(loc='lower right')

        # Save the plot as a high-resolution image
        #plt.savefig('model_accuracy_comparison_make_moons.png', dpi=300)

        # Display the plot
        plt.show()
