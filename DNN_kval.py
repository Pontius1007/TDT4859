import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from Settings import *

full_dataset = np.genfromtxt("sample_wp1_full.csv", delimiter=",", skip_header=1)

dataset = full_dataset[:12000][:]
# training set is 80% of dataset
train_N = int(Settings.training_size * round(len(dataset)))

X_train = dataset[0:train_N, 0:11]
Y_train = dataset[0:train_N, 11]
X_test = dataset[train_N:len(dataset), 0:11]
Y_test = dataset[train_N:len(dataset), 11]

# set validation if k-fold cross validation
if Settings.validate:
    kf = KFold(Settings.n_splits)
    val_y = []
    val_pred = []
    val_score = []
    fold = 0

    for train, val in kf.split(X_train):
        fold += 1
        print("Fold #{}".format(fold))

        x_k_train = X_train[train]
        y_k_train = Y_train[train]
        x_k_test = X_train[val]
        y_k_test = Y_train[val]

        model = Sequential()
        model.add(BatchNormalization(epsilon=0.001))
        model.add(
            Dense(Settings.input_layer, input_dim=len(x_k_train[0]), kernel_initializer='normal', activation='relu'))
        model.add(Dense(Settings.h_layer1, kernel_initializer='normal', activation='relu'))
        model.add(Dense(Settings.h_layer2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(Settings.output_layer))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # train on the rest of the training set
        val_history = model.fit(x_k_train, y_k_train, validation_data=(x_k_test, y_k_test),
                                verbose=Settings.verbose, epochs=Settings.epochs)

        plt.plot(val_history.history['loss'])

        pred = model.predict(x_k_test)
        val_y.append(y_k_test)
        val_pred.append(pred)

        val_score.append(metrics.mean_squared_error(pred, y_k_test))
        print("Fold score (MSE): {}".format(val_score[fold - 1]))

# train on the whole training set
# history = model.fit(X_train, Y_train, epochs=Settings.epochs, batch_size=Settings.batch_size)

print("This is the score you are supposed to check against: ")
print("Mean value score: ", np.mean(val_score))
# plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Mean squared error')
plt.xlabel('epoch')
plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'train'], loc='upper right')
plt.show()

pred = model.predict(X_test)
score = metrics.mean_squared_error(pred, Y_test)
print("Fold score (MSE): {}".format(score))

plt.plot(pred)
plt.plot(Y_test)
plt.title('Real vs predicted value')
plt.legend(['Real', 'Prediction'], loc='upper left')
plt.show()
