import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization


full_dataset = np.genfromtxt("sample_wp1_full.csv", delimiter = ",", skip_header = 1)

dataset = full_dataset[0:10000][:]
#traning set is 80% of dataset
train_N = int(0.8*round(len(dataset)))

X_train = dataset[0:train_N, 0:11]
Y_train = dataset[0:train_N, 11]
X_test = dataset[train_N:len(dataset), 0:11]
Y_test = dataset[train_N:len(dataset), 11]

#set validation if k-fold cross validation
validation = True
if validation:
    kf = KFold(5)
    val_y = []
    val_pred = []
    val_score = []
    fold = 0

    for train, val in kf.split(X_train):
        fold+=1
        print("Fold #{}".format(fold))

        x_k_train = X_train[train]
        y_k_train = Y_train[train]
        x_k_test = X_train[val]
        y_k_test = Y_train[val]

        model = Sequential()
        model.add(BatchNormalization(epsilon=0.001))
        model.add(Dense(40, input_dim=len(x_k_train[0]), kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        #model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        monitor = EarlyStopping(monitor = 'val_loss', min_delta=1e-8, patience=15, verbose=1, mode='auto')
        #train on the rest of the training set
        val_history = model.fit(x_k_train,y_k_train,validation_data=(x_k_test,y_k_test),callbacks=[monitor],verbose=0,epochs=200)

        plt.plot(val_history.history['loss'])

        pred = model.predict(x_k_test)
        val_y.append(y_k_test)
        val_pred.append(pred)

        val_score.append(metrics.mean_squared_error(pred,y_k_test))
        print("Fold score (MSE): {}".format(val_score[fold-1]))


#train on the whole training set
history = model.fit(X_train,Y_train,epochs=50,batch_size=30)

print(np.mean(val_score))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Mean squared error')
plt.xlabel('epoch')
plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'train'], loc='upper right')
plt.show()


pred = model.predict(X_test)
score = metrics.mean_squared_error(pred,Y_test)
print("Fold score (MSE): {}".format(score))


plt.plot(pred)
plt.plot(Y_test)
plt.title('Real vs predicted value')
plt.legend(['Real', 'Prediction'], loc='upper left')
plt.show()
