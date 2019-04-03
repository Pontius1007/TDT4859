# https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
import os

import pandas
from sklearn import metrics
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Dropout
from sklearn.model_selection import TimeSeriesSplit
from Settings import *
from matplotlib import pyplot
import numpy as np


def multiple_train_test_splits(file_name):
    # Load data using Pandas
    cols = list(pandas.read_csv(file_name, nrows=1))
    series = pandas.read_csv(file_name, header=0, usecols=[i for i in cols if i != 'date'])
    # Convert to numpy array
    data = series.to_numpy()

    splits = TimeSeriesSplit(n_splits=Settings.n_splits)
    f1 = pyplot.figure(1)

    x_test = []
    y_test = []

    # Plotting lists
    val_y = []
    val_pred = []
    val_score = []
    validation_hist = []
    index = 1

    for train_index, test_index in splits.split(data):
        training = data[train_index]
        test = data[test_index]

        # Getting features and targets
        x_features = training[:, 0:11]
        x_targets = training[:, 11]
        y_features = test[:, 0:11]
        y_targets = test[:, 11]

        print('Observations: %d' % (len(training) + len(test)))
        print('Training Observations: %d' % (len(training)))
        print('Testing Observations: %d' % (len(test)))
        # The first number must reflect number of splits
        ax1 = f1.add_subplot(Settings.n_splits, 1, 0 + index)
        ax1.plot([x[11] for x in training])
        ax1.plot([None for i in training] + [x[11] for x in test])

        # Does training
        model = create_model()
        # train on the rest of the training set
        val_history = model.fit(x=x_features, y=x_targets, validation_data=(y_features, y_targets),
                                verbose=Settings.verbose,
                                epochs=Settings.epochs)
        validation_hist.append(val_history.history['loss'])

        # Prediction and plotting
        pred = model.predict(y_features)
        val_y.append(y_targets)
        val_pred.append(pred)
        val_score.append(metrics.mean_squared_error(pred, y_targets))
        print("Train test score (MSE): {}".format(val_score[index - 1]))

        x_test = y_features
        y_test = y_targets

        index += 1

    pyplot.show()

    print("Mean value score for all train tests: ", np.mean(val_score))

    # Plotting loss
    plot_loss(validation_hist)

    # Predicting and plotting result
    pred = model.predict(x_test)
    score = metrics.mean_squared_error(pred, y_test)
    print("Train test score (MSE): {}".format(score))

    pyplot.plot(pred)
    pyplot.plot(y_test)
    pyplot.title('Predicted vs realvalue')
    pyplot.legend(['Prediction', 'Real'], loc='upper left')
    pyplot.show()


def plot_loss(validation_hist):
    pyplot.figure(1)
    for x in validation_hist:
        pyplot.plot(x)
    pyplot.title('Model loss')
    pyplot.ylabel('Mean squared error')
    pyplot.xlabel('epoch')
    pyplot.legend(['Train 1', 'Train 2', 'Train 3', 'Train 4', 'Train 5'], loc='upper right')
    pyplot.show()


def create_model():
    model = Sequential()

    # Input layer
    model.add(
        Dense(Settings.input_layer, input_dim=Settings.number_of_features, kernel_initializer='normal', use_bias=False))
    model.add(Activation('relu'))
    model.add(BatchNormalization(epsilon=0.001))

    # Hidden Layer 1
    model.add(Dense(Settings.h_layer1, kernel_initializer='normal'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(epsilon=0.001))

    # Hidden Layer 2
    model.add(Dense(Settings.h_layer2, kernel_initializer='normal'))
    model.add(Activation('relu'))
    # model.add(BatchNormalization(epsilon=0.001))

    # Output layer
    model.add(Dense(Settings.output_layer))
    model.add(BatchNormalization())

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    multiple_train_test_splits("Processed_data/wp1.csv")


if __name__ == '__main__':
    main()
