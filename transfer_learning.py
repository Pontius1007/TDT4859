import os
import numpy as np
from keras.models import model_from_json
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Dropout
from Settings import *


def save_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    weightname = filename + "weights.h5"
    model.save_weights(weightname)
    print("Saved model to disk")


def load_model(filename):
    # load json and create model
    json_file = open(filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weightname = filename + "weights.h5"
    loaded_model.load_weights(weightname)
    print("Loaded model from disk")
    return loaded_model

def create_model(x_train, y_train):


def load_dataset(file_name):
    cols = list(pandas.read_csv(file_name, nrows=1))
    series = pandas.read_csv(file_name, header=0, usecols=[i for i in cols if i != 'date'])
    # Convert to numpy array
    data = series.to_numpy()

    # Plotting lists
    training_length = len(data)*Settings.training_size
    training = data[train_index]
    test = data[test_index]

    # Getting features and targets
    train_features = training[:, 0:11]
    train_targets = training[:, 11]
    test_features = test[:, 0:11]
    test_targets = test[:, 11]
    return train_features, train_targets, test_features, test_targets

def main():

    file_name = "Processed_data/wp1.csv"

    train_features, train_targets, test_features, test_targets = load_dataset(file_name)


    select_windpark()

    create_model()

    load_model()

    save_model()

main()
