import pandas
from keras.models import load_model, save_model
from MTTS_DFFN import create_model
from Settings import *
from matplotlib import pyplot as plt
from sklearn import metrics
from termcolor import cprint
from pyfiglet import figlet_format
from colorama import init
import sys
from os import listdir
from os.path import isfile, join

init(strip=not sys.stdout.isatty())  # strip colors if stdout is redirected


def load_dataset(file_name, wp7):
    cols = list(pandas.read_csv(file_name, nrows=1))
    series = pandas.read_csv(file_name, header=0, usecols=[i for i in cols if i != 'date'])
    # Convert to numpy array
    data = series.to_numpy()

    # Plotting lists
    if wp7:
        features = data[:, 0:11]
        targets = data[:, 11]
        return features, targets
    else:
        training_length = int(len(data) * Settings.training_size)
        training = data[:training_length]
        test = data[training_length:]

        # Getting features and targets
        train_features = training[:, 0:11]
        train_targets = training[:, 11]
        test_features = test[:, 0:11]
        test_targets = test[:, 11]
        return train_features, train_targets, test_features, test_targets


def train_model(x_features, x_targets, y_features, y_targets, file_name):
    model = create_model()
    history = model.fit(x=x_features, y=x_targets, validation_data=(y_features, y_targets),
                        verbose=Settings.verbose,
                        epochs=Settings.epochs)

    # summarize history for accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.title('model MSE for windpark ' + file_name)
    plt.xlabel('Hours')
    plt.legend(['MSE'], loc='upper left')
    plt.show()

    return model


def load_models_and_predict(wind_parks):
    models = []
    for wind_park in range(6):
        models.append(load_model("Models/" + str(wind_park + 1) + ".h5"))

    wp7_features, wp7_targets = load_dataset(wind_parks[-1], True)
    wp7_features = wp7_features[-Settings.wp7_predictions:]
    wp7_targets = wp7_targets[-Settings.wp7_predictions:]
    predictions = []
    for model in models:
        predictions.append(model.predict(wp7_features))

    mses = []
    for prediction in predictions:
        mses.append(metrics.mean_squared_error(wp7_targets, prediction))
        plt.plot(prediction)
    plt.plot(wp7_targets)
    plt.title("Predictions for all 6 windparks on data from windpark 7")
    plt.xlabel('epoch')
    plt.legend(['1', '2', '3', '4', '5', '6', '7 - Target'], loc='upper left')

    plt.show()

    print(mses)


def transfer_learning_training(base_model_name, n_epochs, months_of_data):
    # One months of data translates to about 750 rows in our dataset
    rows_of_data = 750 if (months_of_data == 1) else 2250
    # baseline_model = load_model("Models/" + base_model_name)
    baseline_model = load_model("Models/" + base_model_name + ".h5")
    train_features, train_targets = load_dataset("Processed_data/wp7.csv", True)
    train_features, train_targets = train_features[-rows_of_data:], train_targets[-rows_of_data:]
    # Training
    history = baseline_model.fit(x=train_features, y=train_targets, verbose=Settings.verbose, epochs=n_epochs,
                                 validation_split=0.1)

    # Saving the model
    print("Saving the model...")
    save_model(baseline_model,
               "Models/model" + base_model_name + "epochs" + str(n_epochs) + "months" + str(months_of_data) + ".h5")
    print("Saved as: " + "model" + base_model_name + "epochs" + str(n_epochs) + "months" + str(months_of_data) + ".h5")

    # summarize history for accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.title(
        'MSE for windpark 6 after ' + str(n_epochs) + " epochs " + "and " + str(months_of_data) + " month(s) of data")
    plt.xlabel('epoch')
    plt.legend(['MSE'], loc='upper left')
    plt.show()


def train(file_name, user_input):
    train_features, train_targets, test_features, test_targets = load_dataset(file_name, False)
    model = train_model(train_features, train_targets, test_features, test_targets, file_name)
    save_model(model, "Models/" + str(user_input) + ".h5")


def load_and_predict_transfer():
    models = []
    onlyfiles = [f for f in listdir("Models/Transfer") if isfile(join("Models/Transfer", f))]

    for model_name in onlyfiles:
        models.append(load_model("Models/Transfer/" + model_name))
    models.append(load_model("Models/7.h5"))
    onlyfiles.append("7 - Target")

    wp7_features, wp7_targets = load_dataset("Processed_data/wp7.csv", True)
    # Predictions on the last 500 datapoints which the model has not been trained on
    wp7_features, wp7_targets = wp7_features[15650:-2250], wp7_targets[15650:-2250]
    predictions = []
    for model in models:
        predictions.append(model.predict(wp7_features))
    mses = []
    for prediction in predictions:
        mses.append(metrics.mean_squared_error(wp7_targets, prediction))
        plt.plot(prediction)
    plt.plot(wp7_targets)
    plt.title("Predictions for all transfer wind parks trained on new data from Park 7")
    plt.xlabel('Hours')
    plt.legend(onlyfiles, loc='upper left')

    plt.show()

    print(mses)


def main():
    run = True
    wind_parks = ["Processed_data/wp1.csv", "Processed_data/wp2.csv", "Processed_data/wp3.csv",
                  "Processed_data/wp4.csv", "Processed_data/wp5.csv", "Processed_data/wp6.csv",
                  "Processed_data/wp7.csv"]
    cprint(figlet_format('AI is the future', font='doom'), 'red', attrs=['bold'])
    while run:
        print("Commands: Number 1-7 to train on a given park")
        print("'all' to train on all datasets. 'transfer' to do some transfer learning")
        print("and 'load' to load model 1-6 and predict on park 7. 'pt' for predict transfer")
        user_input = (input("Enter the command you want to run: "))
        if user_input == '1':
            file_name = wind_parks[0]
            train(file_name, user_input)
        elif user_input == '2':
            file_name = wind_parks[1]
            train(file_name, user_input)
        elif user_input == '3':
            file_name = wind_parks[2]
            train(file_name, user_input)
        elif user_input == '4':
            file_name = wind_parks[3]
            train(file_name, user_input)
        elif user_input == '5':
            file_name = wind_parks[4]
            train(file_name, user_input)
        elif user_input == '6':
            file_name = wind_parks[5]
            train(file_name, user_input)
        elif user_input == '7':
            file_name = wind_parks[6]
            train(file_name, user_input)
        elif user_input == "pt":
            load_and_predict_transfer()
        elif user_input == "transfer":
            epochs = int(input("Enter the number of training epochs: "))
            b_model = input("Enter the model you want as your baseline. 1-6: ")
            months = int(input("Enter the number of months for training data. 1 or 3: "))
            transfer_learning_training(b_model, epochs, months)
        elif user_input == 'all':
            for i in range(7):
                file_name = wind_parks[i]
                train_features, train_targets, test_features, test_targets = load_dataset(file_name, False)
                model = train_model(train_features, train_targets, test_features, test_targets, file_name)
                save_model(model, "Models/" + str(i + 1) + ".h5")
        elif user_input == "load":
            load_models_and_predict(wind_parks)
        else:
            print("Operation not recognised. Quitting.")
            run = False


if __name__ == '__main__':
    main()
