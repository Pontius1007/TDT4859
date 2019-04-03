import pandas
from keras.models import model_from_json
from MTTS_DFFN import create_model
from Settings import *
from matplotlib import pyplot as plt


def save_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    print("Saved model as: " + filename + ".json")
    # serialize weights to HDF5
    weightname = filename + "weights.h5"
    model.save_weights(weightname)
    print("Saved weight as: " + weightname)


def load_model(filename):
    # load json and create model
    json_file = open(filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weightname = filename + "weights.h5"
    loaded_model.load_weights(weightname)
    print("Loaded model: " + filename)
    return loaded_model


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
    plt.xlabel('epoch')
    plt.legend(['MSE'], loc='upper left')
    plt.show()

    return model


def load_models_and_predict(wind_parks):
    number_of_last_predictions = 500
    models = []
    for wind_park in range(7):
        models.append(load_model("Models/" + str(wind_park+1)))

    wp7_features, wp7_targets = load_dataset(wind_parks[-1], True)
    predictions = []
    for model in models:
        predictions.append(model.predict(wp7_features[-number_of_last_predictions:]))

    for prediction in predictions:
        plt.plot(prediction)
    plt.title("Predictions for all 6 windparks on data from windpark 7")
    plt.xlabel('epoch')
    plt.legend(['1', '2', '3', '4', '5', '6', '7'], loc='upper left')

    plt.plot(wp7_targets[-number_of_last_predictions:])
    plt.show()


def main():
    do_training = True
    wind_parks = ["Processed_data/wp1.csv", "Processed_data/wp2.csv", "Processed_data/wp3.csv",
                  "Processed_data/wp4.csv", "Processed_data/wp5.csv", "Processed_data/wp6.csv",
                  "Processed_data/wp7.csv"]
    while do_training:
        user_input = (input("Enter wind park ID you want to load and train: "))
        if user_input == '1':
            file_name = wind_parks[0]
        elif user_input == '2':
            file_name = wind_parks[1]
        elif user_input == '3':
            file_name = wind_parks[2]
        elif user_input == '4':
            file_name = wind_parks[3]
        elif user_input == '5':
            file_name = wind_parks[4]
        elif user_input == '6':
            file_name = wind_parks[5]
        elif user_input == '7':
            file_name = wind_parks[6]
        elif user_input == 'all':
            for i in range(7):
                file_name = wind_parks[i]
                train_features, train_targets, test_features, test_targets = load_dataset(file_name, False)
                model = train_model(train_features, train_targets, test_features, test_targets, file_name)
                save_model(model, "Models/" + str(i + 1))
        elif user_input == "load":
            load_models_and_predict(wind_parks)
            do_training = False
        else:
            file_name = ""
            print("Wind park not recognised. Quitting.")
            do_training = False

        if do_training:
            train_features, train_targets, test_features, test_targets = load_dataset(file_name, False)
            model = train_model(train_features, train_targets, test_features, test_targets, file_name)
            save_model(model, "Models/" + str(user_input))


if __name__ == '__main__':
    main()
