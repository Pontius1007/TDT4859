import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib.dates as mdates


def str_to_datetime(date):
    return dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


def main():
    df = pd.read_csv('Processed_data/wp7.csv', names=['dates', 'ws-2', 'ws-1', 'ws', 'ws+1', 'wd-2',
                                                      'wd-1', 'wd', 'wd+1', 'hour_from_06', 'week', 'mounth',
                                                      'production'], sep=',', skiprows=1)

    real_production = df.production[48:]
    real_production.index = np.arange(len(real_production))
    M = len(real_production)
    predicted_production = df.production[:M]

    df['dates'] = df['dates'].apply(str_to_datetime)

    xDates = df.dates.iloc[0:M]

    realFrame = pd.DataFrame({"Dates": xDates, "Real": real_production})
    predFrame = pd.DataFrame({"Dates": xDates, "Prediction": predicted_production})

    print("MSE: ", mean_squared_error(realFrame.Real, predFrame.Prediction))

    fig, ax = plt.subplots()

    ax.plot(realFrame["Dates"], realFrame["Real"])
    ax.plot(predFrame["Dates"], predFrame["Prediction"])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))

    plt.show()


if __name__ == '__main__':
    main()
