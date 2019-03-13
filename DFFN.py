# Taken from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load the dataset
dataset = numpy.loadtxt("sample_wp1.csv", delimiter=",", skiprows=1)

# Split it into target value pairs
features = dataset[:, 0:11]
target = dataset[:, 11]


# Create the model
def baseline_model():
    model = Sequential()
    model.add(Dense(len(features[0]), input_dim=len(features[0]), kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# use scikit-learn’s Pipeline framework to perform the standardization during the model evaluation process,
# within each fold of the cross validation.
# This ensures that there is no data leakage from each test set cross validation fold into the training data.
estimators = [('standardize', StandardScaler()),
              ('mlp', KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=20, verbose=1))]
pipeline = Pipeline(estimators)
# As a general rule and empirical evidence, K = 5 or 10 is generally preferred
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, features, target, cv=kfold)

# cross_val_predict returns an array of the same size as `target` where each entry
# is a prediction obtained by cross validation:
pred = cross_val_predict(pipeline, features, target, cv=kfold)

print("Standardized: %.4f (%.4f) MSE" % (results.mean(), results.std()))

