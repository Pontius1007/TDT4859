import numpy
from keras.models import Sequential
from keras.layers import Dense

# Taken from https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Load the dataset
dataset = numpy.loadtxt("sample_wp1.csv", delimiter=",", skiprows=1)

# Split it into target value pairs
features = dataset[:, 0:11]
target = dataset[:, 11]

# Create the model
model = Sequential()
model.add(Dense(len(features[0]), input_dim=len(features[0]), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
# Binary cross may have to be changed
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mae'])

# Fit the model aka do some training
model.fit(features, target, epochs=10, batch_size=15)

# Evaluate the model
scores = model.evaluate(features, target, verbose=1)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Try some predictions
predictions = model.predict(features[:10])

for i in range(len(predictions)):
    print("Target: " + str(target[i]) + " Guess: " + str(predictions[i]))
