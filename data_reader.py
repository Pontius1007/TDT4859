import numpy
from keras.models import Sequential
from keras.layers import Dense

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
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# Fit the model aka do some training
model.fit(features, target, epochs=100, batch_size=5)

# Evaluate the model
scores = model.evaluate(features, target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
