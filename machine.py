import pandas as pd

dataset = pd.read_csv('cancer.csv')

#delete the first column which is the desired result
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])

y = dataset["diagnosis(1=m, 0=b)"]

#machine learning library to split data
from sklearn.model_selection import train_test_split
#test size is 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Keras to build neural network
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #answer -> diagnosis

model.compile(optimizer='adam', loss='binary_crossentropy', metric=['accuracy'])

#1000 to go over data a lot of times
model.fit(x_train, y_train, epochs=1000)