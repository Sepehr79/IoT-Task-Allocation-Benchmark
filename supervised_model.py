import tensorflow as tf
from tensorflow import keras

from dotenv import load_dotenv
import os

load_dotenv()
NUM_EDGES = int(os.getenv("NUM_EDGES"))

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=[NUM_EDGES+2+2*NUM_EDGES])) # [edges queues, task data, task capacity, user datarates, user last server]
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(NUM_EDGES))

model.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))

model.summary()