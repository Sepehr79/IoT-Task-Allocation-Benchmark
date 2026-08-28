import simulation
from config import *
import tensorflow as tf
from tensorflow import keras

delay_predictor = keras.models.Sequential()
delay_predictor.add(
    keras.layers.Input(shape=[NUM_EDGES+2+2*NUM_EDGES+1])
) 
# [
#  edge queues,
#  task input size,
#  task processing capcacity,
#  user's data rates,
#  user's last allocation,
#  workload
# ]
delay_predictor.add(keras.layers.Dense(64, activation="relu"))
delay_predictor.add(keras.layers.Dense(32, activation="relu"))
delay_predictor.add(keras.layers.Dense(16, activation="relu"))
delay_predictor.add(keras.layers.Dense(NUM_EDGES))

delay_predictor.compile(loss="mse", optimizer=keras.optimizers.Adam(1e-3))

delay_predictor.summary()
