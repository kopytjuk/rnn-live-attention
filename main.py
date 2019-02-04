import argparse
import os

from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense, Activation, GRU, Input, TimeDistributed
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import numpy as np

from dummydata import generate_training_data
from attention import CustomAttentionLayer
from utils import tpr, tnr


EPOCHS = 100
T_seq = 100

if not os.path.exists(experiment_directory):
    os.makedirs(experiment_directory, exist_ok=True)

# Generate training data
X_train, Y_train = generate_training_data(N=1000, max_length=T_seq, seed=42)

X_val, Y_val = X_train[700:], Y_train[700:]
X_test, Y_test = generate_training_data(N=100, max_length=T_seq, seed=4242)


# Define model
with tf.device('/cpu:0'):
    x_tensor = Input(shape=(T_seq, 1))
    inter_tensor = GRU(5, return_sequences=True)(x_tensor)
    y_tensor = CustomAttentionLayer(1, "sigmoid", 5, "dot")(inter_tensor)
    model = Model(inputs=x_tensor, outputs=y_tensor)

# Define callbacks
tb_callback = TensorBoard()

# keras settings
model.compile(optimizer='adam', sample_weight_mode="temporal",
              loss='binary_crossentropy',
              metrics=['accuracy', tpr, tnr])
model.fit(X_train, Y_train, batch_size=64, epochs=EPOCHS, validation_data=(X_val, Y_val))  # starts training


# print some predictions
for i in range(2):

    x_sample = X_test[i, ...]
    x_sample_batch = x_sample.reshape((-1, T_seq, 1))
    y_predicted = model.predict(x_sample_batch)[0, ...]

    print("Prediction %d:"%i)
    print("Input")
    print(x_sample.T)

    print("Predicted")
    print(np.round(y_predicted.T))

    print("GT")
    print(Y_test[i, ...].T)
