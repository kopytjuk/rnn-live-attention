from keras.models import Sequential
from keras.layers import Dense, Activation, GRU, Input, TimeDistributed
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from data import generate_training_data
from attention import CustomAttentionLayer

import numpy as np


EPOCHS = 100
T_seq = 100

X_train, Y_train = generate_training_data(N=1000, max_length=T_seq, seed=42)

X_val, Y_val = X_train[700:], Y_train[700:]

X_test, Y_test = generate_training_data(N=100, max_length=T_seq, seed=4242)


# Define model
x_tensor = Input(shape=(T_seq, 1))
print(x_tensor)
inter_tensor = GRU(5, return_sequences=True)(x_tensor)
print(inter_tensor)
#y_tensor = TimeDistributed(Dense(1, activation="sigmoid"))(inter_tensor)
y_tensor = CustomAttentionLayer(1, "sigmoid", 5, "dot")(inter_tensor)
print("y_tensor", y_tensor)


def tpr(y_true, y_pred):
    # round to integers
    y_pred_rounded = K.round(y_pred)
    y_true = K.round(y_true)

    y_pred_rounded = K.reshape(y_pred_rounded, shape=(-1, T_seq))
    y_true = K.reshape(y_true, shape=(-1, T_seq))

    T = K.sum(y_true, axis=1)
    TP = K.sum(y_pred_rounded*y_true, axis=1)
    return K.mean(TP/T)

def tnr(y_true, y_pred):
    # round to integers
    y_pred_rounded = K.round(y_pred)
    y_true = K.round(y_true)

    y_pred_rounded = y_pred >= 1
    y_true = y_true >= 1

    y_pred_rounded = K.reshape(y_pred_rounded, shape=(-1, T_seq))
    y_true = K.reshape(y_true, shape=(-1, T_seq))

    y_pred_rounded = tf.math.logical_not(y_pred_rounded)
    y_true = tf.math.logical_not(y_true)

    y_true = tf.cast(y_true, "uint8")
    y_pred_rounded = tf.cast(y_pred_rounded, "uint8")

    N = K.sum(y_true, axis=1)
    TN = K.sum(y_pred_rounded*y_true, axis=1)
    return K.mean(TN/N)

model = Model(inputs=x_tensor, outputs=y_tensor)
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
