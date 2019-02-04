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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", required=True, dest="experiment_dir", type=str, help="Directory to save the directory")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=100, help="Max number of epochs to train")
    parser.add_argument("-l", "--length", dest="length", type=int, default=100, help="Sequence length of timeseries")
    parser.add_argument("-p", "--patience", dest="patience", type=int, default=20, help="Number of epochs to continue training after no improvement.")

    args = parser.parse_args()
    EPOCHS = args.epochs
    PATIENCE = args.patience
    T_seq = args.length

    experiment_directory = args.experiment_dir
    log_directory = os.path.join(experiment_directory, "logs")
    model_directory = os.path.join(experiment_directory, "models")

    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory, exist_ok=True)

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)

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
    tb_callback = TensorBoard(log_dir=log_directory, histogram_freq=1)
    es_callback = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    # keras settings
    model.compile(optimizer='adam', sample_weight_mode="temporal",
                loss='binary_crossentropy',
                metrics=['accuracy', tpr, tnr])
    model.fit(X_train, Y_train, batch_size=64, epochs=EPOCHS, validation_data=(X_val, Y_val))  # starts training

    best_model_path = os.path.join(model_directory, "best_model.mdl")
    model.save(best_model_path)


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
