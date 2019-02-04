import numpy as np


def generate_sample_sequence(max_length=100, action_delay=4, num_triggers=10, seed=42):

    randomizer = np.random.RandomState(seed=seed)

    x_series = np.zeros(shape=(max_length,), dtype=int)
    y_series = np.zeros(shape=(max_length,), dtype=int)

    trigger_positions = randomizer.choice(max_length, size=(num_triggers))

    x_series[trigger_positions] = 1

    trigger_effect_positions = trigger_positions + action_delay
    trigger_effect_positions = trigger_effect_positions[trigger_effect_positions<=max_length]
    y_series[trigger_effect_positions] = 1

    return x_series, y_series


def generate_training_data(N, max_length=100, seed=42):

    X = np.zeros((N, max_length, 1))
    Y = np.zeros((N, max_length, 1))

    for i in range(N):
        x_series, y_series = generate_sample_sequence(max_length=max_length, seed=seed, num_triggers=max_length//10)
        X[i, ...] = x_series.reshape((-1, 1))
        Y[i, ...] = y_series.reshape((-1, 1))

    return X, Y

if __name__ == "__main__":

    x_series, y_series = generate_sample_sequence(max_length=30, seed=41, num_triggers=5)

    print("Input")
    print(x_series)

    print("Output")
    print(y_series)

    X, Y = generate_training_data(100)

    print("Training data:")
    print("Input")
    print(X[10, ...].T)

    print("Output")
    print(Y[10, ...].T)