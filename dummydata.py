import numpy as np


def generate_sample_sequence(max_length=100, action_delay=4, num_triggers=10, seed=42):

    randomizer = np.random.RandomState(seed=seed)

    x_series = np.zeros(shape=(max_length,), dtype=float)
    y_series = np.zeros(shape=(max_length,), dtype=float)

    trigger_positions = randomizer.choice(max_length, size=(num_triggers,))

    val = np.random.choice(np.arange(-1, 1.001, 0.01))

    x_series[trigger_positions] = val

    trigger_effect_positions = trigger_positions + action_delay
    trigger_effect_positions = trigger_effect_positions[trigger_effect_positions<=max_length]
    y_series[trigger_effect_positions] = val

    return x_series, y_series


def generate_training_data(N, max_length=100, seed=42):

    X = np.zeros((N, max_length, 1))
    Y = np.zeros((N, max_length, 1))

    for i in range(N):
        x_series, y_series = generate_sample_sequence(max_length=max_length, num_triggers=1)
        X[i, ...] = x_series.reshape((-1, 1))
        Y[i, ...] = y_series.reshape((-1, 1))

    return X, Y

if __name__ == "__main__":

    x_series, y_series = generate_sample_sequence(max_length=30, seed=2, num_triggers=1)

    print("Input")
    print(x_series)

    print("Output")
    print(y_series)

    X, Y = generate_training_data(100)

    print("Training data:")
    print("Input")
    print(X[12, ...].T)

    print("Output")
    print(Y[12, ...].T)