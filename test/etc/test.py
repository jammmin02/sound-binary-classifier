import numpy as np
y = np.load("outputs/cnn_lstm/y_lstm.npy")
print("Label distribution:", np.bincount(y))
