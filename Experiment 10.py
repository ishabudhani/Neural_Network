import tensorflow as tf
from tensorflow.keras import layers, models

# Simple RNN Model
model_rnn = models.Sequential([
    # Input shape: (timesteps, features)
    layers.SimpleRNN(64, input_shape=(10, 1), activation='tanh'),
    layers.Dense(1)
])

model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.summary()