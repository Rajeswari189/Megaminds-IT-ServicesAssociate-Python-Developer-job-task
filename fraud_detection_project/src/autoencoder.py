# Keras autoencoder model
from tensorflow.keras import layers, models

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(8, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    model = models.Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer='adam', loss='mse')

    return model
