"""Machine learning models."""
import tensorflow as tf
from tensorflow.keras import layers


def quantile_loss(y_true, y_pred, tau):
    """
    Loss function that assigns the loss
    as the mean of multiple quantile losses.

    See: https://en.wikipedia.org/wiki/Quantile_regression
    """
    e = y_pred - y_true
    return tf.keras.backend.mean(tf.keras.backend.maximum(tau * e, (tau - 1) * e))


def fractional_quantile_loss(y_true, y_pred, tau):
    """
    Loss function that assigns the loss
    as the mean of multiple quantile losses.

    See: https://en.wikipedia.org/wiki/Quantile_regression
    """
    e = y_pred - y_true
    return tf.keras.backend.mean(
        tf.keras.backend.maximum((tau * e) / y_true, ((tau - 1) * e) / y_true)
    )


def quantile_neural_network(normalizer, tau=0.5, hp=None):
    """Get the neural network model which calculates a quantile loss."""
    if hp is None:
        activation = "relu"
        units = 416
        depth = 3
        learning_rate = 1e-3
    else:
        # Tune the number of units
        # Choose an optimal value between 32-512
        activation = hp.Choice("activation", values=["relu", "elu"])
        units = hp.Int("units", min_value=32, max_value=512, step=32)
        depth = hp.Int("depth", min_value=1, max_value=5)
        learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

    dense_layers = []
    for _ in range(depth):
        dense_layers.append(layers.Dense(units=units, activation=activation))

    nn_model = tf.keras.Sequential(
        [
            normalizer,
        ]
        + dense_layers
        + [
            layers.Dense(units=1, activation="relu"),
        ]
    )

    nn_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=lambda y, y_p: fractional_quantile_loss(y, y_p, tau=tau),
    )

    return nn_model


def neural_network(normalizer, hp=None):
    """Get the neural network model."""
    if hp is None:
        activation = "relu"
        units = 32
        depth = 4
        learning_rate = 1e-3
        # scale = 1000
    else:
        # Tune the number of units
        # Choose an optimal value between 32-512
        activation = hp.Choice("activation", values=["relu", "elu"])
        units = hp.Int("units", min_value=32, max_value=512, step=32)
        depth = hp.Int("depth", min_value=1, max_value=5)
        learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])
        # scale = hp.Choice('scale', values=[10, 100, 1000, 10000])

    dense_layers = []
    for _ in range(depth):
        dense_layers.append(layers.Dense(units=units, activation=activation))

    nn_model = tf.keras.Sequential(
        [
            normalizer,
        ]
        + dense_layers
        + [
            layers.Dense(units=1, activation="relu"),
        ]
    )

    nn_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_absolute_percentage_error",
    )
    return nn_model
