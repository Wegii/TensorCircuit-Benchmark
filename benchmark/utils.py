import numpy as np
import tensorflow as tf


def filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y

def load_mnist():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0

    x_train, y_train = filter_pair(x_train, y_train, 1, 5)
    x_train_small = tf.image.resize(x_train, (3, 3)).numpy()
    x_train_bin = np.array(x_train_small > 0.5, dtype=np.float32)
    x_train_bin = np.squeeze(x_train_bin)[:100]

    x_train_tf = tf.reshape(tf.constant(x_train_bin, dtype=tf.float64), [-1, 9])
    y_train_tf = tf.constant(y_train[:100], dtype=tf.float64)

    mnist_data = (
        tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
        .repeat(200)
        .shuffle(100)
        .batch(32)
    )

    return mnist_data