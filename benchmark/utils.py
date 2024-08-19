import numpy as np
import tensorflow as tf

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'
    
def _filter_pair(x, y, a, b):
    keep = (y == a) | (y == b)
    x, y = x[keep], y[keep]
    y = y == a
    return x, y


def _load_mnist():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0

    x_train, y_train = _filter_pair(x_train, y_train, 1, 5)
    x_train_small = tf.image.resize(x_train, (3, 3)).numpy()
    x_train_bin = np.array(x_train_small > 0.5, dtype=np.float32)
    x_train_bin = np.squeeze(x_train_bin)

    x_train_tf = tf.reshape(tf.constant(x_train_bin, dtype=tf.float64), [-1, 9])
    y_train_tf = tf.constant(y_train, dtype=tf.float64)

    return x_train_tf, y_train_tf


def load_mnist_tf(batch_size: int = 32):
    x_train_tf, y_train_tf = _load_mnist()

    # Create tensorflow dataset
    mnist_data = (
        tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
        .batch(batch_size)
    )

    return mnist_data


def load_mnist_pt(batch_size: int = 32):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_train_tf, y_train_tf = _load_mnist()

    x = torch.tensor(x_train_tf.numpy())
    y = torch.tensor(y_train_tf.numpy())

    # Create pytorch dataset
    mnist_dataset = TensorDataset(x, y)
    train_dataloader = DataLoader(mnist_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    return train_dataloader