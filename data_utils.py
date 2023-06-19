"""Data loading utilities"""

# Some imports
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict
import pickle as pl
import tensorflow_datasets as tfds

def _one_hot(x, k, dtype = jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def _standardize(x):
  """Standardization per sample across feature dimension."""
  axes = tuple(range(1, len(x.shape)))
  mean = jnp.mean(x, axis = axes, keepdims = True)
  std_dev = jnp.std(x, axis = axes, keepdims = True)
  return (x - mean) / std_dev


def load_data(dataset, num_classes):

    #path = f'datasets/{dataset}.dump'    
    #in_file = open(path, 'rb')

    if dataset in ['cifar100', 'cifar10', 'mnist', 'fashion_mnist']:
        ds_train, ds_test = tfds.as_numpy(tfds.load(dataset, data_dir = './',split = ["train", "test"], batch_size = -1, as_dataset_kwargs = {"shuffle_files": False}))
    else:
        raise ValueError("Invalid dataset name.")

    x_train, y_train, x_test, y_test = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])

    x_train = jnp.asarray(x_train, dtype = jnp.float32)
    y_train = jnp.asarray(y_train, dtype = jnp.int32)

    x_test = jnp.asarray(x_test, dtype = jnp.float32)
    y_test = jnp.asarray(y_test, dtype = jnp.int32)

    #get info
    info = config_dict.ConfigDict()
    info.num_train = x_train.shape[0]
    info.num_test = x_test.shape[0]
    info.num_classes = num_classes
    info.in_dim = (1, *x_train[0].shape)

    #standardize input
    x_train, x_test = _standardize(x_train), _standardize(x_test)

    #get one hot encoding for the labels
    y_train = _one_hot(y_train, num_classes)
    y_test = _one_hot(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), info

def load_data_regress(dataset):
    """
    loads mnist dataset from tensorflow_datasets
    """

    path = f'datasets/{dataset}.dump'
    in_file = open(path, 'rb')
    (x_train, y_train), (x_test, y_test) = pl.load(in_file)

    #get info
    info = config_dict.ConfigDict()
    info.num_train = x_train.shape[0]
    info.num_test = x_test.shape[0]
    
    info.in_dim = x_train.shape[1:]
    info.out_dim = y_Train.shape[1:]

    return (x_train, y_train), (x_test, y_test), info

def data_stream(key, ds, batch_size):
    " Creates a data stream with a predifined batch size."
    train_images, train_labels = ds
    num_train = train_images.shape[0]
    num_batches = estimate_num_batches(num_train, batch_size)
    rng = np.random.RandomState(key)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1)*batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]

def estimate_num_batches(num_train, batch_size):
    "Estimates number of batches using dataset and batch size"
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches

