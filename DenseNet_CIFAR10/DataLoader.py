import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    files = os.listdir(data_dir)
    x_train = np.array([[]]).reshape(0,3072)
    y_train = np.array([])
    for f in files:
        if f.endswith("html") or f.endswith("meta") or f.endswith("npy"):
            continue
        with open(os.path.join(data_dir,f), 'rb') as fo:
            ds = pickle.load(fo, encoding='bytes')
            temp_x = np.array(ds[b'data']) 
            temp_y = np.array(ds[b'labels'])

        if f.startswith("data"):
            x_train = np.concatenate((temp_x,x_train), axis=0)
            y_train = np.concatenate((temp_y,y_train), axis=0)

        if f.startswith("test"):
            x_test = temp_x
            y_test = temp_y
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    import sys
    x_test=np.load(sys.path[0] +'/../data_CIFAR/private_test_images_2022.npy')
    
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index=int(x_train.shape[0]*0.8)
    print(split_index)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

