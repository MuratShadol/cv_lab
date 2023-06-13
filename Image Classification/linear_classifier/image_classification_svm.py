import cv2
import numpy as np
from pathlib import Path
from builtins import range
import keras.utils as image
from skimage.io import imread
from linear_classifier import LinearSVM


TRAIN_PATH = 'D:/DataScience/Lab/Image Classification/Classification_data/train/subset/'
TEST_PATH = 'D:/DataScience/Lab/Image Classification/Classification_data/test/subset'


def load_image_files(container_path, chanel, dimension=(64, 64)):
    """
    Iterate for all images in the directory 
    and use only a specific channel of the images, which are in BGR format, 
    and resize the images to a size of (50, 50). 
    Using the Keras pre-processing library the image is converted to an array and then normalised. 
    Append each such derived array into a numpy array X.

    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    count = 0
    train_img = []
    for i, direc in enumerate(folders):
        for f in direc.iterdir():
            count += 1
            img = imread(f)
            img_pred = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
            img_pred = img_pred[:, :, chanel]
            img_pred = image.img_to_array(img_pred)
            img_pred = img_pred / 255
            train_img.append(img_pred)
    X = np.array(train_img)
    return X

# The load_image_files function is called separately for individual channels of the BGR format 
# and stored in X0, X1 and X2 respectively, with shape (600, 50, 50, 1). 
# The individual arrays are merged into numpy array X with shape (600, 50, 50, 3).

X_train = np.zeros((3000, 50, 50, 3))
X0 = load_image_files(TRAIN_PATH, 0)
X1 = load_image_files(TRAIN_PATH, 1)
X2 = load_image_files(TRAIN_PATH, 2)

X_train[:, :, :, 0] = X0[:, :, :, 0]
X_train[:, :, :, 1] = X1[:, :, :, 0]
X_train[:, :, :, 2] = X2[:, :, :, 0]

# Generate y for 6 categories
yb_train = np.zeros(500)
yf_train = np.ones(500)
yg_train = np.full(500, 2)
ym_train = np.full(500, 3)
ys_train = np.full(500, 4)
yst_train = np.full(500, 5) 
y_train = []
y_train = np.concatenate((yb_train, yf_train, yg_train, ym_train, ys_train, yst_train), axis=0)

# Test set
X_test = np.zeros((600, 50, 50, 3))
X0 = load_image_files(TEST_PATH, 0)
X1 = load_image_files(TEST_PATH, 1)
X2 = load_image_files(TEST_PATH, 2)

X_test[:, :, :, 0] = X0[:, :, :, 0]
X_test[:, :, :, 1] = X1[:, :, :, 0]
X_test[:, :, :, 2] = X2[:, :, :, 0]

yb_test = np.zeros(100)
yf_test = np.ones(100)
yg_test = np.full(100, 2)
ym_test = np.full(100, 3)
ys_test = np.full(100, 4)
yst_test = np.full(100, 5) 
y_test = []
y_test = np.concatenate((yb_test, yf_test, yg_test, ym_test, ys_test, yst_test), axis=0)

# Reshape data in useful form
num_train = X_train.shape[0]
mask = list(range(num_train))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = X_test.shape[0]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# Getting data to zero mean
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image

# Append the bias dimension of ones (i.e. bias trick) so that our
# SVM only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

svmd = LinearSVM()
loss_hist = svmd.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=False)
y_train_pred = svmd.predict(X_train)
y_test_pred = svmd.predict(X_test)
print("Training accuracy: %f" %(np.mean(y_train == y_train_pred),))
print("Test accuracy: %f" %(np.mean(y_test == y_test_pred)))

""" 
Training accuracy: 0.396667
Test accuracy: 0.395000

"""
