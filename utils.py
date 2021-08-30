from dataset import load_image
import tensorflow as tf
import argparse
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img as keras_load_img


def parser():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("--mode", default="train", choices=['train', 'test'],
        help="Train for training unet model and test for change detection")
    my_parser.add_argument("--comparison", type=bool, default=False, 
        help="To compare current and prev image")
    my_parser.add_argument("--train_image_dir", help="Directory of train images", required=False)
    my_parser.add_argument("--train_mask_dir", help="Directory of train images", required=False)
    my_parser.add_argument("--test_image_dir", help="Directory of test images", required=False)
    my_parser.add_argument("--test_mask_dir", help="Directory of test images", required=False)
    my_parser.add_argument("--checkpoint_path", help="Path for saving model weights", required=False)
    my_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    my_parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    my_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    my_parser.add_argument("--val_size", type=float, default=0.2, help="Batch size for training")
    my_parser.add_argument("--shuffle_buffer", type=int, default=1000, help="Buffer size for shuffling that spesific number of images")
    my_parser.add_argument("--background_image_path", help="To compare current images with it", required=False)
    my_parser.add_argument("--save_predictions", default=False, help="To save predictions")

    args = my_parser.parse_args()
    return args


def downsample(filters, size, apply_norm=True):
    """Downsamples an input.
    Conv2D => Batchnorm => LeakyRelu
    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
    Returns:
    Downsample Sequential Model
    """   
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(tf.keras.layers.BatchNormalization())


    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
    Returns:
    Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())


    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def difference(tensor1, tensor2, threshold=1):
    # bat = tensor1.shape[0]
    # c = tensor1.shape[1]
    # l = tensor1.shape[2]
    # w = tensor1.shape[3]
    # new_tensor = tf.zeros((bat, c, l, w), dtype=tf.float32)
    # tensor1 = tensor1.numpy()
    # tensor2 = tensor2.numpy()
    # subtract = tf.subtract(tensor1, tensor2)
    subtract = tf.abs(tensor1 - tensor2)
    new_tensor = tf.where(tf.abs(subtract) <= threshold, 0.0, tensor1)
    # new_tensor = tf.Variable(new_tensor, trainable=True)
    return new_tensor


def save_predictions(model, dataset):
    for i, (image, mask) in enumerate(dataset.take(len(dataset))):
        predict = model.predict(image)
        # print(predict.shape)
        predict = np.where(predict > 0.5, 255.0, 0.0)
        predict = predict.squeeze()
        print(predict.shape)
        cv2.imwrite(f'predictions/{i}.jpg', predict)


def read_background(path):
    img = keras_load_img(path)
    img = img.resize((256, 256))
    img = np.asarray(img, dtype=np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

