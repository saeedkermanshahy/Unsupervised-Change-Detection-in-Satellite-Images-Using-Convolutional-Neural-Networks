from re import S
from PIL.Image import new
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import difference, downsample, upsample


class Unet(tf.keras.Model):

  def __init__(self, comparison=False, mode='train'):
    super(Unet, self).__init__()
    self.comparison = comparison
    self.mode = mode
    self.prev_down_1 = None
    self.prev_down_2 = None
    self.prev_down_3 = None
    self.prev_down_4 = None
    self.prev_down_5 = None
    
    # self.down1 = layers.Conv2D(64, 4, strides=2, padding='same',
    #                          kernel_initializer=initializer(), use_bias=False)
    # self.down1 = layers.BatchNormalization()
    # self.down1 = layers.LeakyReLU() # (bs, 128, 128, 64)

    self.down1 = downsample(16, 4, apply_norm=False)
    self.down2 = downsample(32, 4)  # (bs, 64, 64, 128)
    self.down3 = downsample(64, 4)  # (bs, 32, 32, 256)
    self.down4 = downsample(128, 4)  # (bs, 16, 16, 512)
    self.down5 = downsample(256, 4)  # (bs, 8, 8, 512)


    self.up4 = upsample(128, 4, apply_dropout=True)  # (bs, 16, 16, 256)
    self.up3 = upsample(64, 4)  # (bs, 32, 32, 128)
    self.up2 = upsample(32, 4)  # (bs, 64, 64, 64)
    self.up1 = upsample(16, 4)  # (bs, 128, 128, 32)

    initializer = tf.random_normal_initializer(0., 0.02)
    self.last = tf.keras.layers.Conv2DTranspose(
      1, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='sigmoid')  # (bs, 256, 256, 3)
    
    self.concat = tf.keras.layers.Concatenate()
  
  def difference(self, tensor1, tensor2, threshold):
    subtract = tf.abs(tensor1 - tensor2)
    new_tensor = tf.where(tf.abs(subtract) <= threshold, 0.0, tensor1)
    return new_tensor

  def call(self, inputs):
    down1 = self.down1(inputs)
    down2 = self.down2(down1)
    down3 = self.down3(down2)
    down4 = self.down4(down3)
    down5 = self.down5(down4)

    
    if self.mode=='test' and self.comparison==False:
        self.prev_down_1 = down1
        self.prev_down_2 = down2
        self.prev_down_3 = down3
        self.prev_down_4 = down4
        self.prev_down_5 = down5

    
    if self.mode=='test' and self.comparison==True:
        down1 = tf.abs(down1 - self.prev_down_1)
        down1 = tf.where(tf.abs(down1) <= 0.001, 0.0, down1)

        down2 = tf.abs(down2 - self.prev_down_2)
        down2 = tf.where(tf.abs(down2) <= 0.001, 0.0, down2)

        down3 = tf.abs(down3 - self.prev_down_3)
        down3 = tf.where(tf.abs(down3) <= 0.001, 0.0, down3)

        down4 = tf.abs(down4 - self.prev_down_4)
        down4 = tf.where(tf.abs(down4) <= 0.001, 0.0, down4)

        down5 = tf.abs(down5 - self.prev_down_5)
        down5 = tf.where(tf.abs(down5) <= 0.0001, 0.0, down5)
        # down1 = self.difference(down1, self.prev_down_1, threshold=0.6)
        # down2 = self.difference(down2, self.prev_down_2, threshold=0.6)
        # down3 = self.difference(down3, self.prev_down_3, threshold=0.8)
        # down4 = self.difference(down4, self.prev_down_4, threshold=1.0)
        # down5 = self.difference(down5, self.prev_down_5, threshold=1.2)



    up4 = self.up4(down5)
    up4 = self.concat([up4, down4])
    up3 = self.up3(up4)
    up3 = self.concat([up3, down3])
    up2 = self.up2(up3)
    up2 = self.concat([up2, down2])
    up1 = self.up1(up2)
    up1 = self.concat([up1, down1])

    last = self.last(up1)

    return last

