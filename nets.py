import tensorflow as tf 
from tensorflow.contrib import layers

def conv2d(x, kernel_size, channels, strides = [1, 1], padding = 'SAME'):
    return layers.conv2d(x, num_outputs = channels, kernel_size = kernel_size, stride = strides, padding = padding)

def tConv2d(x, kernel_size, channels, strides = [1, 1], padding = 'SAME'):
    return layers.conv2d_transpose(x, num_outputs = channels, kernel_size = kernel_size, stride = strides, padding = padding)

def res_block(x, kernel_size, channels):
    r1 = conv2d(x, kernel_size, channels)
    r2 = conv2d(r1, kernel_size, channels)
    return r1 + r2

class Gen():

    def __init__(self, x):
        #96 x 96 x 3

        #Encoder 
        self.cnv1 = conv2d(x, [3, 3], 8, strides = [2, 2]) # --> 48 x 48 x 8
        self.cnv2 = conv2d(self.cnv1, [3, 3], 16, strides = [2, 2]) # --> 24 x 24 x 16
        self.cnv3 = conv2d(self.cnv2, [3, 3], 32, strides = [2, 2]) # --> 12 x 12 x 32
        self.cnv4 = conv2d(self.cnv3, [3, 3], 64, strides = [2, 2]) # --> 6 x 6 x 64

        #Decoder 
        self.tcnv1 = tConv2d(self.cnv4, [3, 3], 32, strides = [2, 2]) # --> 12 x 12 x 32
        self.tcnv2 = tConv2d(self.tcnv1 + self.cnv3, [3, 3], 16, strides = [2, 2]) # --> 24 x 24 x 16
        self.tcnv3 = tConv2d(self.tcnv2 + self.cnv2, [3, 3], 8, strides = [2, 2]) # --> 48 x 48 x 8
        self.tcnv4 = tConv2d(self.tcnv3 + self.cnv1, [3, 3], 3, strides = [2, 2]) # --> 96 x 96 x 3

        #Some fancy stuff
        self.cnv5 = conv2d(self.tcnv4 + x, [3, 3], 16) # --> 96 x 96 x 16
        self.res1 = res_block(self.cnv5, [4, 4], 32) # --> 96 x 96 x 32
        self.logit = conv2d(self.res1, [4, 4], 3) # --> 96 x 96 x 3

        #Output 
        self.out = tf.nn.sigmoid(self.logit)

class Disc():

    def __init__(self, x):
        #96 x 96 x 3
        self.cnv1 = conv2d(x, [3, 3], 16, strides = [2, 2]) # --> 48 x 48 x 8
        self.cnv2 = conv2d(self.cnv1, [3, 3], 32, strides = [2, 2]) # --> 24 x 24 x 32
        self.cnv3 = conv2d(self.cnv2, [3, 3], 64, strides = [2, 2]) # --> 12 x 12 x 64
        self.cnv4 = conv2d(self.cnv3, [3, 3], 128, strides = [2, 2]) # --> 6 x 6 x 128 
        self.cnv5 = conv2d(self.cnv4, [3, 3], 64, strides = [2, 2]) # --> 3 x 3 x 64 
        self.cnv7 = conv2d(self.cnv6, [3, 3], 1, strides = [1, 1], padding = 'VALID') # --> 1 x 1 x 1