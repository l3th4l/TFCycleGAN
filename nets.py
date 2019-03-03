import tensorflow as tf 
from tensorflow import layers

def conv2d(x, kernel_size, channels, strides = [1, 1], padding = 'same', name = None):
    return layers.conv2d(x, filters = channels, kernel_size = kernel_size, strides = strides, padding = padding, name = name)

def tConv2d(x, kernel_size, channels, strides = [1, 1], padding = 'same', name = None):
    return layers.conv2d_transpose(x, filters = channels, kernel_size = kernel_size, strides = strides, padding = padding, name = name)

def res_block(x, kernel_size, channels, name = 'res'):
    r1 = conv2d(x, kernel_size, channels, name = name + '1')
    r2 = conv2d(r1, kernel_size, channels, name = name + '2')
    return r1 + r2

def gen(x, name = 'gen'):
    with tf.variable_scope(name):
        #96 x 96 x 3      
        #Encoder 
        cnv1 = conv2d(x, [3, 3], 8, strides = [2, 2], name = 'cnv1') # --> 48 x 48 x 8
        cnv2 = conv2d(cnv1, [3, 3], 16, strides = [2, 2], name = 'cnv2') # --> 24 x 24 x 16
        cnv3 = conv2d(cnv2, [3, 3], 32, strides = [2, 2], name = 'cnv3') # --> 12 x 12 x 32
        cnv4 = conv2d(cnv3, [3, 3], 64, strides = [2, 2], name = 'cnv4') # --> 6 x 6 x 64

        cnv4 = layers.batch_normalization(cnv4) # Batch norm

        #Decoder 
        tcnv1 = tConv2d(cnv4, [3, 3], 32, strides = [2, 2], name = 'tcnv1') # --> 12 x 12 x 32
        tcnv2 = tConv2d(tcnv1 + cnv3, [3, 3], 16, strides = [2, 2], name = 'tcnv2') # --> 24 x 24 x 16
        tcnv3 = tConv2d(tcnv2 + cnv2, [3, 3], 8, strides = [2, 2], name = 'tcnv3') # --> 48 x 48 x 8
        tcnv4 = tConv2d(tcnv3 + cnv1, [3, 3], 3, strides = [2, 2], name = 'tcnv4') # --> 96 x 96 x 3
        
        tcnv4 = layers.batch_normalization(tcnv4)

        #Some fancy stuff
        cnv5 = conv2d(tcnv4 + x, [3, 3], 16, name = 'cnv5') # --> 96 x 96 x 16
        res1 = res_block(cnv5, [4, 4], 32, name = 'res1') # --> 96 x 96 x 32

        res1 =  layers.batch_normalization(res1) # Batch norm

        logit = conv2d(res1, [4, 4], 3, name = 'logit') # --> 96 x 96 x 3
        #Output 
        return logit

def disc(x, name = 'disc'):
    with tf.variable_scope(name):
        #96 x 96 x 3
        cnv1 = conv2d(x, [3, 3], 16, strides = [2, 2], name = 'cnv1') # --> 48 x 48 x 8
        cnv2 = conv2d(cnv1, [3, 3], 32, strides = [2, 2], name = 'cnv2') # --> 24 x 24 x 32
        cnv3 = conv2d(cnv2, [3, 3], 64, strides = [2, 2], name = 'cnv3') # --> 12 x 12 x 64
        cnv4 = conv2d(cnv3, [3, 3], 128, strides = [2, 2], name = 'cnv4') # --> 6 x 6 x 128 
        cnv5 = conv2d(cnv4, [3, 3], 64, strides = [2, 2], name = 'cnv5') # --> 3 x 3 x 64 

        cnv5 = layers.batch_normalization(cnv5) # Batch norm

        logit = conv2d(cnv5, [3, 3], 1, strides = [1, 1], padding = 'valid', name = 'cnv6') # --> 1 x 1 x 1
        return logit