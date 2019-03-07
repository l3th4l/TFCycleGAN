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
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        #96 x 96 x 3      
        #Encoder 
        cnv1 = conv2d(x, [3, 3], 8, strides = [2, 2], name = 'cnv1') # --> 48 x 48 x 8
        cnv2 = conv2d(cnv1, [3, 3], 16, strides = [2, 2], name = 'cnv2') # --> 24 x 24 x 16

        res1 = res_block(cnv2, [4, 4], 32, name = 'res1') # --> 24 x 24 x 32

        cnv3 = conv2d(res1, [3, 3], 64, strides = [2, 2], name = 'cnv3') # --> 12 x 12 x 64
        cnv4 = conv2d(cnv3, [3, 3], 128, strides = [2, 2], name = 'cnv4') # --> 6 x 6 x 128
        cnv5 = conv2d(cnv4, [3, 3], 256, strides = [2, 2], name = 'cnv5') # --> 3 x 3 x 256
        cnv6 = conv2d(cnv5, [3, 3], 512, strides = [1, 1], name = 'cnv6', padding = 'valid') # --> 1 x 1 x 512 
        
        fc1 = conv2d(cnv6, [1, 1], 1024, strides = [1, 1], name = 'fc1', padding = 'valid') # --> 1 x 1 x 1024

        #cnv4 = layers.batch_normalization(cnv4) # Batch norm

        #Decoder 
        tcnv1 = tConv2d(fc1, [3, 3], 256, strides = [1, 1], name = 'tcnv1', padding = 'valid') # --> 3 x 3 x 256
        tcnv2 = tConv2d(tcnv1 + cnv5, [3, 3], 128, strides = [2, 2], name = 'tcnv2') # --> 6 x 6 x 128
        tcnv3 = tConv2d(tcnv2 + cnv4, [3, 3], 64, strides = [2, 2], name = 'tcnv3') # --> 12 x 12 x 64

        res2 = res_block(tcnv3, [4, 4], 32, name = 'res2') # --> 12 x 12 x 32

        tcnv4 = tConv2d(res2, [3, 3], 32, strides = [2, 2], name = 'tcnv4') # --> 24 x 24 x 32
        tcnv5 = tConv2d(tcnv4, [3, 3], 32, strides = [2, 2], name = 'tcnv5') # --> 48 x 48 x 16
        tcnv6 = tConv2d(tcnv5, [3, 3], 32, strides = [2, 2], name = 'tcnv6') # --. 96 x 96 x 32
        
        #tcnv4 = layers.batch_normalization(tcnv4)

        #Some fancy stuff
        cnv7 = conv2d(tcnv6, [3, 3], 16, name = 'cnv7') # --> 96 x 96 x 16
        res3 = res_block(cnv7, [4, 4], 16, name = 'res3') # --> 96 x 96 x 32

        #res1 =  layers.batch_normalization(res1) # Batch norm

        logit = conv2d(res3, [4, 4], 3, name = 'logit') # --> 96 x 96 x 3
        #Output 
        return logit, tf.nn.sigmoid(logit)

def disc(x, name = 'disc'):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        #96 x 96 x 3
        cnv1 = conv2d(x, [3, 3], 16, strides = [2, 2], name = 'cnv1') # --> 48 x 48 x 8
        cnv2 = conv2d(cnv1, [3, 3], 32, strides = [2, 2], name = 'cnv2') # --> 24 x 24 x 32
        cnv3 = conv2d(cnv2, [3, 3], 64, strides = [2, 2], name = 'cnv3') # --> 12 x 12 x 64
        cnv4 = conv2d(cnv3, [3, 3], 128, strides = [2, 2], name = 'cnv4') # --> 6 x 6 x 128 
        cnv5 = conv2d(cnv4, [3, 3], 64, strides = [2, 2], name = 'cnv5') # --> 3 x 3 x 64 

        #cnv5 = layers.batch_normalization(cnv5) # Batch norm

        logit = conv2d(cnv5, [3, 3], 1, strides = [1, 1], padding = 'valid', name = 'cnv6') # --> 1 x 1 x 1
        return logit, tf.nn.sigmoid(logit)