import tensorflow as tf 
import numpy as np 
import skimage 
import os

import nets

path = './Datasets/processed/'
dom_a = 'celeb/'
dom_b = 'anime/'

a_namelist = os.listdir(path + dom_a)
b_namelist = os.listdir(path + dom_b)

#Loading data 
class datGen():
    
    def make_batch(dat, path, batch_size, ind = 0, offset = 0):
        imgs = []
        rnd_indexes = np.array(range(batch_size)) + ind * batch_size + offset
        for index in rnd_indexes:
            imgs.append(datGen.fetch_img('%s/%s' % (path, dat[index])))
        # returns normalized and flattened images along with the labels
        return np.reshape(np.array(imgs) / 256, [batch_size, 96 * 96 * 3])

    def fetch_img(path):
        return skimage.io.imread('%s' % (path))

#Define ops 
a_real = tf.placeholder(tf.float32, shape = [None, 96 * 96 * 3])
a_real_reshaped = tf.reshape(a_real, [-1, 96, 96, 3])

b_real = tf.placeholder(tf.float32, shape = [None, 96 * 96 * 3])
b_real_reshaped = tf.reshape(b_real, [-1, 96, 96, 3])

with tf.variable_scope('Model', reuse = tf.AUTO_REUSE) as scope:
    #Generator output
    l_b_gen, b_gen = nets.gen(a_real_reshaped, name = 'g_AtoB')
    l_a_gen, a_gen = nets.gen(b_real_reshaped, name = 'g_BtoA')

    #Discriminator outputs for real inputs
    l_a_dis, a_dis = nets.disc(a_real_reshaped, name = 'disc_A')
    l_b_dis, b_dis = nets.disc(b_real_reshaped, name = 'disc_B')

    #Discriminator outputs for fake input 
    l_a_gen_dis, a_gen_dis = nets.disc(a_gen, name = 'disc_A')
    l_b_gen_dis, b_gen_dis = nets.disc(b_gen, name = 'disc_B')

    #Cyclic generator output
    l_a_cyc, a_cyc = nets.gen(b_gen, name = 'g_BtoA')
    l_a_cyc, b_cyc = nets.gen(a_gen, name = 'g_AtoB')

#Discriminator loss for real inputs 
loss_dis_a_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(a_dis), logits = l_a_dis))
#tf.reduce_mean(tf.squared_difference(a_dis, 1))
loss_dis_b_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(b_dis), logits = l_b_dis))
#tf.reduce_mean(tf.squared_difference(b_dis, 1))

#Discriminator loss for fake inputs
loss_dis_a_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(a_gen_dis), logits = l_a_gen_dis))
#tf.reduce_mean(tf.squared_difference(a_gen_dis, 0))
loss_dis_b_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(b_gen_dis), logits = l_b_gen_dis))
#tf.reduce_mean(tf.squared_difference(b_gen_dis, 0))

#Generator loss
loss_gen_a_1 = tf.reduce_mean(tf.squared_difference(a_gen_dis, 1))
loss_gen_b_1 = tf.reduce_mean(tf.squared_difference(b_gen_dis, 1))

#Cyclic loss 
loss_cyc = tf.reduce_mean(tf.squared_difference(a_real_reshaped, a_cyc)) + tf.reduce_mean(tf.squared_difference(b_real_reshaped, b_cyc))

y = 12.0
e = 4.4
#Combined loss 
loss_gen_a = e * loss_gen_a_1 + y * loss_cyc
loss_gen_b = e * loss_gen_b_1 + y * loss_cyc
loss_dis_a = (loss_dis_a_1 + loss_dis_a_2) / 2
loss_dis_b = (loss_dis_b_1 + loss_dis_b_2) / 2

#Trainable variables 
tvars = tf.trainable_variables()

var_gen_a = [v for v in tvars if 'g_BtoA' in v.name]
var_gen_b = [v for v in tvars if 'g_AtoB' in v.name]
var_dis_a = [v for v in tvars if 'disc_A' in v.name]
var_dis_b = [v for v in tvars if 'disc_B' in v.name]

#Learning rate
lr = 0.0002

#Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

#Optimize ops
opt_gen_a = optimizer.minimize(loss_gen_a, var_list = var_gen_a)
opt_gen_b = optimizer.minimize(loss_gen_b, var_list = var_gen_b)
opt_dis_a = optimizer.minimize(loss_dis_a, var_list = var_dis_a)
opt_dis_b = optimizer.minimize(loss_dis_b, var_list = var_dis_b)