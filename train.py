import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt 
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
    
    def make_batch(self, dat, path, batch_size, ind = 0, offset = 0,  aug_chance = 0.3):
        imgs = []
        rnd_indexes = np.array(range(batch_size)) + ind * batch_size + offset
        for index in rnd_indexes:
            imgs.append(self.fetch_img('%s/%s' % (path, dat.loc[index][0]), aug_chance))
        # returns normalized and flattened images along with the labels
        return np.reshape(np.array(imgs) / 256, [batch_size, 96 * 96 * 4])

    def fetch_img(self, path, aug_chance):
        r = np.random.uniform()
        angle = 0
        if r < aug_chance:
            angle = np.random.uniform(90)
        return self.augment(skimage.io.imread('%s' % (path)), angle = angle)
    
    def augment(self, img, angle = 0):
        if angle == 0:
            return img
        return skimage.transform.rotate(img, angle)

#Define ops 
a_real = tf.placeholder(tf.float32, shape = [None, 96, 96, 3])
b_real = tf.placeholder(tf.float32, shape = [None, 96, 96, 3])

with tf.variable_scope('Model') as scope:
    #Generator output
    b_gen = nets.gen(a_real, name = 'g_AtoB')
    a_gen = nets.gen(b_real, name = 'g_BtoA')

    #Discriminator outputs for real inputs
    a_dis = nets.disc(a_real, name = 'disc_A')
    b_dis = nets.disc(b_real, name = 'disc_B')

    #Discriminator outputs for fake input 
    a_gen_dis = nets.disc(a_gen, name = 'disc_A')
    b_gen_dis = nets.disc(b_gen, name = 'disc_B')

    #Cyclic generator output
    a_cyc = nets.gen(b_gen, name = 'g_BtoA')
    b_cyc = nets.gen(a_gen, name = 'g_AtoB')

#Discriminator loss for real inputs 
loss_dis_a_1 = tf.reduce_mean(tf.squared_difference(a_dis, 1))
loss_dis_b_1 = tf.reduce_mean(tf.squared_difference(b_dis, 1))

#Discriminator loss for fake inputs
loss_dis_a_2 = tf.reduce_mean(tf.squared_difference(a_gen_dis, 0))
loss_dis_b_2 = tf.reduce_mean(tf.squared_difference(b_gen_dis, 0))

#Generator loss
loss_gen_a_1 = tf.reduce_mean(tf.squared_difference(a_gen_dis, 1))
loss_gen_b_1 = tf.reduce_mean(tf.squared_difference(b_gen_dis, 1))

#Cyclic loss 
loss_cyc = tf.reduce_mean(tf.squared_difference(a_real, a_cyc)) + tf.reduce_mean(tf.squared_difference(b_real, b_cyc))

#Combined loss 
loss_gen_a = loss_gen_a_1 + 10 * loss_cyc
loss_gen_b = loss_gen_b_1 + 10 * loss_cyc
loss_dis_a = (loss_dis_a_1 + loss_dis_a_2) / 2
loss_dis_b = (loss_dis_b_1 + loss_dis_b_2) / 2

#Trainable variables 
tvars = tf.trainable_variables(scope = 'Model')

var_gen_a = [v for v in tvars if 'g_BtoA' in v]
var_gen_b = [v for v in tvars if 'g_AtoB' in v]
var_dis_a = [v for v in tvars if 'disc_A' in v]
var_dis_b = [v for v in tvars if 'disc_B' in v]

#Learning rate
lr = 0.01

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)

#Optimize ops
opt_gen_a = optimizer.minimize(loss_gen_a, var_list = var_gen_a)
opt_gen_b = optimizer.minimize(loss_gen_b, var_list = var_gen_b)
opt_dis_a = optimizer.minimize(loss_dis_a, var_list = var_dis_a)
opt_dis_b = optimizer.minimize(loss_dis_b, var_list = var_dis_b)