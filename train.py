from trainOps import * 
from matplotlib import pyplot as plt
import time

#Losses
losses_g_a = []
losses_g_b = []

losses_cyc = []

losses_d_a = []
losses_d_b = []

#Hyperparams
batch_size = 100
epochs = 200

dom_a_len = len(a_namelist) // batch_size - 2
dom_b_len = len(b_namelist) // batch_size - 2

min_dom_len = dom_a_len if dom_a_len < dom_b_len else dom_b_len

#Session ops
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list = tvars)
sess = tf.InteractiveSession()

sess.graph.finalize()

#Initialize variables
sess.run(init)
#saver.restore(sess, './models/weights_24/iter_450.ckpt')
#Create directories for models, generated images, loss
if not os.path.exists('./models'):
    os.mkdir('./models')

if not os.path.exists('./gen_output'):
    os.mkdir('./gen_output')

if not os.path.exists('./losses'):
    os.mkdir('./losses')

#Fixed data
fixed_dat_a = datGen.make_batch(a_namelist, path + dom_a, 4, 0, 5)
fixed_dat_b = datGen.make_batch(b_namelist, path + dom_b, 4, 0, 5)
fig, ax = plt.subplots(4, 4, True, True)

#Training loop
for epoch in range(epochs):
    for i in range(min_dom_len):
        #Start time
        s_time = time.time()

        #Make batch
        a_ind = np.random.randint(dom_a_len)
        b_ind = np.random.randint(dom_b_len)
        offset = np.random.randint(batch_size)

        dat_a = datGen.make_batch(a_namelist, path + dom_a, batch_size, a_ind, offset)
        dat_b = datGen.make_batch(b_namelist, path + dom_b, batch_size, b_ind, offset)

        #Data load time
        d_time = time.time()

        #Optimize 
        l_g_a, l_g_b, l_d_a, l_d_b, l_c, _, _, _, _ = sess.run([loss_gen_a, 
                                                                loss_gen_b, 
                                                                loss_dis_a, 
                                                                loss_dis_b, 
                                                                loss_cyc, 
                                                                opt_dis_a, 
                                                                opt_dis_b, 
                                                                opt_gen_a, 
                                                                opt_gen_b], 
                                                                feed_dict = {a_real : dat_a, b_real : dat_b})
        
        #Optimization time
        o_time = time.time()

        #Add losses to the lists 
        '''
        losses_g_a.append(np.mean(l_g_a))
        losses_g_b.append(np.mean(l_g_b))
        losses_cyc.append(np.mean(l_c))
        losses_d_a.append(np.mean(l_d_a))
        losses_d_b.append(np.mean(l_d_b))
        '''
        
        if i % 50 == 0:
            print('[epoch : %i , iter : %i] losses : discriminator[ a : %f , b : %f] , generator[ a : %f , b : %f], cycle[%f]' % 
                  (epoch, i, np.mean(l_d_a), np.mean(l_d_b), np.mean(l_g_a), np.mean(l_g_b), np.mean(l_c)))
            
            #save the model
            try:    
                os.mkdir('./models/weights_%i' % (epoch))
            except:
                pass
            saver.save(sess, './models/weights_%i/iter_%i.ckpt' % (epoch, i))        

            #display generated faces
            gen_b = sess.run(b_gen, feed_dict = {a_real : fixed_dat_a[:4]})
            gen_a = sess.run(a_gen, feed_dict = {b_real : fixed_dat_b[:4]})
            cyc_b = sess.run(b_cyc, feed_dict = {b_real : fixed_dat_b[:4]})
            cyc_a = sess.run(a_cyc, feed_dict = {a_real : fixed_dat_a[:4]})

            for j in range(4):
                ax[j, 0].imshow(gen_a[j])
                ax[j, 1].imshow(gen_b[j])
                ax[j, 2].imshow(cyc_a[j])
                ax[j, 3].imshow(cyc_b[j])

            fig.savefig('./gen_output/epoch_%i_%i' % (epoch, i))

        #Image save time
        sv_time = time.time() 

        print('time (iter : %i) : [data load : %f, optimization : %f, saving : %f] (total) : %f' % (
                                                                                        i,
                                                                                        d_time - s_time,
                                                                                        o_time - d_time, 
                                                                                        sv_time - o_time, 
                                                                                        sv_time - s_time))

saver.save(sess, './models/weights_%i/iter_%i.ckpt' % (epoch, i))     