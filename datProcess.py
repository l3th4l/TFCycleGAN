import os 
from skimage import transform
from skimage import io
import numpy as np 

root = './Datasets/'
new_path = 'processed/'

domain_a = 'celeb/'
domain_b = 'anime/'

#os.mkdir(root + new_path)
#os.mkdir(root + new_path + domain_b)
#os.mkdir(root + new_path)
#os.mkdir(root + new_path + domain_a)

#For domain b
sub_b = os.listdir(root + domain_b)
for subdir in sub_b:
    imgs = os.listdir(root + domain_b + subdir)
    for img in imgs:
        try:
            im = io.imread(root + domain_b + subdir + '/' + img)
            im = transform.resize(im, [96, 96])
            io.imsave(root + new_path + domain_b + '/' + img , im)
        except:
            pass

#For domain a
imgs = os.listdir(root + domain_a)
for img in imgs:
    im = io.imread(root + domain_a + '/' + img)
    im = transform.resize(im, [96, 96])
    io.imsave(root + new_path + domain_a + '/' + img , im)