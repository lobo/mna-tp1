# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:42:03 2017

@author: pfierens
"""


from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt

mypath      = 'att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = 10
trnno       = personno*trnperper

#TRAINING SET
images = np.zeros([trnno,areasize])
imno = 0
for dire in onlydirs:
    for k in range(1,trnperper+1):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        images[imno,:] = np.reshape(a,[1,areasize])
        imno += 1

    
#CARA MEDIA
meanimage = np.mean(images,0)
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
fig.suptitle('Imagen media')

#resto la media
images  = [images[k,:]-meanimage for k in range(images.shape[0])]

#PCA
U,S,V = np.linalg.svd(images,full_matrices = False)

#Primera autocara...
eigen1 = (np.reshape(V[0,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen1,cmap='gray')
fig.suptitle('Primera autocara')

eigen2 = (np.reshape(V[1,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Segunda autocara')

eigen3 = (np.reshape(V[2,:],[versize,horsize]))*255
fig, axes = plt.subplots(1,1)
axes.imshow(eigen2,cmap='gray')
fig.suptitle('Tercera autocara')

