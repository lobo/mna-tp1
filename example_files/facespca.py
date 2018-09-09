# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:32:14 2017

@author: pfierens
"""
from os import listdir
from os.path import join, isdir
from scipy import ndimage as im
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

mypath      = 'att_faces/'
onlydirs    = [f for f in listdir(mypath) if isdir(join(mypath, f))]

#image size
horsize     = 92
versize     = 112
areasize    = horsize*versize

#number of figures
personno    = 40
trnperper   = 6
tstperper   = 4
trnno       = personno*trnperper
tstno       = personno*tstperper

#TRAINING SET
images = np.zeros([trnno,areasize])
person = np.zeros([trnno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(1,trnperper+1):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        images[imno,:] = np.reshape(a,[1,areasize])
        person[imno,0] = per
        imno += 1
    per += 1

#TEST SET
imagetst  = np.zeros([tstno,areasize])
persontst = np.zeros([tstno,1])
imno = 0
per  = 0
for dire in onlydirs:
    for k in range(trnperper,10):
        a = im.imread(mypath + dire + '/{}'.format(k) + '.pgm')/255.0
        imagetst[imno,:]  = np.reshape(a,[1,areasize])
        persontst[imno,0] = per
        imno += 1
    per += 1


    
#CARA MEDIA
meanimage = np.mean(images,0)
print(meanimage[0])
print((np.reshape(meanimage,[versize,horsize])*255)[0][0])
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(meanimage,[versize,horsize])*255,cmap='gray')
fig.suptitle('Imagen media')
plt.show()
#resto la media
images  = [images[k,:]-meanimage for k in range(images.shape[0])]
imagetst= [imagetst[k,:]-meanimage for k in range(imagetst.shape[0])]

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

nmax = V.shape[0]
accs = np.zeros([nmax,1])
for neigen in range(1,nmax):
    #Me quedo sólo con las primeras autocaras
    B = V[0:neigen,:]
    #proyecto
    improy      = np.dot(images,np.transpose(B))
    imtstproy   = np.dot(imagetst,np.transpose(B))
        
    #SVM
    #entreno
    clf = svm.LinearSVC()
    clf.fit(improy,person.ravel())
    accs[neigen] = clf.score(imtstproy,persontst.ravel())
    print('Precisión con {0} autocaras: {1} %\n'.format(neigen,accs[neigen]*100))

fig, axes = plt.subplots(1,1)
axes.semilogy(range(nmax),(1-accs)*100)
axes.set_xlabel('No. autocaras')
axes.grid(which='Both')
fig.suptitle('Error')

