import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os import path, listdir

def load_images(directory):
    """Load images and return them as a byte array representing the grayscale values."""
    subdirectories = [f for f in listdir(args.directory)]
    grayscale_images = [[],[]]

    for face in subdirectories:
        images_paths = [(directory + "/" +  face + "/" +  f) for f in 
                listdir(directory + "/" + face)]
        for image_path in images_paths:
            grayscale_image = to_grayscale(image_path)
            grayscale_images[0].append(grayscale_image)
            grayscale_images[1].append(grayscale_image)
    return np.asarray(grayscale_images[0])

def to_grayscale(image_path):
    return list(Image.open(image_path).convert('L').tobytes())
    

# Modified gram schmidt as described here:
# https://www.inf.ethz.ch/personal/gander/papers/qrneu.pdf
def gram_schmidt(A):
    n = len(A[0])
    m = len(A)
    r = np.zeros([n,n])
    q = np.zeros([m,n])
    for k in range(n):
        s = 0
        for j in range(m):
            s = s + A[j][k] ** 2
        r[k,k] = s ** 0.5
        for j in range(m):
            q[j][k] = A[j][k] / r[k][k]
        for i in range(k+1, n):
            aux = 0
            for j in range(m):
                aux+= A[j][i] * q[j][k]
            r[k][i] = aux
            for j in range(m):
                A[j][i] = A[j][i] - r[k][i] * q[j][k]
            
    return q, r


parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Complete path with sub-directories which contain images.")
args = parser.parse_args()

grayscaled_images = load_images(args.directory)
print(grayscaled_images)

# Normalizing
grayscaled_images = grayscaled_images / 255.0

# Substract the mean face
mean_face =  np.mean(grayscaled_images, 0)
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(mean_face,[112,92])*255,cmap='gray')
fig.suptitle('Imagen media')
plt.show()

grayscaled_images = mean_face
