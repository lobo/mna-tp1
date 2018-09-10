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
    matrix = np.copy(A)
    for k in range(n):
        s = 0
        for j in range(m):
            s = s + matrix[j][k] ** 2
        r[k,k] = s ** 0.5
        for j in range(m):
            q[j][k] = matrix[j][k] / r[k][k]
        for i in range(k+1, n):
            aux = 0
            for j in range(m):
                aux+= matrix[j][i] * q[j][k]
            r[k][i] = aux
            for j in range(m):
                matrix[j][i] = matrix[j][i] - r[k][i] * q[j][k]
            
    return q, r

def inverse_iteration(matrix, eigenvalue):
     n = len(matrix)
     I = np.identity(n)
     eigenvector = np.random.rand(n, 1)
     inv = np.linalg.inv(matrix - eigenvalue * I)
     for i in range(100):
         aux = np.dot(inv, eigenvector)
         aux = aux/np.linalg.norm(aux)
         eigenvector = aux

     return eigenvector


def eigen_values_and_vectors(A):
    n = len(A)
    Q, R = householder_QR(A)
    matrix = np.copy(A)
    for k in range(10000):
        Q, R = householder_QR(matrix)
        matrix = R.dot(Q)

    eigen_vectors = np.zeros([n, n])
    eigen_values = sorted(np.diagonal(matrix), key=abs)[::-1]
    for i in range(len(eigen_values)):
       eigen_vector = inverse_iteration(A, eigen_values[i])
       for j in range(n):
           eigen_vectors[j][i] = eigen_vector[j]
    return eigen_values, eigen_vectors



# HouseHolder QR method as described here:
# https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
def householder_QR(A):
    m = len(A)
    n = len(A[0])
    Q = np.identity(m)
    R = np.copy(A)
    for j in range(n):
        normx = np.linalg.norm(R[j:m,j])
        s = - np.sign(R[j,j])
        u1 = R[j,j] - s * normx
        w = R[j:m, j].reshape((-1,1)) / u1
        w[0] = 1
        tau = -s * u1 / normx
        R[j:m, :] = R[j:m, :] - (tau * w) * np.dot(w.reshape((1,-1)),R[j:m,:])
        Q[:, j:n] = Q[:,j:n] - (Q[:,j:m].dot(w)).dot(tau*w.transpose())

    return Q, R


# Schur decomposition as described here:
# https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
def schur_decomposition(A):
    """docstring for schurDecomposition"""
    matrix = np.copy(A)
    U = np.identity(len(A))
    for k in range(1000):
        Q, R = householder_QR(matrix)
        matrix = R.dot(Q)
        U = U.dot(Q)
    return U, matrix


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




m = len(A)
n = len(A[0])
Q = np.identity(m)
#for k in range(n):
z = A[k:m, k]
v = [[- np.sign(z[0])*np.linalg.norm(z) - z[0]],-z[1:len(z)]]
v = np.asarray([item for sublist in v for item in sublist]).reshape((-1,1))
v = v / (v.T.dot(v) ** 0.5)

for j in range(n):
    A[k:m, j] = A[k:m, j] -  v.dot(2 * v.T.dot(A[k:m, j]))
for j in range(m):
    Q[k:m, j] = Q[k:m, j] - v.dot(2 * v.T.dot(Q[k:m, j]))
    return Q
