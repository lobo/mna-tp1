import argparse
import matplotlib.pyplot as plt
import numpy as np
import image
import eig

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Complete path with sub-directories which contain images.")
args = parser.parse_args()

grayscaled_images = image.load_images(args.directory)

# Normalizing
grayscaled_images = grayscaled_images / image.NORMALIZE_FACTOR

# Substract the mean face
mean_face =  np.mean(grayscaled_images, 0)
grayscaled_images -= mean_face

# Show the mean face
#fig, axes = plt.subplots(1,1)
#axes.imshow(np.reshape(mean_face,[image.VERTICAL_SIZE,image.HORIZONTAL_SIZE])*image.NORMALIZE_FACTOR,cmap='gray')
#fig.suptitle('Imagen media')
#plt.show()

# Let's call A the matrix with the images.
# We need the eigenfaces. These are the columns of the matrix V in the SVD decomposition of A.
# The columns of V are the eigenvectors of what we call the right covariance matrix (AᵀxA).
# The problem is that this matrix is too large (10304x10304). So instead we calculate the columns U in the SVD decomposition.
# The columns of U are the eigenvectors of the left covariance matrix (AxAᵀ). With the columns of U, we can get the columns of V by knowing that:
# Aᵀ x u = σ x v
# where u is a column of U, σ is the corresponding singular value and v is the corresponding column of V
left_covariance_matrix = grayscaled_images.dot(grayscaled_images.transpose())

eigen_values = eig.sorted_eigen_values(left_covariance_matrix)
first_left_singular_vector = eig.inverse_iteration(left_covariance_matrix, eigen_values[0])
eigen_face = grayscaled_images.transpose().dot(first_left_singular_vector)/np.sqrt(eigen_values[0])

# First eigenface according to us
eigen1 = (np.reshape(eigen_face,[image.VERTICAL_SIZE, image.HORIZONTAL_SIZE]))*image.NORMALIZE_FACTOR
fig, axes = plt.subplots(1,1)
axes.imshow(eigen1,cmap='gray')
fig.suptitle('First eigenface according to us')
plt.show()

# First eigenface according to pfierens
U,S,V = np.linalg.svd(grayscaled_images,full_matrices = False)
eigen1 = (np.reshape(V[0,:],[image.VERTICAL_SIZE,image.HORIZONTAL_SIZE]))*image.NORMALIZE_FACTOR
fig, axes = plt.subplots(1,1)
axes.imshow(eigen1,cmap='gray')
fig.suptitle('First eigenface according to pfierens')
plt.show()