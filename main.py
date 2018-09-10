import argparse
import matplotlib.pyplot as plt
import numpy as np
import image
import eig

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Complete path with sub-directories which contain images.")
args = parser.parse_args()

grayscaled_images = image.load_images(args.directory)
print(grayscaled_images)

# Normalizing
grayscaled_images = grayscaled_images / image.NORMALIZE_FACTOR

# Substract the mean face
mean_face =  np.mean(grayscaled_images, 0)
fig, axes = plt.subplots(1,1)
axes.imshow(np.reshape(mean_face,[image.VERTICAL_SIZE,image.HORIZONTAL_SIZE])*image.NORMALIZE_FACTOR,cmap='gray')
fig.suptitle('Imagen media')
plt.show()

grayscaled_images = mean_face