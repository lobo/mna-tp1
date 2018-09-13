#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import argparse
import matplotlib.pyplot as plt
import numpy as np
import image
import eig
from sklearn import svm
import video_kernel

CAPTURED_VARIANCE = 0.9

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Complete path with sub-directories containing images")
args = parser.parse_args()

print("Loading images...")
grayscaled_images, categories = image.load_images(args.directory)

# Normalizing?
grayscaled_images = (grayscaled_images - image.NORMALIZE_FACTOR / 2) / (image.NORMALIZE_FACTOR / 2)

print("Building kernel...")
m = len(grayscaled_images)
# Build kernel. We use a 2 degree polynomial
K = np.zeros((m, m))
for i in range(m):
	for j in range(m):
		K[i,j] = grayscaled_images[i].transpose().dot(grayscaled_images[j])**2

# Center kernel in feature space
print("Centering kernel...")
inverse_m_matrix = np.ones((m,m))/m
K = K - inverse_m_matrix.dot(K) - K.dot(inverse_m_matrix) + inverse_m_matrix.dot(K).dot(inverse_m_matrix)

print("Computing eigenvalues...")
eigen_values = eig.sorted_eigen_values(K)

total_eigen_values_sum = sum(eigen_values)
partial_eigen_value_sum = 0
used_eigen_faces = 0
for ev in eigen_values:
	partial_eigen_value_sum += ev
	used_eigen_faces += 1
	if (partial_eigen_value_sum/total_eigen_values_sum > CAPTURED_VARIANCE):
		break

pseudo_eigen_faces = list()
print("Computing " + str(used_eigen_faces) + " eigenfaces...")
for i in range(used_eigen_faces):
	pseudo_eigen_face = eig.inverse_iteration(K, eigen_values[i])
	pseudo_eigen_face = [item for sublist in pseudo_eigen_face for item in sublist]	# The above computation yields the eigenface as a list of one-dimentional lists, so we flatten it
	pseudo_eigen_face /= np.sqrt(eigen_values[i])
	pseudo_eigen_faces.append(pseudo_eigen_face)
pseudo_eigen_faces = np.asarray(pseudo_eigen_faces)

print("Projecting images...")
projected_images = pseudo_eigen_faces.dot(K).transpose()

# Classify
print("Classifying...")
classifier = svm.LinearSVC()
image_classes = [category for category in categories for _ in range(image.IMAGES_PER_DIRECTORY)]
classifier.fit(projected_images, image_classes)

# Predict
video_kernel.recognize_faces(K, pseudo_eigen_faces, grayscaled_images, classifier)