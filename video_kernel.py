import numpy as np
import image
import cv2
import open_cv.FaceDetection as fd

HAAR_CASCADE_FRONTAL_FACE_PATH = "./open_cv/haarcascade_frontalface_default.xml"

FRAME_COLOR = (0, 255, 0)	# Green
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = FRAME_COLOR
TEXT_ELEVATION = 16
TEXT_THICKNESS = 2

def recognize_faces(K, pseudo_eigen_faces, images, classifier):
	faceCascade = cv2.CascadeClassifier(HAAR_CASCADE_FRONTAL_FACE_PATH)
	video_capture = cv2.VideoCapture(0)
	m = len(K)
	inverse_m_matrix = np.ones((m,m))/m
	inverse_m_vector = np.ones((1,m))/m

	while(True):
		# Capture frame
	    ret, frame = video_capture.read()
	    # Turn to gray scal
	    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    # Detect faces
	    faces = faceCascade.detectMultiScale(grayscaled_frame, minSize=(image.HORIZONTAL_SIZE, image.VERTICAL_SIZE))
	    # Draw a rectangle around detected faces
	    for face in faces:
	        x, y, width, height = fd.resizeFace(face)
	        cv2.rectangle(frame, (x, y), (x + width, y + height), FRAME_COLOR)
	        captured_image = fd.cropImage(frame, fd.resizeFace(face))
	        captured_image = fd.resizeImg(captured_image)
	        captured_image = captured_image.convert('L')	# 'L' stands for grayscale mode
	        captured_image = np.array(captured_image).ravel()
	        captured_image = (np.array(captured_image) - image.NORMALIZE_FACTOR / 2) / (image.NORMALIZE_FACTOR / 2)
	        captured_image_K = np.dot(captured_image, images.transpose()) ** 2
        	captured_image_K = captured_image_K - np.dot(inverse_m_vector, K) - np.dot(captured_image_K, inverse_m_matrix) + np.dot(inverse_m_vector, np.dot(K, inverse_m_matrix))
	        captured_image_K = captured_image_K[0]
	        projected_image = pseudo_eigen_faces.dot(captured_image_K.transpose()).transpose()
	        name = classifier.predict([projected_image])
	        cv2.putText(frame, name[0], (x, y - TEXT_ELEVATION), fontFace=TEXT_FONT, fontScale=1, color=TEXT_COLOR, thickness=TEXT_THICKNESS)

	    # Display the complete frame
	    cv2.imshow('Video', frame)

	    # Magic
	    if cv2.waitKey(1) & 0xFF == ord(' ') and frame is not None and len(faces) > 0:
	        captured_image = fd.cropImage(frame, fd.resizeFace(faces[0]))
	        captured_image = fd.resizeImg(captured_image)

	video_capture.release()
	cv2.destroyAllWindows()