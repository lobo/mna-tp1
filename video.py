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

def recognize_faces(mean_face, eigen_faces, classifier):
	faceCascade = cv2.CascadeClassifier(HAAR_CASCADE_FRONTAL_FACE_PATH)
	video_capture = cv2.VideoCapture(0)
	
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
	        new_image = fd.cropImage(frame, fd.resizeFace(face))
	        new_image = fd.resizeImg(new_image)
	        new_image = new_image.convert('L')	# 'L' stands for grayscale mode
	        new_image = np.array(new_image).ravel()
	        new_image = (np.array(new_image) / image.NORMALIZE_FACTOR) - mean_face
	        new_image = np.dot(np.array(new_image), eigen_faces.transpose())
	        name = classifier.predict([new_image])
	        cv2.putText(frame, name[0], (x, y - TEXT_ELEVATION), fontFace=TEXT_FONT, fontScale=1, color=TEXT_COLOR, thickness=TEXT_THICKNESS)

	    # Display the complete frame
	    cv2.imshow('Video', frame)

	    # Magic
	    if cv2.waitKey(1) & 0xFF == ord(' ') and frame is not None and len(faces) > 0:
	        new_image = fd.cropImage(frame, fd.resizeFace(faces[0]))
	        new_image = fd.resizeImg(new_image)

	video_capture.release()
	cv2.destroyAllWindows()