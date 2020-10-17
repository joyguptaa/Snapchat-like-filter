import cv2 as cv
import numpy as np
from utility import image_resize
capture = cv.VideoCapture(0)
#video_path = "./video/goggles-&-stash.mp4"  # Lastly we save the video stream
#fps = 24

face_cascade = cv.CascadeClassifier("cascade/haarcascade_frontalface_alt.xml")
eyes_cascade = cv.CascadeClassifier("cascade/haarcascade_eye.xml")
nose_cascade = cv.CascadeClassifier("cascade/Nose18x15.xml")

glasses      = cv.imread("props/glasses.png", -1)
mustache     = cv.imread("props/mustache.png", -1) #-1 is because both images having trasparent background"

# Capture Video
while True:
	return_val, frame = capture.read()
	if return_val == False:
		continue

	gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Grayscale Image => better for computations
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

	frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA) # To work with transparent background images

	for (x, y, w, h) in faces:
		roi_gray  =  gray[y : y + h, x : x + w] # Region Of Interest in grayscale
		roi_color = frame[y : y + h, x : x + w]# Region Of Interest in colored image
		#cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3) #Drawing Recatngle

		# DETECT EYES 
		eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor = 1.5, minNeighbors = 5)
		for (ex, ey, ew, eh) in eyes:
			#cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3) #Drawing Recatngle
			roi_eyes = roi_gray[ey : ey + eh, ex : ex + ew]
			glasses2 = image_resize(glasses.copy(), width = ew)

			gw, gh, gc = glasses2.shape

			for i in range(0, gw):
				for j in range(0, gh):

					# If it is not transparent
					if glasses2[i,j][3] != 0: 
						roi_color[ey + i, ex + j] = glasses2[i,j]
			
		# DETECT NOSE 
		nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor = 1.5, minNeighbors = 5)
		for (nx, ny, nw, nh) in nose:
			#cv.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3) #Drawing Recatngle
			roi_nose = roi_gray[ny : ny + nh, nx : nx + nw]
			mustache2 = image_resize(mustache.copy(), width = nw)

			mw, mh, mc = mustache2.shape

			for i in range(0, mw):
				for j in range(0, mh):

					# If it is not transparent
					if mustache2[i,j][3] != 0: 
						roi_color[ny + i + 20, nx + j] = mustache2[i,j]

	cv.imshow("Frame", frame)
	key_pressed = cv.waitKey(1) & 0xFF
	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break

capture.release()
cv.destroyAllWindows()