import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
video_path = "./video/goggles-&-stash.mp4"

fps = 24

face_cascade = cv.CascadeClassifier("cascade/haarcascade_frontalface_alt.xml")
eyes_cascade = cv.CascadeClassifier("cascade/haarcascade_eye.xml")
nose_cascade = cv.CascadeClassifier("cascade/Nose18x15.xml")

glasses      = cv.imread("props/glasses.png, -1")
mustache     = cv.imread("props/mustache.png, -1") #-1 is because both images having trasparent background"

# Capture Video
while True:
	return_val, frame = capture.read() #Capture Frame by Frame
	if return_val == False:
		continue

	cv.imshow("Frame", frame)
	key_pressed = cv.waitKey(1) & 0xFF
	if key_pressed == ord('q'): # ord('q') gives ASCII value of q
		break

capture.release()
cv.destroyAllWindows()