# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

# constants
CONFIDENCE_MIN = 0.5
X_DIST_THRESHOLD = 50
PROTOTXT_FILE_PATH = 'deploy.prototxt.txt'
MODEL_FILE_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'

# variables
verbose_image = True
verbose_terminal = True
ideal_height = 150

# print statement that only prints in verbose mode
def terminal_print(text):
	if(verbose_terminal):
		print(text)

def move_forward():
	terminal_print('Move forward')
	# TO-DO: serial code to move robot forward

def move_backward():
	terminal_print('Move backward')
	# TO-DO: serial code to move robot backward

def turn_right():
	terminal_print('Turn right')
	# TO-DO: serial code to turn robot right

def turn_left():
	terminal_print('Turn left')
	# TO-DO: serial code to turn robot left

# load our serialized model from disk
terminal_print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE_PATH, MODEL_FILE_PATH)
 
# initialize the video stream and allow the camera sensor to warm up
terminal_print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# get center of the frame
frame_center_x = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_center_y = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

num_faces = 0
face_index = -1
old_size = -1
old_x = -1
old_y = -1

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = vs.read()
	#frame = imutils.resize(frame, width=400)

	if(verbose_image):
		cv2.circle(frame, (frame_center_x,frame_center_y), 3, (255, 0, 0), 2)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()


    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < CONFIDENCE_MIN:
			continue
 
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		x = (startX+endX)/2
		y = (startY+endY)/2
		
		if(old_x < 1):
			old_x = x
			old_y = y

		if(abs(x - old_x) < X_DIST_THRESHOLD):
			old_x = x
			old_y = y

			# draw the bounding box of the face along with the associated
			# probability
			if(verbose_image):
				textY = startY - 10 if startY - 10 > 10 else startY + 10
				text = "{:.2f}%".format(confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				cv2.circle(frame, (x,y), 3, (0, 0, 255), 2)
				cv2.putText(frame, text, (startX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

			x_diff = x - frame_center_x
			size_difference = abs(startY - endY) - ideal_height

			# If the new x-coordinate and old x-coordinate difference exceeds the threshold, rotate the robot accordingly
			if(abs(x_diff) > 130):
				if(x_diff > 0):
					turn_left()
				if(x_diff < 0):
					turn_right()
			
			# If the new distance and old distance difference exceeds the threshold, move the robot accordingly
			if(abs(size_difference) > 20):
				if(size_difference < 0):
					move_forward()	
				if(size_difference > 0):
					move_backward()

		else:
			# Draw a rectangle around the faces
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()