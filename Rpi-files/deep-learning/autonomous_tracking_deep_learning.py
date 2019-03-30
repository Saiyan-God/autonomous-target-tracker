# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# constants
CONFIDENCE_MIN = 0.5
DIST_THRESHOLD = 50
PROTOTXT_FILE_PATH = 'deploy.prototxt.txt'
MODEL_FILE_PATH = 'res10_300x300_ssd_iter_140000.caffemodel'

# variables
verbose_image = True
verbose_terminal = True
tracking = False
ideal_height = 150

# print statement that only prints in verbose mode
def terminal_print(text):
	if(verbose_terminal):
		print(text)

def print_commands():
	terminal_print('Commands: ')
	terminal_print('h\tPrint Commands')
	terminal_print('q\tQuit Program')
	terminal_print('v\tToggle Video Indicators')
	terminal_print('t\tToggle Tracking')
	terminal_print('y\tCycle Target\t(Cannot be tracking)')
	terminal_print('w\tMove Forward\t(Cannot be tracking)')
	terminal_print('s\tMove Backwards\t(Cannot be tracking)')
	terminal_print('a\tTurn Left\t(Cannot be tracking)')
	terminal_print('d\tToggle Right\t(Cannot be tracking)')

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

print_commands()

# get center of the frame
frame_center_x = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_center_y = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

num_faces = 0
tracking_index = 0
face_index = -1
old_size = -1
old_x = -1
old_y = -1

tracking_face_data = []
tracking_face_labels = []
counter = 0
training_data = pickle.loads(open("embeddings.pickle", "rb").read())
p_label = "positive"

embeddings_lst = training_data['embeddings']
labels_lst = training_data['names']
le = LabelEncoder()

face_embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

recognizer = SVC(C=1.0, kernel="linear", probability=True)

target_lost = False

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = vs.read()
	#frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	faces_lst = []

	target_undetected = True

    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > CONFIDENCE_MIN:
			faces_lst.append(i)

	if(num_faces != len(faces_lst)):
		tracking_index = 0
		num_faces = len(faces_lst)
		for i in range(0, num_faces):
			box = detections[0, 0, faces_lst[i], 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			x = (startX+endX)/2
			y = (startY+endY)/2
			if(abs(x - old_x) < DIST_THRESHOLD and abs(y - old_y) < DIST_THRESHOLD):
				face_index = i
				break

	for i in range(0, num_faces):
		confidence = detections[0, 0, faces_lst[i], 2]

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, faces_lst[i], 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		x = (startX+endX)/2
		y = (startY+endY)/2

		face_frame = frame[startY:endY, startX:endX]

		if(tracking and face_index == i and old_x < 0):
			old_x = x
			old_y = y

		if target_lost and len(tracking_face_data) > 5:
			faceBlob = cv2.dnn.blobFromImage(face_frame, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			face_embedder.setInput(faceBlob)
			vec = face_embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			if name == p_label:
				old_x = x
				old_y = y
				target_lost = False
				terminal_print('Target Found')

		if(tracking and not target_lost and abs(x - old_x) < DIST_THRESHOLD and abs(y - old_y) < DIST_THRESHOLD):
			target_undetected = False

			old_x = x
			old_y = y

			if(counter > 2**len(tracking_face_data)):
				faceBlob = cv2.dnn.blobFromImage(face_frame, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				face_embedder.setInput(faceBlob)
				tracking_face_data.append(face_embedder.forward().flatten())
				tracking_face_labels.append(p_label)
				if len(tracking_face_data) > 5:
					final_labels = le.fit_transform(labels_lst + tracking_face_labels)
					recognizer.fit(embeddings_lst + tracking_face_data, final_labels)
				# create an image file. Not required to do so as the training data for 
				# a particular target does not need to persist 
				#cv2.imwrite("positive_data/frame_{}.jpg".format(len(tracking_face_data)), face_frame)

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

			counter +=1

		elif (not tracking and tracking_index == i):
			# Draw a rectangle around the potential face to track
			if(verbose_image):
				cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

		else:
			# Draw a rectangle around the faces
			if(verbose_image):
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)


	if(verbose_image):
		cv2.circle(frame, (frame_center_x,frame_center_y), 3, (255, 0, 0), 2)
		tracking_text = 'Tracking: ' + ('True' if tracking else 'False')
		tracking_text_color = (0, 255, 0) if tracking else (255, 0, 0)
		cv2.putText(frame, tracking_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_text_color, 2)

	if tracking and not target_lost and target_undetected:
		target_lost = True
		terminal_print('Lost Target')

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
  
	# break from the loop
	if key == ord("h"):
		print_commands()

	# break from the loop
	if key == ord("q"):
		break

	# toggle verbose image
	if key == ord("v"):
		verbose_image = not verbose_image

	# toggle tracking mode
	if key == ord("t") and num_faces > 0:
		tracking = not tracking
		if tracking:
			face_index = tracking_index
		else:
			old_x = -1
			oly_y = -1

			target_lost = False
			tracking_face_data = []
			tracking_face_labels = []
			counter = 0

	# check next potential target
	if key == ord("y") and not tracking:
		tracking_index = tracking_index + 1 if tracking_index < num_faces - 1 else 0

	# move forward manually
	if key == ord("w") and not tracking:
		move_forward()
 
	# move backward manually
	if key == ord("s") and not tracking:
		move_backward()

	# turn left manually
	if key == ord("a") and not tracking:
		turn_left()

	# turn right manually
	if key == ord("d") and not tracking:
		turn_right()
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
