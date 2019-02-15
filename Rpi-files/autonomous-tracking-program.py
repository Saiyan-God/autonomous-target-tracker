import cv2
import sys

# Program requires an xml file of classification data
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# variables used to keep track of detected object's position and distance
old_x = 0
old_size = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    for (x, y, w, h) in faces:
	
	# Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        x_difference = x - old_x
	size_difference = w - old_size

	# If the new x-coordinate and old x-coordinate difference exceeds the threshold, rotate the robot accordingly
        if(abs(x_difference) > 10):
		old_x = x	
		if(x_difference > 0):
			print("Rotate left")	
		if(x_difference < 0):
			print("Rotate right")
	        #print("Coordinates - x: {}, y {}".format(x,y))


	# If the new distance and old distance difference exceeds the threshold, move the robot accordingly
	if(abs(size_difference) > 20):
		old_size = w
		if(size_difference < 0):
			print("Move closer")
		if(size_difference > 0):
			print("Move further")
	        #print("Width, Height: {}, {}".format(w,h))

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
