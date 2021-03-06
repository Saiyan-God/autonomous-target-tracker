import cv2
import sys
#import serial
import time 

#ser = serial.Serial('/dev/ttyUSB0', 9600)

# Program requires an xml file of classification data
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

frame_center_x = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_center_y = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

old_x = -1
old_y = -1


center_width = 150


# variables used to keep track of detected object's distance
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
    

    # Indicate on the frame where the center is using a blue circle
    cv2.circle(frame, (frame_center_x,frame_center_y), 3, (255, 0, 0), 2)

    time.sleep(0.050)

    for (x, y, w, h) in faces:

	if(old_x < 0):
		old_x = x
		old_y = y

	if(len(faces) == 1 or abs(x - old_x) < 20):

		old_x = x
		oly_y = y

		# Draw a rectangle around the faces
       		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# Keeps track of the target's center with a red circle
		cv2.circle(frame, (x+w/2,y+h/2), 3, (0, 0, 255), 2)

	        x_difference = x + w/2 - frame_center_x
		size_difference = w - old_size
		size_difference = w - center_width

		# If the new x-coordinate and old x-coordinate difference exceeds the threshold, rotate the robot accordingly
	        if(abs(x_difference) > 130):
			if(x_difference > 0):
				print("Rotate left")
				#ser.write(b'd')
#				ser.write(b'\n')	
			if(x_difference < 0):
				print("Rotate right")
				#ser.write(b'a')
#				ser.write(b'\n')	
		        #print("Coordinates - x: {}, y {}".format(x,y))


		# If the new distance and old distance difference exceeds the threshold, move the robot accordingly
		if(abs(size_difference) > 10):
		#if(abs(size_difference)*10/(old_size+1) > 1):
			old_size = w
			if(size_difference < 0):
				print("Move closer")
#				ser.write(b'w')
#				ser.write(b'\n')	
			if(size_difference > 0):
				print("Move further")
#				ser.write(b's')
#				ser.write(b'\n')	
		        #print("Width, Height: {}, {}".format(w,h))

	else:
		# Draw a rectangle around the faces
       		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
