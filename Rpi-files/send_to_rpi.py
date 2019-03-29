import cv2 as cv2
import sys, json, requests

# Program requires an xml file of classification data
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
pi_url = "http://192.168.0.155"
#video_capture = cap_receive = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#video_capture = cv2.VideoCapture('udpsrc port=5002 ! application/x-rtp, payload=96 ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
video_capture = cv2.VideoCapture('udpsrc port=5200 !  application/x-rtp, encoding-name=JPEG,payload=26 !  rtpjpegdepay !  jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
frame_center_x = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_center_y = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)


# variables used to keep track of detected object's distance
old_size = 0

def receive():
   # cap_receive = cv2.VideoCapture('C:\\gstreamer\\1.0\\x86_64\\bin\\xyz.flv')
    cap_receive = cv2.VideoCapture('autovideosrc !  videoconvert ! appsink', cv2.CAP_GSTREAMER)
    if not cap_receive.isOpened():
        print('VideoCapture not opened')
        exit(0)

    while True:
        ret,frame = cap_receive.read()

        if not ret:
            print('empty frame')
            break

        cv2.imshow('receive', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            print('Here')
            break

    cap_receive.release()



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


    for (x, y, w, h) in faces:
	
	# Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Keeps track of the target's center with a red circle
        cv2.circle(frame, (x+w // 2,y+h // 2), 3, (0, 0, 255), 2)
s
        x_difference = x + w/2 - frame_center_x
        size_difference = w - old_size

        # If the new x-coordinate and old x-coordinate difference exceeds the threshold, rotate the robot accordingly
        if(abs(x_difference) > 50):
            if(x_difference > 0):
                r = requests.post(pi_url, data=json.dumps({'direction': 'a'}))	
        if(x_difference < 0):
                r = requests.post(pi_url, data=json.dumps({'direction': 'd'}))
                #print("Coordinates - x: {}, y {}".format(x,y))


        # If the new distance and old distance difference exceeds the threshold, move the robot accordingly
        if(abs(size_difference) > 20):
            old_size = w
            if(size_difference < 0):
                r = requests.post(pi_url, data=json.dumps({'direction': 'w'}))
            if(size_difference > 0):
                r = requests.post(pi_url, data=json.dumps({'direction': 's'}))
                #print("Width, Height: {}, {}".format(w,h))

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#receive()
# When everything is done, release the capture
#video_capture.release()
cv2.destroyAllWindows()
