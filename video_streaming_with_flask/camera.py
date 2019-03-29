import cv2, requests, json, time, os
from imutils.video import VideoStream
import numpy as np
import imutils, requests, json, threading

# constants
CONFIDENCE_MIN = 0.5
X_DIST_THRESHOLD = 50
PROTOTXT_FILE_PATH =  os.path.abspath('deploy.prototxt.txt')
MODEL_FILE_PATH = os.path.abspath('res10_300x300_ssd_iter_140000.caffemodel')

# variables
verbose_image = True
verbose_terminal = True
ideal_height = 150
pi_url = "http://192.168.43.253"

# print statement that only prints in verbose mode
def terminal_print(text):
	if(verbose_terminal):
	    print(text)

def move_forward():
    terminal_print('Move forward')
    r = requests.post(pi_url, data=json.dumps({'direction': 'w'}))
	# TO-DO: serial code to move robot forward

def move_backward():
    terminal_print('Move backward')
    r = requests.post(pi_url, data=json.dumps({'direction': 's'}))
    # TO-DO: serial code to move robot backward

def turn_right():
    terminal_print('Turn right')
    r = requests.post(pi_url, data=json.dumps({'direction': 'a'}))
	# TO-DO: serial code to turn robot right

def turn_left():
    terminal_print('Turn left')
    r = requests.post(pi_url, data=json.dumps({'direction': 'd'}))
	# TO-DO: serial code to turn robot left


video_counter = 0
class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = None  # cv2.VideoWriter('./recordings/video.avi', self.fourcc, 20.0, (640,480));

    def run(self):
        global video_counter
        video_counter += 1
        video_path = './recordings/video' + str(video_counter) + '.avi'
        print(video_path)
        self.out = cv2.VideoWriter(video_path, self.fourcc, 20.0, (640,480))
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video_capture = cv2.VideoCapture('udpsrc port=5200 !  application/x-rtp, encoding-name=JPEG,payload=26 !  rtpjpegdepay !  jpegdec ! videoconvert ! appsink')
        time.sleep(2.0)
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE_PATH, MODEL_FILE_PATH)
        self.frame_center_x = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        self.frame_center_y = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
        self.pi_url = "http://192.168.0.155"
        self.num_faces = 0
        face_index = -1
        self.old_x = -1
        self.old_y = -1
                                                                                                                                                                                                                                                                                                                                                            
        self.old_size = -1
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.is_record = False
        self.out = None

        self.recordingThread = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video_capture.read()
        #frame = imutils.resize(frame, width=400)

        if(verbose_image):
            cv2.circle(frame, (self.frame_center_x,self.frame_center_y), 3, (255, 0, 0), 2)
    
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    
        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()


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
            
            if(self.old_x < 1):
                self.old_x = x
                self.old_y = y

            if(abs(x - self.old_x) < X_DIST_THRESHOLD):
                self.old_x = x
                self.old_y = y

                # draw the bounding box of the face along with the associated
                # probability
                if(verbose_image):
                    textY = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "{:.2f}%".format(confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (x,y), 3, (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                x_diff = x - self.frame_center_x
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
    

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.video_capture)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False
        if self.recordingThread != None:
            self.recordingThread.stop()
