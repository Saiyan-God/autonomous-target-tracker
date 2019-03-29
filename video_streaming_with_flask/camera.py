import cv2, json, time, os
from imutils.video import VideoStream
import numpy as np
import imutils, requests, json, threading

# constants
CONFIDENCE_MIN = 0.5
DIST_THRESHOLD = 50
PROTOTXT_FILE_PATH =  os.path.abspath('deploy.prototxt.txt')
MODEL_FILE_PATH = os.path.abspath('res10_300x300_ssd_iter_140000.caffemodel')

# variables
verbose_image = True
verbose_terminal = True
tracking = False
ideal_height = 150

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

# print statement that only prints in verbose mode
def terminal_print(text):
	if(verbose_terminal):
	    print(text)


video_counter = 0
class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        self.fourcc = cv2.VideoWriter_fourcc('H','2','6','4')  # or self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.out = None

    def run(self):
        global video_counter
        video_counter += 1
        video_path = './recordings/video' + str(video_counter) + '.mp4'
        print(video_path)
        self.out = cv2.VideoWriter(video_path, 0x31637661, 20.0, (640,480))  # or replace 0x31637661 with self.fourcc
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
    import requests
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video_capture = cv2.VideoCapture(0)
        #self.video_capture = cv2.VideoCapture('udpsrc port=5200 !  application/x-rtp, encoding-name=JPEG,payload=26 !  rtpjpegdepay !  jpegdec ! videoconvert ! appsink')
        time.sleep(2.0)
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE_PATH, MODEL_FILE_PATH)
        self.frame_center_x = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        self.frame_center_y = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
        self.pi_url = "http://192.168.43.253"
        self.num_faces = 0
        self.face_index = -1
        self.old_x = -1
        self.old_y = -1
        self.tracking_index = 0

        self.old_size = -1
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.is_record = False
        self.out = None

        self.recordingThread = None

    def __del__(self):
        self.video.release()

    def move_forward(self):
        terminal_print('Move forward')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 'w'}))
	    # TO-DO: seriarecordingThreadrecordingThreadrecordingThreadrecordingThreadrecordingThreadrecordingThreadl code to move robot forward

    def move_backward(self):
        terminal_print('Move backward')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 's'}))
        # TO-DO: serial code to move robot backward

    def turn_right(self):
        terminal_print('Turn right')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 'a'}))
        # TO-DO: serial code to turn robot right

    def stop(self):
        terminal_print('Stop')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 'x'}))
        # TO-DO: serial code to turn robot right

    def turn_left(self):
        terminal_print('Turn left')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 'd'}))
        # TO-DO: serial code to turn robot left

    def get_frame(self):
        ret, frame = self.video_capture.read()
        frame = cv2.resize(frame, (680, 480))
       # frame = imutils.resize(frame, width=640, height=480)

        if(verbose_image):
            cv2.circle(frame, (self.frame_center_x,self.frame_center_y), 3, (255, 0, 0), 2)
            tracking_text = 'Tracking: ' + ('True' if tracking else 'False')
            tracking_text_color = (0, 255, 0) if tracking else (255, 0, 0)
            cv2.putText(frame, tracking_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_text_color, 2)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

        faces_lst = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > CONFIDENCE_MIN:
                faces_lst.append(i)

        if(self.num_faces != len(faces_lst)):
            self.tracking_index = 0
            self.num_faces = len(faces_lst)
            for i in range(0, self.num_faces):
                box = detections[0, 0, faces_lst[i], 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                x = (startX+endX)/2
                y = (startY+endY)/2
                if(abs(x - self.old_x) < DIST_THRESHOLD and abs(y - self.old_y) < DIST_THRESHOLD):
                    self.face_index = i
                    break

        for i in range(0, self.num_faces):
            confidence = detections[0, 0, faces_lst[i], 2]

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, faces_lst[i], 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            x = (startX+endX)/2
            y = (startY+endY)/2

            if(tracking and self.face_index == i and self.old_x < 0):
                self.old_x = x
                self.old_y = y

            if(tracking and abs(x - self.old_x) < DIST_THRESHOLD and abs(y - self.old_y) < DIST_THRESHOLD):
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
                        self.turn_left()
                    if(x_diff < 0):
                        self.turn_right()

                # If the new distance and old distance difference exceeds the threshold, move the robot accordingly
                if(abs(size_difference) > 20):
                    if(size_difference < 0):
                        self.move_forward()
                    if(size_difference > 0):
                        self.move_backward()
            elif (not tracking and self.tracking_index == i):
                # Draw a rectangle around the potential face to track
                if(verbose_image):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            else:
                # Draw a rectangle around the faces
                if(verbose_image):
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
