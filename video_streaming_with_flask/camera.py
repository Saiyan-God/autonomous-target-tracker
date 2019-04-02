import cv2, json, time, os
from imutils.video import VideoStream
import numpy as np
import imutils, requests, json, threading

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# constants
CONFIDENCE_MIN = 0.5
DIST_THRESHOLD = 50
PROTOTXT_FILE_PATH =  os.path.abspath('deploy.prototxt.txt')
MODEL_FILE_PATH = os.path.abspath('res10_300x300_ssd_iter_140000.caffemodel')

# variables
verbose_terminal = True




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
        #self.video_capture = cv2.VideoCapture(0)
        self.video_capture = cv2.VideoCapture('udpsrc port=5200 !  application/x-rtp, encoding-name=JPEG,payload=26 !  rtpjpegdepay !  jpegdec ! videoconvert ! appsink')
        time.sleep(2.0)
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE_PATH, MODEL_FILE_PATH)
        self.frame_center_x = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        self.frame_center_y = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
        self.pi_url = "http://172.17.38.242"
        self.num_faces = 0
        self.face_index = -1
        self.old_x = -1
        self.old_y = -1 
        self.tracking_index = 0

        self.old_size = -1

        self.tracking = False
        self.verbose_image = True

        self.tracking_face_data = []
        self.tracking_face_labels = []
        self.counter = 0
        self.training_data = pickle.loads(open("embeddings.pickle", "rb").read())
        self.p_label = "positive"

        self.embeddings_lst = self.training_data['embeddings']
        self.labels_lst = self.training_data['names']
        self.le = LabelEncoder()

        self.face_embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

        self.recognizer = SVC(C=1.0, kernel="linear", probability=True)

        self.target_lost = False


        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.is_record = False
        self.out = None
        self.ideal_height = 150
        self.recordingThread = None

    def __del__(self):
        self.video.release()
    
    def toogle_tracking(self):
        self.tracking = not self.tracking
        if self.tracking:
			self.face_index = self.tracking_index
        else:
			self.old_x = -1
			self.oly_y = -1

			self.target_lost = False
			self.tracking_face_data = []
			self.tracking_face_labels = []
			self.counter = 0

    def toogle_verbose_video(self):
        self.verbose_image = not self.verbose_image
    
    def track_next_target(self):
        if not self.tracking:
            self.tracking_index = self.tracking_index + 1 if self.tracking_index < self.num_faces - 1 else 0

    def move_forward(self):
        terminal_print('Move forward')
        r = requests.post(self.pi_url, data=json.dumps({'direction': 'w'}))

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

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        self.net.setInput(blob)
        detections = self.net.forward()

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

            face_frame = frame[startY:endY, startX:endX]

            if(self.tracking and self.face_index == i and self.old_x < 0):
                self.old_x = x
                self.old_y = y

            if self.target_lost and len(self.tracking_face_data) > 5:
                try:
                    faceBlob = cv2.dnn.blobFromImage(face_frame, 1.0 / 255, (96, 96),
                        (0, 0, 0), swapRB=True, crop=False)
                    self.face_embedder.setInput(faceBlob)
                    vec = self.face_embedder.forward()

                    preds = self.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = self.le.classes_[j]

                    if name == self.p_label:
                        self.old_x = x
                        self.old_y = y
                        self.target_lost = False
                        terminal_print('Target Found')
                except:
                    continue

            if(self.tracking and not self.target_lost and abs(x - self.old_x) < DIST_THRESHOLD and abs(y - self.old_y) < DIST_THRESHOLD):
                target_undetected = False
                self.old_x = x
                self.old_y = y


                if(self.counter > 1.5**len(self.tracking_face_data)):
                    try:
                        faceBlob = cv2.dnn.blobFromImage(face_frame, 1.0 / 255,
                            (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        self.face_embedder.setInput(faceBlob)
                        self.tracking_face_data.append(self.face_embedder.forward().flatten())
                        self.tracking_face_labels.append(self.p_label)
                        if len(self.tracking_face_data) > 5:
                            final_labels = self.le.fit_transform(self.labels_lst + self.tracking_face_labels)
                            self.recognizer.fit(self.embeddings_lst + self.tracking_face_data, final_labels)
                        # create an image file. Not required to do so as the training data for 
                        # a particular target does not need to persist 
                        #cv2.imwrite("positive_data/frame_{}.jpg".format(len(tracking_face_data)), face_frame)
                    except:
                        continue

                # draw the bounding box of the face along with the associated
                # probability
                if(self.verbose_image):
                    textY = startY - 10 if startY - 10 > 10 else startY + 10
                    text = "{:.2f}%".format(confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (x,y), 3, (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                x_diff = x - self.frame_center_x
                size_difference = abs(startY - endY) - self.ideal_height

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

                self.counter += 1

            elif (not self.tracking and self.tracking_index == i):
                # Draw a rectangle around the potential face to track
                if(self.verbose_image):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            else:
                # Draw a rectangle around the faces
                if(self.verbose_image):
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        if(self.verbose_image):
            cv2.circle(frame, (self.frame_center_x,self.frame_center_y), 3, (255, 0, 0), 2)
            tracking_text = 'Tracking: ' + ('True' if self.tracking else 'False')
            tracking_text_color = (0, 255, 0) if self.tracking else (255, 0, 0)
            cv2.putText(frame, tracking_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_text_color, 2)

        if self.tracking and not self.target_lost and target_undetected:
            self.target_lost = True
            terminal_print('Lost Target')

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
