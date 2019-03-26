import cv2
from threading import Thread
class VideoCamera(object):
    def __init__(self, file_name, video_capture):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.cascPath = "./haarcascade_frontalface_default.xml"
        self.video_capture = video_capture
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.frame_center_x = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
        self.frame_center_y = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
                                                                                                                                                                                                                                                                                                                                                            
        self.old_size = 0
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        
#        fourcc = cv2.VideoWriter_fourcc(*'XVID')
#        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
#        while(self.video_capture.isOpened()):
#            ret, frame = self.video_capture.read()
#            if ret==True:
#                # write the frame
#                out.write(frame)
#
#                cv2.imshow('frame',frame)
#                if cv2.waitKey(1) & 0xFF == ord('q'):
#                    break
#            else:
#                break

        # Release everything if job is finished
        #self.video_capture.release()
        #out.release()
        #cv2.destroyAllWindows()
        
#        thread = Thread(target = threaded_function)
#        thread.start()
#        thread.join()
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(file_name,self.fourcc, 20.0, (640,480))
    
    def __del__(self):
        self.video.release()
        self.out.release()
        
    global record
    def record(video_capture, out):
        while(video_capture.isOpened()):
            ret, frame = video_capture.read()
            if ret==True:
                # write the frame
                out.write(frame)

                #cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break  
    
    def start_recording(self):
        self.recording_thread = Thread(target = record, args=(self.video_capture, self.out))
        self.recording_thread.start()
    
    
    def stop_recording(self):
        self.out.release()
        self.recording_thread.join()

    def get_frame(self):
        ret, frame = self.video_capture.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        #ret, frame = cv2.imencode('.jpg', image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        
        

        # Indicate on the frame where the center is using a blue circle
        cv2.circle(frame, (self.frame_center_x, self.frame_center_y), 3, (255, 0, 0), 2)


        for (x, y, w, h) in faces:
        
        # Draw a rectangle around the faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Keeps track of the target's center with a red circle
            cv2.circle(frame, (x+w // 2,y+h // 2), 3, (0, 0, 255), 2)

            x_difference = x + w/2 - self.frame_center_x
            size_difference = w - self.old_size

            # If the new x-coordinate and old x-coordinate difference exceeds the threshold, rotate the robot accordingly
            if(abs(x_difference) > 50):
                if(x_difference > 0):
                    print('a')
                    #r = requests.post(pi_url, data=json.dumps({'direction': 'a'}))	
            if(x_difference < 0):
                    print('d')
                    #r = requests.post(pi_url, data=json.dumps({'direction': 'd'}))
                    #print("Coordinates - x: {}, y {}".format(x,y))


            # If the new distance and old distance difference exceeds the threshold, move the robot accordingly
            if(abs(size_difference) > 20):
                self.old_size = w
                if(size_difference < 0):
                    print('w')
                    #r = requests.post(pi_url, data=json.dumps({'direction': 'w'}))
                if(size_difference > 0):
                    print('s')
                    #r = requests.post(pi_url, data=json.dumps({'direction': 's'}))
                    #print("Width, Height: {}, {}".format(w,h))
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()