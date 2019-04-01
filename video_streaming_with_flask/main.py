#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response, jsonify, request, send_file, make_response
from flask_cors import CORS, cross_origin
from camera import VideoCamera
import time
import os, os.path
import re
from flask_socketio import SocketIO, emit


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

video_camera = None
global_frame = None

socketio = SocketIO(app)

def instansiate_camera():
    return VideoCamera()

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/record_status', methods=['POST'])
@cross_origin()
def record_status():
    global video_camera
    if video_camera == None:
        video_camera = instansiate_camera()

    json = request.get_json()

    status = json['status']
    if status == "true":
        video_camera.start_record()
        response = jsonify(result="started")
        return response
    else:
        video_camera.stop_record()
        response = jsonify(result="stopped")
        return response



def video_stream():
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = instansiate_camera()

    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings')
def recordings():
    DIR = './recordings'
    num_videos = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    return jsonify({"num_videos": num_videos})

@app.route('/toggle_tracking')
def toggle_tracking():
    global video_camera

    if video_camera == None:
        video_camera = instansiate_camera() 

    video_camera.toogle_tracking()

    return jsonify({"success": True, "current_target_index": video_camera.tracking_index})

@app.route('/track_next_target')
def track_next_target():
    global video_camera

    if video_camera == None:
        video_camera = instansiate_camera()   

    video_camera.track_next_target()
    return jsonify({"success": True, "current_target_index": video_camera.tracking_index})

@app.route('/recording', methods=['GET'])
@cross_origin()
def recording():
    try:
        recording_no = request.args.get('recording_no')
    except:
        recording_no = 1
    if recording_no is None:
        recording_no = 1
    print(recording_no)
    vid_path = './recordings/video{}.mp4'.format(recording_no)
    print(vid_path)
    file_size = os.stat(vid_path).st_size
    start = 0
    length = 10240  # can be any default length you want

    range_header = request.headers.get('Range', None)
    if range_header:
        m = re.search('([0-9]+)-([0-9]*)', range_header)  # example: 0-1000 or 1250-
        g = m.groups()
        byte1, byte2 = 0, None
        if g[0]:
            byte1 = int(g[0])
        if g[1]:
            byte2 = int(g[1])
        if byte1 < file_size:
            start = byte1
        if byte2:
            length = byte2 + 1 - byte1
        else:
            length = file_size - start

    with open(vid_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)

    rv = Response(chunk, 206, mimetype='video/mp4', content_type='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    return rv
    #return Response(g, direct_passthrough=True)


@socketio.on('move')
def test_message(message):
    global video_camera

    if video_camera == None:
        video_camera = instansiate_camera()
    print "direction: " + message
    if message == 'forward': video_camera.move_forward()
    if message == 'backward': video_camera.move_backward()
    if message == 'left': video_camera.turn_left()
    if message == 'right': video_camera.turn_right()
    if message ==  'stop': video_camera.stop()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
    socketio.run(app)
