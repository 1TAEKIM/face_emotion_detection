from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
from io import BytesIO
from PIL import Image
from keras.models import model_from_json
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


emotion_detected = False  
selected_emotion = None  


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def generate_frames():
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)

            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frame, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # 감정 분석 결과 전송
            socketio.emit('update_frame', {'frame': f"data:image/jpeg;base64,{frame_encoded}", 'emotion': prediction_label})

            yield f"data:image/jpeg;base64,{frame_encoded}"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit('response', {'data': 'Connected'})

# 이벤트 핸들러 함수 정의
@socketio.on('video_feed')
def handle_video_feed(data):
    print("Video feed requested")
    video_thread = VideoThread(socketio)
    video_thread.start()

class VideoThread(Thread):
    def __init__(self, socketio):
        Thread.__init__(self)
        self.socketio = socketio
        self.prediction_label = ""

    def run(self):
        webcam = cv2.VideoCapture(0)

        while True:
            success, frame = webcam.read()
            if not success:
                break

            # 프레임 처리 및 감정 분석 코드 추가
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)

            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                self.prediction_label = labels[pred.argmax()]
                cv2.putText(frame, '% s' % (self.prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # 프레임 및 감정 결과 전송
            self.socketio.emit('update_frame', {'frame': f"data:image/jpeg;base64,{frame_encoded}", 'emotion': self.prediction_label})



if __name__ == '__main__':
    socketio.run(app, debug=True)