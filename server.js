const express = require('express');
const http = require('http');
const socketIO = require('socket.io');
const cv = require('opencv4nodejs');

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

const json_file = require('fs').readFileSync("facialemotionmodel.json", "utf8");
const model_json = JSON.parse(json_file);
const model = new cv.Net();
model.readModelFromJSON(JSON.stringify(model_json));
model.readWeight("facialemotionmodel.weights");

const haar_file = cv.data.haarcascades + 'haarcascade_frontalface_default.xml';
const face_cascade = new cv.CascadeClassifier(haar_file);

const labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'};

const extractFeatures = (image) => {
    const feature = new cv.Mat(image);
    feature.resize(48, 48);
    feature.convertTo(feature, cv.CV_32F);
    feature.div(255.0);
    return feature.reshape(1, 48, 48, 1);
};

const analyzeEmotion = (image) => {
    const gray = image.cvtColor(cv.COLOR_BGR2GRAY);
    const faces = face_cascade.detectMultiScale(gray, 1.3, 5);

    try {
        for (const {x, y, width, height} of faces) {
            const faceImage = gray.getRegion(new cv.Rect(x, y, width, height));
            faceImage.resize(48, 48);
            const img = extractFeatures(faceImage);
            const pred = model.predict(img);
            const emotionLabel = labels[pred.argmax()];
            return emotionLabel;
        }
    } catch (e) {
        console.error(`Error during emotion analysis: ${e}`);
        return null;
    }
};

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

io.on('connection', (socket) => {
    console.log('Client connected');

    const webcam = new cv.VideoCapture(0);

    setInterval(() => {
        const frame = webcam.read();
        const frameEncoded = cv.imencode('.jpg', frame).toString('base64');
        const emotion = analyzeEmotion(frame);
        socket.emit('update_frame', {frame: frameEncoded, emotion: emotion});
    }, 1000 / 30); // 30 frames per second
});

server.listen(3000, () => {
    console.log('Server is running on port 3000');
});
