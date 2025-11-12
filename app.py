import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import cv2
from deepface import DeepFace

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None, image_url=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save uploaded image
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"upload_{ts}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(file.stream).convert('RGB')
    img.save(filepath, format='JPEG')

    # Read image
    cv_img = cv2.imread(filepath)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    result = "No face detected."

    for (x, y, w, h) in faces:
        face_crop = cv_img[y:y + h, x:x + w]

        # Emotion detection
        analysis = DeepFace.analyze(
            face_crop, actions=['emotion'], enforce_detection=False
        )
        emotion = analysis[0]['dominant_emotion']

        if emotion.lower() == 'happy':
            result = "😊 The person looks happy!"
        elif emotion.lower() == 'sad':
            result = "😢 The person looks sad!"
        else:
            result = f"The detected emotion is {emotion}."

        # Draw box and label
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(cv_img, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
        break  # check only first face

    # Save output image
    out_name = f"result_{ts}.jpg"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    cv2.imwrite(out_path, cv_img)

    return render_template(
        'index.html',
        result=result,
        image_url=url_for('static', filename=f'uploads/{out_name}')
    )

if __name__ == '__main__':
    app.run(debug=True)