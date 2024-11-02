import os
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model

model = load_model('emotion_detection_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

emotion_to_tag = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'rock',
    'neutral': 'chill',
    'surprise': 'electronic',
    'fear': 'ambient',
    'disgust': 'classical'
}

LASTFM_API_KEY = '0b155c6c808f49437ee27d72f2e94790'
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"


def get_music_recommendations_by_tag(tag):
    try:
        params = {
            'method': 'tag.gettoptracks',
            'tag': tag,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': 5
        }
        response = requests.get(LASTFM_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        tracks = [(track['name'], track['artist']['name']) for track in data['tracks']['track']]
        return tracks
    except requests.exceptions.RequestException as e:
        print("Last.fm API Error:", e)
        return []

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0 
        roi_reshaped = np.reshape(roi_resized, (1, 48, 48, 1))

        prediction = model.predict(roi_reshaped)
        emotion = emotion_labels[np.argmax(prediction)]

        if emotion != last_emotion:
            print(f"Detected emotion: {emotion}")
            last_emotion = emotion
            tag = emotion_to_tag.get(emotion, 'pop') 
            recommended_tracks = get_music_recommendations_by_tag(tag)

            print(f"Recommended tracks for {emotion} mood:")
            for track in recommended_tracks:
                print(f"{track[0]} by {track[1]}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
