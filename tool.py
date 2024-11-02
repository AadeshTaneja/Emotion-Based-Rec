import os
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model

# Load the trained emotion detection model
model = load_model('emotion_detection_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Emotion-to-genre/tag mapping
emotion_to_tag = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'rock',
    'neutral': 'chill',
    'surprise': 'electronic',
    'fear': 'ambient',
    'disgust': 'classical'
}

# Last.fm API setup
LASTFM_API_KEY = '0b155c6c808f49437ee27d72f2e94790'
LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

# Function to get music recommendations from Last.fm based on mood tag
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
        response.raise_for_status()  # Check for request errors
        data = response.json()
        
        # Extract track names and artists
        tracks = [(track['name'], track['artist']['name']) for track in data['tracks']['track']]
        return tracks
    except requests.exceptions.RequestException as e:
        print("Last.fm API Error:", e)
        return []

# Initialize the webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI and prepare for emotion prediction
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0  # Normalize
        roi_reshaped = np.reshape(roi_resized, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(roi_reshaped)
        emotion = emotion_labels[np.argmax(prediction)]

        # Fetch new recommendations if emotion changes
        if emotion != last_emotion:
            print(f"Detected emotion: {emotion}")
            last_emotion = emotion
            tag = emotion_to_tag.get(emotion, 'pop')  # Default to 'pop' if no mapping
            recommended_tracks = get_music_recommendations_by_tag(tag)

            # Display recommended tracks based on detected mood
            print(f"Recommended tracks for {emotion} mood:")
            for track in recommended_tracks:
                print(f"{track[0]} by {track[1]}")

        # Display emotion on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with predictions
    cv2.imshow('Emotion Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
