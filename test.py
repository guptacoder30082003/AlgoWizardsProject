import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time
from threading import Thread, Lock

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Confused', 'Surprise']
VIDEO_FILE = 'sample.mp4'
current_speed = 1.0
MIN_SPEED = 0.1
MAX_SPEED = 1.0
SPEED_STEP = 0.1
last_emotion_check = 0
EMOTION_CHECK_INTERVAL = 2.0

webcam = cv2.VideoCapture(0)
video = cv2.VideoCapture(VIDEO_FILE)
video_fps = video.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / video_fps)
paused = False
exit_flag = False
DETECTION_CONFIDENCE_THRESHOLD = 0.5

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

speed_lock = Lock()
display_speed = 1.0
last_emotion = ""

def get_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: (f[2]*f[3]), reverse=True)
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        if np.sum([roi_gray]) > 100:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi, verbose=0)[0]
            max_pred = np.max(preds)
            if max_pred > DETECTION_CONFIDENCE_THRESHOLD:
                emotion = class_labels[preds.argmax()]
                if emotion == 'Neutral' and max_pred < 0.7:
                    return 'Confused', max_pred, (x, y, w, h)
                return emotion, max_pred, (x, y, w, h)
    return None, 0, None

def video_player():
    global current_speed, paused, exit_flag, display_speed, last_emotion_check
    while not exit_flag:
        if not paused:
            ret, vid_frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            with speed_lock:
                vid_frame = cv2.resize(vid_frame, (640, 360))
                vid_frame_display = vid_frame.copy()
                next_check = max(0, EMOTION_CHECK_INTERVAL - (time.time() - last_emotion_check))
                cv2.putText(vid_frame_display, f"Speed: {display_speed:.1f}x", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vid_frame_display, f"Next check: {next_check:.1f}s", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Video Player', vid_frame_display)
                
                delay = max(1, int(frame_delay / current_speed))
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord(' '):
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('q'):
                exit_flag = True
        else:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('q'):
                exit_flag = True

player_thread = Thread(target=video_player)
player_thread.start()

try:
    cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
    recent_face_coords = None
    recent_emotion = ""
    recent_confidence = 0
    consecutive_confused = 0

    while not exit_flag:
        ret, frame = webcam.read()
        if not ret:
            break

        current_time = time.time()
        info_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: (f[2]*f[3]), reverse=True)
            x, y, w, h = faces[0]
            recent_face_coords = (x, y, w, h)
            
            if current_time - last_emotion_check >= EMOTION_CHECK_INTERVAL:
                emotion, confidence, _ = get_emotion(frame)
                last_emotion_check = current_time
                
                if emotion:
                    if emotion != recent_emotion or confidence != recent_confidence:
                        print(f"Emotion changed: {emotion} (Confidence: {confidence*100:.1f}%)")
                    
                    recent_emotion = emotion
                    recent_confidence = confidence

                    with speed_lock:
                        if emotion == 'Confused' and confidence > 0.5:
                            consecutive_confused = min(5, consecutive_confused + 1)
                            current_speed = max(MIN_SPEED, 1.0 - (consecutive_confused * SPEED_STEP))
                            print(f"Confused detected - Adjusting speed to {current_speed:.1f}x")
                        else:
                            if consecutive_confused > 0:
                                print("Resetting to normal speed")
                            consecutive_confused = 0
                            current_speed = MAX_SPEED
                        display_speed = current_speed

        if recent_face_coords:
            x, y, w, h = recent_face_coords
            cv2.rectangle(info_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if recent_emotion:
                cv2.putText(info_frame, f"{recent_emotion} ({recent_confidence*100:.1f}%)", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        next_check = max(0, EMOTION_CHECK_INTERVAL - (current_time - last_emotion_check))
        cv2.putText(info_frame, f"Speed: {display_speed:.1f}x", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_frame, f"Next check: {next_check:.1f}s", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Emotion Detection', info_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            exit_flag = True
            print("Exiting program...")
            break

finally:
    exit_flag = True
    player_thread.join()
    webcam.release()
    video.release()
    cv2.destroyAllWindows()