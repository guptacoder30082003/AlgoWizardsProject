# Emotion Based Video Speed Controller

This project uses a webcam to detect the user's emotions and controls the speed of a video playback based on the emotion detected.

If the user looks **confused**, the video will automatically **slow down**.  
When the user is **normal or happy**, the video will play at **normal speed**.

---

## Features

- Detects emotions like Happy, Confused, Angry, Neutral, Surprise
- Slows down video if confusion is detected
- Works with webcam and a sample video file
- Real-time emotion detection
- Easy to pause/resume video with space bar
- Exit anytime by pressing `q`

---

## Requirements

- Python 3.10.0
- Install required libraries:

```
pip install -r requirements.txt
```

The `requirements.txt` contains:

```
opencv-python==4.8.0.76
numpy==1.23.5
keras==2.10.0
tensorflow==2.10.0
```

---

## How to Run

1. Make sure you have Python 3.10.0 installed.

2. Clone the project or download the files.

3. Place your video file (example: `sample.mp4`) in the project folder.

4. Place your trained emotion detection model (`Emotion_Detection.h5`) in the project folder.

5. Open the terminal in the project folder.

6. Install the requirements:

```
pip install -r requirements.txt
```

7. Run the script:

```
python your_script_name.py
```

(Replace `your_script_name.py` with the name of your Python file.)

---

## Controls

- `Space bar` → Pause or Resume video
- `q` → Quit the program

---

## Notes

- The video will restart automatically when it finishes.
- The system checks your emotion every 2 seconds.
- If no face is detected, video will continue at normal speed.

---

## Folder Structure

```
/project-folder
│
├── Emotion_Detection.h5
├── sample.mp4
├── your_script_name.py
├── requirements.txt
└── README.md
```

---

## Credits

- OpenCV for face detection
- Keras and TensorFlow for emotion detection
- You, for creating this awesome project! 🚀
