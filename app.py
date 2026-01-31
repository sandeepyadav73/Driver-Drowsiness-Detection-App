from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import winsound
import os

# ===============================
# PATH CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
ALARM_PATH = os.path.join(BASE_DIR, "assets", "alarm.wav")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# ===============================
# MEDIAPIPE INIT
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# THRESHOLDS
# ===============================
EYE_AR_THRESH = 0.25
EYE_AR_FRAMES = 20
YAWN_THRESH = 0.6
YAWN_FRAMES = 15

COUNTER = 0
YAWN_COUNTER = 0
ALARM_ON = False

# ===============================
# HELPERS
# ===============================
def sound_alarm():
    winsound.PlaySound(
        ALARM_PATH,
        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
    )

def euclidean(p1, p2):
    return dist.euclidean(p1, p2)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_ratio(top, bottom, left, right):
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal

# ===============================
# VIDEO STREAM
# ===============================
def generate_frames():
    global COUNTER, YAWN_COUNTER, ALARM_ON

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if result.multi_face_landmarks:
            for face in result.multi_face_landmarks:

                def lm(i):
                    return (
                        int(face.landmark[i].x * w),
                        int(face.landmark[i].y * h)
                    )

                # Eye landmarks
                left_eye = [lm(i) for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [lm(i) for i in [362, 385, 387, 263, 373, 380]]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                # Mouth landmarks
                top = lm(13)
                bottom = lm(14)
                left = lm(78)
                right = lm(308)
                mar = mouth_ratio(top, bottom, left, right)

                # Draw landmarks
                for p in left_eye + right_eye:
                    cv2.circle(frame, p, 2, (0, 255, 0), -1)
                cv2.circle(frame, top, 2, (255, 0, 0), -1)
                cv2.circle(frame, bottom, 2, (255, 0, 0), -1)

                # ===============================
                # DROWSINESS LOGIC
                # ===============================
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            sound_alarm()
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    if ALARM_ON:
                        ALARM_ON = False
                        winsound.PlaySound(None, winsound.SND_PURGE)

                # ===============================
                # YAWN LOGIC
                # ===============================
                if mar > YAWN_THRESH:
                    YAWN_COUNTER += 1
                    if YAWN_COUNTER >= YAWN_FRAMES:
                        cv2.putText(frame, "YAWN DETECTED", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    YAWN_COUNTER = 0

                cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
