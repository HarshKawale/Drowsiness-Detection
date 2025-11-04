import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from threading import Thread, Lock

# =========================
# Config
# =========================
IMG_SIZE = 64
MOUTH_YAWN_AREA_THRESHOLD = 1000
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
MOUTH = [78, 308, 13, 14]

# Detection at ~2–3 FPS for responsiveness, while camera streams full-speed
DETECTION_FPS = 3.0  # set to 2.0 or 3.0 as desired
DETECTION_INTERVAL = 1.0 / DETECTION_FPS

# Camera capture tuning (lower resolution reduces CPU; keep 30fps if supported)
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Option: downscale frame for detection only, to reduce Mediapipe/TF cost
# Drawing is still on the original frame using scaled boxes.
DETECT_DOWNSCALE = 0.5  # 0.5–0.8 recommended; 1.0 disables downscale

# JPEG encoding quality
JPEG_QUALITY = 75

# =========================
# Shared state
# =========================
detection_output = {
    "status": "Waiting...",
    "yawn_state": "No Yawn",
    "eye_state": "Unknown",
    "yawn_count": 0
}

last_yawn_active = False
latest_frame = None            # latest raw frame from camera
latest_annotated = None        # latest annotated frame (with boxes)
lock = Lock()

# Start/stop control
run_flag = False
grabber_thread = None
detector_thread = None

# =========================
# Utilities
# =========================
def beep():
    try:
        import winsound
        winsound.Beep(1500, 400)
        return
    except Exception:
        pass
    try:
        import os
        os.system('play -nq -t alsa synth 0.4 sine 1500')
        return
    except Exception:
        pass
    try:
        print('\a')
    except Exception:
        pass


class DrowsinessTimer:
    def __init__(self, delay=0.5):
        self.delay = delay
        self.start_time = None

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()

    def reset(self):
        self.start_time = None

    def alert(self):
        return self.start_time is not None and (time.time() - self.start_time) >= self.delay


# =========================
# Load models and Mediapipe
# =========================
eye_model = tf.keras.models.load_model("eye_model.keras")
yawn_model = tf.keras.models.load_model("yawn_model.h5")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Helpers
# =========================
def rect_scale(rect, scale_x, scale_y):
    x1, y1, x2, y2 = rect
    return int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

# =========================
# Threads
# =========================
def camera_grabber():
    """
    Grabs frames as fast as possible for smooth display.
    Applies capture hints to reduce latency and CPU usage.
    """
    global latest_frame, run_flag
    # CAP_DSHOW hint helps on Windows; harmless elsewhere
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Apply capture settings (only if supported by device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with lock:
                latest_frame = frame
            # Very small sleep to avoid pegging CPU
            time.sleep(0.002)
    finally:
        cap.release()

yawn_prediction_buffer = []
buffer_size = 5  # Number of recent frames to consider
smooth_threshold = 0.5  # Average threshold on smoothed prediction

def detection_worker():
    """
    Runs detection at a throttled FPS (2-3 FPS).
    Draws overlays, updates detection_output and latest_annotated.
    Uses optional downscale to reduce compute load while preserving stream quality.
    """
    global latest_frame, latest_annotated, detection_output, run_flag, last_yawn_active

    drowsy_timer = DrowsinessTimer(delay=0.5)
    beeped = False
    last_detect_t = 0.0

    while run_flag:
        # Snapshot latest frame
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        now = time.time()
        status = "Waiting..."  # ensure defined each iteration

        if (now - last_detect_t) >= DETECTION_INTERVAL:
            last_detect_t = now

            # Optionally downscale frame for faster detection
            if 0 < DETECT_DOWNSCALE < 1.0:
                small = cv2.resize(frame, None, fx=DETECT_DOWNSCALE, fy=DETECT_DOWNSCALE, interpolation=cv2.INTER_LINEAR)
                scale_x = 1.0 / DETECT_DOWNSCALE
                scale_y = 1.0 / DETECT_DOWNSCALE
                proc_frame = small
            else:
                proc_frame = frame
                scale_x = scale_y = 1.0

            rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = proc_frame.shape

            results = face_mesh.process(rgb_frame)
            left_eye_state = right_eye_state = "Open"
            yawn_state = "No Yawn"
            both_eyes_closed = False

            # Boxes to draw on original frame coordinates
            eye_boxes = []
            mouth_box = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_closed = right_closed = False

                # Eyes
                for eye_indices, eye_name in zip([LEFT_EYE, RIGHT_EYE], ["Left", "Right"]):
                    pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices])
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    x1, y1 = max(x1 - 3, 0), max(y1 - 3, 0)
                    x2, y2 = min(x2 + 3, w), min(y2 + 3, h)
                    eye_crop = proc_frame[y1:y2, x1:x2]
                    state = "Open"
                    if eye_crop.size > 0:
                        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
                        eye_img = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
                        eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB) / 255.0
                        eye_rgb = np.expand_dims(eye_rgb, axis=0)
                        pred = eye_model.predict(eye_rgb, verbose=0)[0][0]
                        print(f"[DEBUG] {eye_name} Eye model prediction: {pred:.3f}")
                        state = "Closed" if pred < 0.5 else "Open"
                        # Store scaled box for drawing on original frame
                        eye_boxes.append(rect_scale((x1, y1, x2, y2), scale_x, scale_y))

                    if eye_name == "Left":
                        left_eye_state = state
                        left_closed = (state == "Closed")
                    else:
                        right_eye_state = state
                        right_closed = (state == "Closed")

                both_eyes_closed = left_closed and right_closed

                # Mouth/yawn
                pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in MOUTH])
                x1, y1 = pts[:, 0].min() - 10, pts[:, 1].min() - 10
                x2, y2 = pts[:, 0].max() + 10, pts[:, 1].max() + 10
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)
                mw, mh = x2 - x1, y2 - y1
                mouth_crop = proc_frame[y1:y2, x1:x2]

                if mouth_crop.size > 0:
                    mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
                    mouth_img = cv2.resize(mouth_gray, (IMG_SIZE, IMG_SIZE))
                    mouth_rgb = cv2.cvtColor(mouth_img, cv2.COLOR_GRAY2RGB) / 255.0
                    mouth_rgb = np.expand_dims(mouth_rgb, axis=0)
                    pred = yawn_model.predict(mouth_rgb, verbose=0)[0][0]
                    # Update prediction buffer
                    yawn_prediction_buffer.append(pred)
                    if len(yawn_prediction_buffer) > buffer_size:
                        yawn_prediction_buffer.pop(0)

                    # Compute smoothed prediction
                    smoothed_pred = sum(yawn_prediction_buffer) / len(yawn_prediction_buffer)

                    # Use smoothed prediction for yawn state
                    yawn_state = "Yawn" if smoothed_pred > 0.009 else "No Yawn"

                    print(f"Yawn model raw: {pred:.3f}, smoothed: {smoothed_pred:.3f} -> {yawn_state}")

                elif mw > 50 and mh > 50 and (mw * mh) > MOUTH_YAWN_AREA_THRESHOLD:
                    yawn_state = "Yawn"

                # Save scaled mouth box and grid
                if mw > 0 and mh > 0:
                    mouth_box = rect_scale((x1, y1, x2, y2), scale_x, scale_y)

                # Yawn rising-edge counting
                with lock:
                    if yawn_state == "Yawn" and not last_yawn_active:
                        detection_output["yawn_count"] = detection_output.get("yawn_count", 0) + 1
                        last_yawn_active = True
                    elif yawn_state != "Yawn" and last_yawn_active:
                        last_yawn_active = False


                # Status (no text overlay; only update detection_output)
                if both_eyes_closed:
                    drowsy_timer.start()
                else:
                    drowsy_timer.reset()
                    beeped = False

                if drowsy_timer.alert() or yawn_state == "Yawn":
                    status = "DROWSY"
                    if not beeped:
                        try:
                            beep()
                        except Exception:
                            pass
                        beeped = True
                else:
                    status = "ALERT"
                    beeped = False

                # Draw overlays on original full-size frame
                draw = frame.copy()
                # Eyes
                for (ex1, ey1, ex2, ey2) in eye_boxes:
                    cv2.rectangle(draw, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
                # Mouth with grid
                if mouth_box:
                    mx1, my1, mx2, my2 = mouth_box
                    cv2.rectangle(draw, (mx1, my1), (mx2, my2), (255, 0, 0), 2)
                    mw2, mh2 = mx2 - mx1, my2 - my1
                    step_x, step_y = max(mw2 // 6, 1), max(mh2 // 4, 1)
                    for i in range(1, 6):
                        xline = mx1 + i * step_x
                        cv2.line(draw, (xline, my1), (xline, my2), (255, 0, 0), 1)
                    for j in range(1, 4):
                        yline = my1 + j * step_y
                        cv2.line(draw, (mx1, yline), (mx2, yline), (255, 0, 0), 1)

                with lock:
                    latest_annotated = draw
                    detection_output["status"] = status
                    detection_output["yawn_state"] = yawn_state
                    detection_output["eye_state"] = f"Left: {left_eye_state}, Right: {right_eye_state}"
            else:
                # No face detected; pass-through frame and reset states
                with lock:
                    latest_annotated = frame
                    detection_output["status"] = "Waiting..."
                    detection_output["yawn_state"] = "No Yawn"
                    detection_output["eye_state"] = "Unknown"
        else:
            # Between detections, keep stream smooth with latest raw frame if annotated missing
            with lock:
                if latest_annotated is None and latest_frame is not None:
                    latest_annotated = latest_frame.copy()

        time.sleep(0.002)


# =========================
# Public API for Flask app
# =========================
def start_detection():
    """
    Starts camera and detection threads on first call.
    Called implicitly by generate_frames() when /start is opened.
    """
    global run_flag, grabber_thread, detector_thread
    if run_flag:
        return
    run_flag = True
    grabber_thread = Thread(target=camera_grabber, daemon=True)
    detector_thread = Thread(target=detection_worker, daemon=True)
    grabber_thread.start()
    detector_thread.start()


def stop_detection():
    """
    Signals threads to stop; camera will be released by the grabber thread.
    """
    global run_flag
    run_flag = False


def generate_frames():
    """
    MJPEG generator:
    - Starts detection if not running
    - Streams latest annotated frames
    """
    start_detection()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while run_flag:
        with lock:
            frame = latest_annotated.copy() if latest_annotated is not None else None
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            time.sleep(0.005)
