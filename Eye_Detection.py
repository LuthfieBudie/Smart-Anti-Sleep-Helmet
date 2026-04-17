import cv2
import mediapipe as mp
import numpy as np
import time
import requests
  
esp32 = "192.168.253.188"
def send_command_to_esp32(command): 
    try:
        url = f"http://{esp32}/{command}"
        response = requests.get(url, timeout=1.5)
        print("Command sent:", command, "Status:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Failed to send command:", e)
        
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])  # vertikal 1
    B = dist(eye[2], eye[4])  # vertikal 2
    C = dist(eye[0], eye[3])  # horizontal
    return (A + B) / (2.0 * C)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

ear_low = 0.13
ear_high = 0.24
tidur = False
start_sleep_time = None
led_terakhir = 0
interval_led = 5
start_range_time = None
start_above_time = None
led_state = False


while True:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if not ret:
        continue

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        left_eye = [(int(face.landmark[i].x * w),
                     int(face.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face.landmark[i].x * w),
                      int(face.landmark[i].y * h)) for i in RIGHT_EYE]

        # =========================
        # Gambar titik landmark
        # =========================
        for p in left_eye + right_eye:
            cv2.circle(frame, p, 2, (0, 255, 0), -1)

        # =========================
        # Gambar garis EAR (horizontal & vertikal)
        # =========================
        # Left eye
        cv2.line(frame, left_eye[0], left_eye[3], (255, 0, 0), 1)
        cv2.line(frame, left_eye[1], left_eye[5], (0, 0, 255), 1)
        cv2.line(frame, left_eye[2], left_eye[4], (0, 0, 255), 1)

        # Right eye
        cv2.line(frame, right_eye[0], right_eye[3], (255, 0, 0), 1)
        cv2.line(frame, right_eye[1], right_eye[5], (0, 0, 255), 1)
        cv2.line(frame, right_eye[2], right_eye[4], (0, 0, 255), 1)

        # ========================= 
        # Hitung EAR
        # =========================
        ear = (eye_aspect_ratio(left_eye) +
               eye_aspect_ratio(right_eye)) / 2.0

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
        if ear_low <= ear <= ear_high:
            start_sleep_time = None
            start_above_time = None
            if start_range_time is None:
                start_range_time = time.time()
            durasi_range = time.time() - start_range_time
            if durasi_range >= 3 and not led_state:
                send_command_to_esp32("led")
                led_state = True
                start_range_time = None


        elif ear > ear_high:
            start_range_time = None
            start_sleep_time = None
            if tidur:
                send_command_to_esp32("off")
                tidur = False
            if led_state:
                if start_above_time is None:
                    start_above_time = time.time()
                durasi_above = time.time() - start_above_time
                if durasi_above >= 3:
                    send_command_to_esp32("off")
                    led_state = False
                    start_above_time = None

  
        elif ear < ear_low:
            start_above_time = None
            start_range_time = None
            if start_sleep_time is None:
                start_sleep_time = time.time()
            durasi = time.time() - start_sleep_time
            if durasi >= 3 and not tidur:
                print("tidur")
                send_command_to_esp32("on")
                send_command_to_esp32("led")
                tidur = True

                
        else:
            start_above_time = None
            start_sleep_time = None
            start_range_time = None
                



    cv2.imshow("EAR Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
    
