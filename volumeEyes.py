import cv2
import numpy as np
import mediapipe as mp
import subprocess
import time

# Inisialisasi Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mengatur volume sistem menggunakan pactl
def set_volume(volume):
    # volume harus dalam rentang 0 hingga 100
    volume = max(0, min(volume, 100))
    # Setel volume dengan pactl
    subprocess.call(f"pactl set-sink-volume @DEFAULT_SINK@ {volume}%", shell=True)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Variabel untuk melacak waktu kondisi mata
start_time_open = None
start_time_closed = None
is_open = False
is_closed = False
volume = 50  # Volume awal diatur ke 50%

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    results_face = face_mesh.process(frame_rgb)

    # Inisialisasi status mata
    left_eye_open = 0
    right_eye_open = 0

    # Jika wajah terdeteksi
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Gambar landmark wajah
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Ambil koordinat landmark mata kiri dan mata kanan
            left_eye_upper = face_landmarks.landmark[386]
            left_eye_lower = face_landmarks.landmark[374]
            right_eye_upper = face_landmarks.landmark[159]
            right_eye_lower = face_landmarks.landmark[145]

            # Hitung jarak vertikal antara kelopak mata atas dan bawah
            left_eye_open = np.sqrt((left_eye_upper.x - left_eye_lower.x) ** 2 + (left_eye_upper.y - left_eye_lower.y) ** 2)
            right_eye_open = np.sqrt((right_eye_upper.x - right_eye_lower.x) ** 2 + (right_eye_upper.y - right_eye_lower.y) ** 2)

    # Ambang batas untuk mendeteksi mata melotot atau terpejam
    eye_open_threshold = 0.02
    eye_closed_threshold = 0.02

    current_time = time.time()

    # Cek kondisi mata melotot
    if left_eye_open > eye_open_threshold and right_eye_open > eye_open_threshold:
        if not is_open:
            start_time_open = current_time
            is_open = True
        elif current_time - start_time_open > 3:
            volume = 100
            set_volume(volume)
            start_time_closed = None  # Reset start_time_closed jika mata melotot
            is_closed = False
    else:
        is_open = False

    # Cek kondisi mata terpejam
    if left_eye_open < eye_closed_threshold and right_eye_open < eye_closed_threshold:
        if not is_closed:
            start_time_closed = current_time
            is_closed = True
        elif current_time - start_time_closed > 3:
            volume = 0
            set_volume(volume)
            start_time_open = None  # Reset start_time_open jika mata terpejam
            is_open = False
    else:
        is_closed = False

    # Tampilkan volume pada frame
    cv2.putText(frame, f'Volume: {volume}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Volume Control', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilis dan tutup jendela
cap.release()
cv2.destroyAllWindows()
