import cv2
import numpy as np
import mediapipe as mp
import subprocess

# Inisialisasi Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mengatur volume sistem menggunakan pactl
def set_volume(volume):
    # volume harus dalam rentang 0 hingga 100
    volume = max(0, min(volume, 100))
    # Setel volume dengan pactl
    subprocess.call(f"pactl set-sink-volume @DEFAULT_SINK@ {volume}%", shell=True)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    results = hands.process(frame_rgb)
    print(results)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil koordinat landmark ibu jari dan jari telunjuk
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Hitung jarak antara ibu jari dan jari telunjuk
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_finger_tip_x, index_finger_tip_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            distance = np.sqrt((thumb_tip_x - index_finger_tip_x) ** 2 + (thumb_tip_y - index_finger_tip_y) ** 2)

            # Skala jarak untuk mendapatkan nilai volume (0-100)
            volume = np.clip(int((distance / 200) * 100), 0, 100)
            set_volume(volume)
          


            # Tampilkan jarak dan volume pada frame
            cv2.putText(frame, f'Volume: {volume}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Volume Control', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilis dan tutup jendela
cap.release()
cv2.destroyAllWindows()
