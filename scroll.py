import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import signal
import sys

# Fungsi untuk menangani interupsi (misalnya Ctrl+C)
def signal_handler(sig, frame):
    print('Exiting gracefully')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Daftarkan signal handler untuk SIGINT
signal.signal(signal.SIGINT, signal_handler)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Fungsi untuk menggulir layar
def scroll_page(direction):
    if direction == 'down':
        pyautogui.scroll(-100)

# Variabel untuk melacak waktu tindakan terakhir
last_action_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    results = hands.process(frame_rgb)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil koordinat landmark ibu jari dan jari telunjuk
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Hitung jarak antara ibu jari dan jari telunjuk
            thumb_tip_y = thumb_tip.y * frame.shape[0]
            index_finger_tip_y = index_finger_tip.y * frame.shape[0]
            distance = thumb_tip_y - index_finger_tip_y

            # Tentukan arah gulir berdasarkan jarak antara jari
            current_time = time.time()
            if 40 < distance < 60 and (current_time - last_action_time) > 1:
                scroll_page('down')
                direction = 'down'
                last_action_time = current_time
            else:
                direction = 'none'

            # Tampilkan koordinat dan arah gulir pada frame
            cv2.putText(frame, f'Scroll: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Distance: {distance:.2f} pixels', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Scroll Control', frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilis dan tutup jendela
cap.release()
cv2.destroyAllWindows()
