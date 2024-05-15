import numpy as np
import cv2
import mediapipe as mp
from cvzone.PoseModule import PoseDetector

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Inicjalizacja detektora ruchu
fgbg = cv2.createBackgroundSubtractorMOG2()

# Inicjalizacja detektora postaw ciała za pomocą Mediapipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Funkcja do rysowania krawędzi postaci na obrazie
def draw_body_edges(image, body_keypoints):
    # Zdefiniuj pary punktów kluczowych, które łączysz, aby utworzyć krawędzie postaci
    edges = [(11, 12), (11, 23), (23, 24), (24, 12), (11, 13), (12, 14), (13, 15), (14, 16),
             (13, 17), (14, 18), (17, 19), (18, 20), (19, 21), (20, 22)]

    # Przekształć obraz na obiekt NumPy
    image_np = np.array(image)

    # Iteruj po punktach kluczowych
    for landmark in body_keypoints.landmark:
        # Konwertuj współrzędne punktów kluczowych z wartości względnych na wartości absolutne
        height, width, _ = image_np.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)

        # Narysuj punkt kluczowy
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # Narysuj krawędzie postaci
    for edge in edges:
        start_point = body_keypoints.landmark[edge[0]]
        end_point = body_keypoints.landmark[edge[1]]
        if start_point.visibility > 0 and end_point.visibility > 0:
            start_x, start_y = int(start_point.x * width), int(start_point.y * height)
            end_x, end_y = int(end_point.x * width), int(end_point.y * height)
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Zastosowanie detektora ruchu do klatki
    fgmask = fgbg.apply(frame)

    # Wykrycie krawędzi na obrazie przetworzonym
    edges = cv2.Canny(frame, 30, 150)

    # Stworzenie obrazu zawierającego tylko zarysy krawędzi
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[np.where((edges_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

    # Wykrywanie postaci na obrazie za pomocą Mediapipe
    results = pose.process(frame)

    if results.pose_landmarks:
        # Narysuj krawędzie postaci na oryginalnym obrazie
        draw_body_edges(frame, results.pose_landmarks)

    # Połączenie oryginalnego obrazu z kamerki z zarysami krawędzi i prostokątami detekcji ludzi
    result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)

    cv2.imshow('Original Frame with Edges and Humans', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Zwolnienie kamery i zamknięcie okna
cap.release()
cv2.destroyAllWindows()
