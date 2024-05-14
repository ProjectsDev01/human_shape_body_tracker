import numpy as np
import cv2

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Inicjalizacja detektora ruchu
fgbg = cv2.createBackgroundSubtractorMOG2()

# Wczytanie klasyfikatora Haar Cascade do detekcji ludzi
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

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

    # Detekcja ludzi na oryginalnej klatce
    humans = human_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Narysowanie prostokątów wokół wykrytych ludzi
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Połączenie oryginalnego obrazu z kamerki z zarysami krawędzi i prostokątami detekcji ludzi
    result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)

    cv2.imshow('Original Frame with Edges and Humans', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Zwolnienie kamery i zamknięcie okna
cap.release()
cv2.destroyAllWindows()
