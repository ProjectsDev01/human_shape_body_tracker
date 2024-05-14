import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    # Wykrycie krawędzi na obrazie przetworzonym
    edges = cv2.Canny(frame, 30, 150)

    # Stworzenie obrazu zawierającego tylko zarysy krawędzi
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[np.where((edges_colored==[255,255,255]).all(axis=2))] = [0,0,255]

    # Połączenie oryginalnego obrazu z kamerki z zarysami krawędzi
    result = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)

    cv2.imshow('Original Frame with Edges', result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
