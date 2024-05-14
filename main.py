import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    
    # Wykrycie krawędzi na masce
    edges = cv2.Canny(fgmask, 30, 150)

    # Nałożenie krawędzi na oryginalną klatkę
    frame_with_edges = cv2.bitwise_and(frame, frame, mask=edges)

    # Wyświetlenie oryginalnej klatki z nałożonymi krawędziami
    cv2.imshow('frame with edges', frame_with_edges)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
