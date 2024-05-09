import cv2

# Utworzenie obiektu BackgroundSubtractorMOG2
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Uruchomienie kamery
cap = cv2.VideoCapture(0)

while True:
    # Odczytanie klatki z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Zastosowanie BackgroundSubtractorMOG2 do klatki
    fg_mask = background_subtractor.apply(frame)

    # Wyświetlenie klatki i maske na obrazie
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Przerwanie pętli po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknięcie kamery i okien
cap.release()
cv2.destroyAllWindows()
