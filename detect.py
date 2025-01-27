import cv2
from ultralytics import YOLO
import os

# Wczytaj wytrenowany model
model = YOLO('runs/detect/train25/weights/best.pt')  # Ścieżka do wytrenowanego modelu (pliku .pt)

# Funkcja do przetwarzania obrazu
def detect_objects(image):
    results = model(image)               # Wykrywanie obiektów
    annotated_frame = results[0].plot()  # Rysowanie wykrytych obiektów
    return annotated_frame

# Wykrywanie na zdjęciu
def detect_on_image(image_path):
    image = cv2.imread(image_path)                  # Wczytaj zdjęcie
    annotated_image = detect_objects(image)         # Wykryj obiekty
    cv2.imshow("Detected Objects", annotated_image) # Wyświetl obraz z wykrytymi obiektami
    cv2.waitKey(0)                                  # Naciśnij dowolny klawisz, aby zamknąć
    cv2.destroyAllWindows()                         # Zamknij okna

# Wykrywanie z kamery
def detect_from_camera():
    cap = cv2.VideoCapture(0)                           # Kamera (0 dla domyślnej)
    while cap.isOpened():                               # Dopóki kamera jest otwarta
        ret, frame = cap.read()                         # Wczytaj klatkę
        if not ret:                                     # Jeśli nie udało się wczytać klatki, przerwij
            break
        annotated_frame = detect_objects(frame)         # Wykryj obiekty
        cv2.imshow("Detected Objects", annotated_frame) # Wyświetl obraz z wykrytymi obiektami

        if cv2.waitKey(1) & 0xFF == ord('q'):           # Naciśnij 'q', aby zakończyć
            break
    cap.release()                                       # Zwolnij kamerę
    cv2.destroyAllWindows()                             # Zamknij okna



# Wykrywanie na wszystkich zdjęciach w folderze
def detect_on_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)                          # Utwórz folder wyjściowy, jeśli nie istnieje

    for filename in os.listdir(input_folder):               # Dla każdego pliku w folderze wejściowym
        input_path = os.path.join(input_folder, filename)   #Pełna ścieżka do pliku wejściowego
        if os.path.isfile(input_path):                      # Jeśli to plik
            image = cv2.imread(input_path)                  # Wczytaj zdjęcie
            if image is None:                               # Jeśli nie udało się wczytać zdjęcia
                print(f"Nie udało się wczytać pliku: {filename}")
                continue

            annotated_image = detect_objects(image)         # Wykryj obiekty

            output_path = os.path.join(output_folder, filename) # Pełna ścieżka do pliku wyjściowego
            cv2.imwrite(output_path, annotated_image)       # Zapisz przetworzone zdjęcie
            print(f"Przetworzono: {filename} -> {output_path}")


# Wybierz metodę wykrywania
detect_on_image("test/images/o9.jpg")       # Wykrywanie na zdjęciu
# detect_from_camera()            # Wykrywanie z kamery
# detect_on_folder("test/images", "test/result25")  # Wykrywanie na wszystkich zdjęciach w folderze
