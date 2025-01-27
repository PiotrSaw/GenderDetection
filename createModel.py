if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO()  # Stwórz model YOLO

    # Rozpocznij trening modelu
    model.train(
        data='D:/Users/Piotr Sawecki/Documents/objectDetection/program/ObjectDetection-8/data.yaml',      # Dane treningowe
        epochs=100,          # Liczba epok treningowych
        imgsz=640,          # Rozmiar obrazu
        batch=6,            # Rozmiar batcha - liczba obrazów przetwarzanych jednocześnie
        device='0',         # Numer karty graficznej
    )

    # Zapisz model po treningu do pliku .pt
    model.export(format="torchscript")
