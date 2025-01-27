from roboflow import Roboflow


# Pobierz dane z Roboflow do zmiennej dataset
rf = Roboflow(api_key="GJEYNucPFKAw8Xebo8Aj")
project = rf.workspace("piotr-qzkyd").project("objectdetection-vdafc")
version = project.version(8)
dataset = version.download("yolov11")