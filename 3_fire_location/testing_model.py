from ultralytics import YOLO


model = YOLO("best.pt")


model.predict("0.jpg", imgsz=640, conf=0.1, save=True)
