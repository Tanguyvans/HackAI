from ultralytics import YOLO

# Initialize the YOLO model with the pre-trained weights
model = YOLO("best_70.pt")

# # Perform prediction on the image and store the results
# results = model.predict("0.jpg", imgsz=640, conf=0.1, save=True)

# # Plot the prediction results
# results.show()

results = model(['0.jpg', '1.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')  # save to disk
