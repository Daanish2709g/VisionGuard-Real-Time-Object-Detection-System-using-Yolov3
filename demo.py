import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

def process_image(img):
    """Resize, normalize, and expand image dimensions for YOLO model input."""
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_classes(file):
    """Read class names from a file."""
    if not os.path.exists(file):
        raise FileNotFoundError(f"Class names file '{file}' not found.")
    
    with open(file) as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def draw(image, boxes, scores, classes, all_classes):
    """Draw bounding boxes and labels on the image."""
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # Draw rectangle and label
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        label = f'{all_classes[cl]} {score:.2f}'
        cv2.putText(image, label, (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        print(f'class: {all_classes[cl]}, score: {score:.2f}')
        print(f'box coordinates x,y,w,h: {box}')

def detect_image(image, yolo, all_classes):
    """Detect objects in a single image using YOLO model."""
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print(f'YOLO inference time: {end - start:.2f}s')

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

def detect_video(video, yolo, all_classes):
    """Detect objects in a video frame by frame."""
    video_path = os.path.join("videos", "test", video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")
    
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        raise RuntimeError(f"Failed to open video file '{video_path}'.")

    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare video writer for saving output
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(os.path.join("videos", "res", video),
                                   cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

    while True:
        ret, frame = camera.read()
        if not ret:
            print("End of video or failed to capture frame.")
            break

        processed_frame = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", processed_frame)
        video_writer.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to quit
            break

    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize YOLO model with threshold values
    yolo = YOLO(confidence_threshold=0.6, iou_threshold=0.5)

    # Load class names from file
    classes_file = 'data/coco_classes.txt'
    all_classes = get_classes(classes_file)

    # Process video
    video_file = 'library1.mp4'
    detect_video(video_file, yolo, all_classes)
