**Car Detection and Counting System**
**Technical Documentation**

This document describes a Python-based computer vision system designed to detect and count cars in a video stream using a deep learning model. The system utilizes the **YOLOv8 (You Only Look Once)** object detection framework together with **OpenCV** for video processing and visualization.
The program processes frames from a video source (either a camera or a video file), detects cars using a pretrained neural network model, draws bounding boxes around detected vehicles, and displays the total number of cars detected in each frame in real time.
The script is executed from the command line with a specified video source.
Example execution command:

```bash
python3 detector.py ( video source )
```

If no source is provided, the system defaults to the primary webcam.

## Architecture

The system consists of the following main components:
1. **Video Input Module**
   Responsible for capturing frames from a video file or camera device.
2. **Object Detection Model**
   A pretrained YOLOv8 model used to identify objects within each frame.
3. **Filtering Module**
   Filters detections to include only vehicles of class "car".
4. **Visualization Module**
   Draws bounding boxes, confidence labels, and the car counter on the frame.
5. **Display and Control Module**
   Displays the processed frames in a window and handles user input to terminate the program.

## Dependencies

The system requires the following software libraries:

* Python 3.11+
* OpenCV (`cv2`)
* Ultralytics YOLO (`ultralytics`)

Installation of necessary libraries:
```bash
pip install -r requirements.txt
```

## Model Description
The program uses the pretrained model: ***yolov8l.pt***
This is the **YOLOv8 Large** model provided by the Ultralytics framework. The larger model offers improved detection accuracy compared to smaller versions (such as YOLOv8m), though it requires more computational resources.
Only objects classified as **cars** are processed in the detection pipeline.

```
CAR_CLASS_ID = 2
```

This corresponds to the class index used in the COCO dataset, which YOLO models are trained on.

## Configuration Parameters

Several parameters control the behavior of the detection system.

### Detection Parameters

**Confidence Threshold** is 0.35

This value determines the minimum confidence level required for a detection to be accepted.

## Input Sources
The program supports two types of input:

### Webcam Input

If no argument is provided, the system uses the default camera.

```bash
python3 detector.py
```

### Video File
A video file can be provided as an argument.

Example:
```
python3 detector.py video.mp4
```

## Performance Considerations

The YOLOv8 Large model provides high detection accuracy but requires significant computational resources.
Performance depends on:
* GPU availability
* CPU performance
* video resolution
* frame rate
For real-time applications, a GPU-enabled environment is recommended.
