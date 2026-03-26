An intelligent traffic light control system based on computer vision.  
Analyzes the road situation in real time, detects vehicles and pedestrians,  
calculates optimal traffic light phase durations, and saves statistics.

## Demo

![Smart Traffic AI Demo](https://raw.githubusercontent.com/IbrokhimN/trafic-detector/main/docs/vids/video.gif)

## Features

- region of Interest selection to count objects only within the road area
- object detection: cars, motorcycles, buses, trucks, pedestrians (YOLOv8)
- object tracking (ByteTrack)
- speed estimation
- elderly pedestrian detection (to increase crossing time)
- emergency vehicle recognition (priority switching)
- dynamic green time calculation for vehicles and pedestrians
- incident detection (sudden stop)
- traffic density heatmap
- real‑time density mini‑graphs
- environmental analytics (estimated CO₂ and fuel savings)
- CSV and JSON logging for dashboard
- photo analysis mode with annotated image and JSON report

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IbrokhimN/trafic-detector.git
   cd trafic-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

   *Note:* The YOLOv8l model will be downloaded automatically on first run.

## Usage

### Video / Webcam / RTSP

```bash
python src/main.py                # webcam
python src/main.py video.mp4      # video file
python src/main.py rtsp://user:pass@ip/stream   # RTSP stream
```

On startup, a window will appear to select the road zone (ROI).  
Click on the corners of the road (minimum 3 points), then press `ENTER`.  
Press `ESC` to skip ROI (the whole frame will be used).

#### Controls during video mode

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / resume |
| `h` | Toggle heatmap |
| `g` | Toggle density graphs |
| `i` | Toggle HUD panel |
| `d` | Toggle debug lane lines |
| `r` | Redefine ROI |
| `f` | Toggle fullscreen |
| `+` / `-` | Move count line up/down |

### Photo mode

```bash
python src/main.py photo.jpg
```

The image will be analyzed, results printed to the console, and the following files will be created:
- `<image_name>_annotated.<ext>` — image with annotated objects
- `<image_name>_stats.json` — detailed statistics

## Output Files

- `traffic_log.csv` — step‑by‑step statistics (timestamp, frame, vehicle/pedestrian counts, skipped objects, crossings, average speed, phase, incidents)
- `dashboard_data.json` — data for a web dashboard (latest snapshot and history)
- `<image_name>_stats.json` — photo analysis statistics

## Project Structure

```
trafic-detector/
├── docs/                       # Documentation (optional)
├── src/
│   ├── dataset/                # Datasets folder (optional)
│   ├── console_dashboard.py    # Rich dashboard and session summary
│   ├── detection_utils.py      # Helper classes (incidents, eco, DB)
│   ├── main.py                 # Entry point
│   ├── photo_mode.py           # Photo processing
│   ├── requirements.txt        # Dependencies
│   ├── roi.py                  # Region of Interest selection
│   ├── traffic_light.py        # Traffic light class and timing calculation
│   ├── video_mode.py           # Main video/stream processing loop
│   └── visualization.py        # Drawing functions (HUD, boxes, graphs)
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
