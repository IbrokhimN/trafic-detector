import sys
import csv
import json
import math
import time
import os
from datetime import datetime
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from roi import select_roi, point_in_roi, draw_roi_overlay
from detection_utils import IncidentDetector, EcoEstimator, VehicleDB
from traffic_light import TrafficLight, calc_green_time
from visualization import (speed_color, draw_traffic_light_widget, draw_mini_graph,
                           draw_hud, draw_box_corners, draw_label, C, FONT)
from console_dashboard import build_console_dashboard, print_session_summary

# confs
MODEL_PATH = "yolov8l.pt"
CONFIDENCE = 0.35
IOU_THRESH = 0.45

CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
CLASS_BUS = 5
CLASS_TRUCK = 7
VEHICLE_CLASSES = [CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK]
CLASS_PERSON = 0
ALL_CLASSES = VEHICLE_CLASSES + [CLASS_PERSON]
CLASS_NAMES = {0: "person", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
EMERGENCY_CLASSES = [CLASS_BUS, CLASS_TRUCK]

SPEED_SCALE = 3.6
HEATMAP_DECAY = 0.993
GRAPH_HISTORY = 250
ELDERLY_HEIGHT_RATIO = 0.18
DASHBOARD_UPDATE_SEC = 2.0

CSV_LOG = "traffic_log.csv"
JSON_DASHBOARD = "dashboard_data.json"

def run_video_mode(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame")
        sys.exit(1)

    roi_polygon = select_roi(first_frame.copy())
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n[START] {frame_w}x{frame_h}, FPS: {fps_video:.0f}")
    print(f"[MODEL] Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    traffic_light = TrafficLight()
    incident_detector = IncidentDetector()
    eco = EcoEstimator()
    vehicle_db = VehicleDB()
    track_history = defaultdict(lambda: deque(maxlen=50))

    count_line_y = int(frame_h * 0.60)
    crossed_ids = set()
    count_up = 0
    count_down = 0
    NUM_LANES = 4
    lane_width = frame_w // NUM_LANES
    heatmap_accum = np.zeros((frame_h, frame_w), dtype=np.float32)
    graph_vehicles = deque(maxlen=GRAPH_HISTORY)
    graph_pedestrians = deque(maxlen=GRAPH_HISTORY)

    session_start = time.time()
    frame_num = 0
    total_pedestrians_seen = set()
    emergency_events = 0
    incident_events = 0
    max_density = 0
    skipped_outside = 0

    csv_file = open(CSV_LOG, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame", "vehicles", "pedestrians", "skipped",
                         "count_up", "count_down", "avg_speed", "traffic_phase", "incidents"])

    show_heatmap = False
    show_graph = True
    show_info = True
    show_debug = False
    paused = False
    prev_time = time.time()
    fps_display = 0.0
    last_dashboard_update = 0.0
    dashboard_history = []

    print("[RUN] Processing started. 'q' = quit, 'r' = new ROI\n")

    MAIN_WINDOW = "Smart Traffic AI"
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Stream ended.")
                break
            frame_num += 1

            results = model.track(source=frame, classes=ALL_CLASSES, conf=CONFIDENCE,
                                  iou=IOU_THRESH, persist=True, verbose=False, tracker="bytetrack.yaml")
            detections = results[0].boxes
            heatmap_accum *= HEATMAP_DECAY

            vehicles = []
            pedestrians = []
            emergency_detected = False
            has_elderly = False
            speeds_list = []
            frame_skipped = 0

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bw, bh = x2 - x1, y2 - y1

                if not point_in_roi(cx, cy, roi_polygon):
                    frame_skipped += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 1)
                    continue

                if cls_id in VEHICLE_CLASSES:
                    track_history[track_id].append((cx, cy))
                    hist = track_history[track_id]
                    speed = 0.0
                    if len(hist) >= 2:
                        speed = math.hypot(hist[-1][0] - hist[-2][0], hist[-1][1] - hist[-2][1]) * SPEED_SCALE
                    speeds_list.append(speed)

                    if track_id >= 0 and track_id not in crossed_ids and len(hist) >= 2:
                        if hist[-2][1] < count_line_y <= hist[-1][1]:
                            crossed_ids.add(track_id); count_down += 1
                        elif hist[-2][1] > count_line_y >= hist[-1][1]:
                            crossed_ids.add(track_id); count_up += 1

                    cv2.circle(heatmap_accum, (cx, cy), 35, 1.0, -1)
                    is_incident = incident_detector.update(track_id, speed, cx, cy)
                    if cls_id in EMERGENCY_CLASSES and speed > 30:
                        emergency_detected = True
                    vehicle_db.update(track_id, cls_id, speed, (cx, cy))
                    eco.update(track_id, speed)

                    vehicles.append({"box": (x1,y1,x2,y2), "cls": cls_id, "conf": conf,
                                     "tid": track_id, "speed": speed, "cx": cx, "cy": cy,
                                     "incident": is_incident, "hist": hist})

                elif cls_id == CLASS_PERSON:
                    total_pedestrians_seen.add(track_id)
                    is_elderly = bh / frame_h > ELDERLY_HEIGHT_RATIO
                    if is_elderly:
                        has_elderly = True
                    pedestrians.append({"box": (x1,y1,x2,y2), "conf": conf, "tid": track_id, "elderly": is_elderly})

            skipped_outside += frame_skipped
            vehicle_density = len(vehicles)
            max_density = max(max_density, vehicle_density)
            ped_count = len(pedestrians)
            if emergency_detected:
                emergency_events += 1
            traffic_light.tick(vehicle_density, ped_count, has_elderly, emergency_detected)

            frame = draw_roi_overlay(frame, roi_polygon)

            for v in vehicles:
                x1, y1, x2, y2 = v["box"]
                color = C["red"] if v["incident"] else speed_color(v["speed"])
                thick = 3 if max(x2-x1, y2-y1) > 150 else 2
                draw_box_corners(frame, x1, y1, x2, y2, color, thick)
                cv2.circle(frame, (v["cx"], v["cy"]), 4, color, -1, cv2.LINE_AA)

                cls_name = CLASS_NAMES.get(v["cls"], "?")
                lbl = f"#{v['tid']} {cls_name}"
                if v["speed"] > 0:
                    lbl += f" {v['speed']:.0f}km/h"
                draw_label(frame, lbl, x1, y1, color, 0.45)

                if len(v["hist"]) > 1:
                    cv2.polylines(frame, [np.array(v["hist"], dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
                if v["incident"]:
                    cv2.putText(frame, "!! INCIDENT !!", (v["cx"] - 50, v["cy"] - 20), FONT, 0.7, C["red"], 2, cv2.LINE_AA)

            for p in pedestrians:
                x1, y1, x2, y2 = p["box"]
                color = C["orange"] if p["elderly"] else C["cyan"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, "ELDERLY" if p["elderly"] else "person", (x1, y1-5), FONT, 0.4, color, 1, cv2.LINE_AA)

            cv2.line(frame, (0, count_line_y), (frame_w, count_line_y), C["orange"], 2)
            cv2.putText(frame, f"COUNT LINE | UP:{count_up} DOWN:{count_down}", (10, count_line_y - 8), FONT, 0.45, C["orange"], 1, cv2.LINE_AA)

            if show_heatmap:
                hm = np.clip(heatmap_accum / (heatmap_accum.max() + 1e-5), 0, 1)
                frame = cv2.addWeighted(frame, 0.65, cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.35, 0)

            draw_traffic_light_widget(frame, traffic_light.phase, traffic_light.timer, frame_w - 110, 10)

            graph_vehicles.append(vehicle_density)
            graph_pedestrians.append(ped_count)
            if show_graph:
                draw_mini_graph(frame, list(graph_vehicles), frame_w - 230, frame_h - 130, 210, 55, "Vehicles")
                draw_mini_graph(frame, list(graph_pedestrians), frame_w - 230, frame_h - 65, 210, 55, "Pedestrians")

            now = time.time()
            fps_display = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            avg_speed = sum(speeds_list) / len(speeds_list) if speeds_list else 0

            if show_info:
                draw_hud(frame, [
                    ("In Zone",        str(vehicle_density),               C["green"]),
                    ("Outside (skip)", str(frame_skipped),                 C["gray"]),
                    ("Pedestrians",    str(ped_count),                     C["cyan"]),
                    ("Crossed up",     str(count_up),                      C["green"]),
                    ("Crossed down",   str(count_down),                    C["red"]),
                    ("Avg Speed",      f"{avg_speed:.1f} km/h",            C["yellow"]),
                    ("Incidents",      str(len(incident_detector.active)), C["red"]),
                    ("Emergency",      "YES!" if emergency_detected else "—", C["red"] if emergency_detected else C["gray"]),
                    ("CO2 Saved",      f"{eco.co2_saved:.0f} g",          C["green"]),
                    ("Phase",          traffic_light.phase["name"][:10],   C["cyan"]),
                    ("FPS",            f"{fps_display:.0f}",               C["white"]),
                ])

            if show_debug:
                for i in range(1, NUM_LANES):
                    lx = i * lane_width
                    cv2.line(frame, (lx, 0), (lx, frame_h), (50, 50, 50), 1)

            csv_writer.writerow([datetime.now().isoformat(), frame_num, vehicle_density,
                                 ped_count, frame_skipped, count_up, count_down,
                                 f"{avg_speed:.1f}", traffic_light.phase["name"],
                                 len(incident_detector.active)])

            if time.time() - last_dashboard_update >= DASHBOARD_UPDATE_SEC:
                last_dashboard_update = time.time()
                snap = {"timestamp": datetime.now().isoformat(), "frame": frame_num,
                        "vehicles_now": vehicle_density, "pedestrians_now": ped_count,
                        "skipped_outside": frame_skipped, "total_up": count_up, "total_down": count_down,
                        "avg_speed": round(avg_speed, 1), "fps": round(fps_display, 1),
                        "roi_active": roi_polygon is not None}
                dashboard_history.append(snap)
                if len(dashboard_history) > 500:
                    dashboard_history = dashboard_history[-500:]
                with open(JSON_DASHBOARD, "w") as jf:
                    json.dump({"history": dashboard_history, "latest": snap}, jf, indent=2)

                elapsed_now = time.time() - session_start
                os.system('cls' if os.name == 'nt' else 'clear')
                dashboard = build_console_dashboard(
                    vehicle_count=vehicle_density,
                    ped_count=ped_count,
                    has_elderly=has_elderly,
                    emergency=emergency_detected,
                    avg_speed=avg_speed,
                    count_up=count_up,
                    count_down=count_down,
                    frame_skipped=frame_skipped,
                    incidents=len(incident_detector.active),
                    phase_name=traffic_light.phase["name"],
                    fps=fps_display,
                    frame_num=frame_num,
                    elapsed=elapsed_now,
                    eco_co2=eco.co2_saved,
                    eco_fuel=eco.fuel_saved * 1000,
                    roi_active=roi_polygon is not None,
                )
                console.print(dashboard)

        hints = "q:Quit p:Pause h:Heat g:Graph i:Info d:Debug r:ROI f:Fullscreen +/-:Line"
        cv2.putText(frame, hints, (10, frame_h - 8), FONT, 0.35, C["gray"], 1, cv2.LINE_AA)
        cv2.imshow(MAIN_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("h"):
            show_heatmap = not show_heatmap
        elif key == ord("g"):
            show_graph = not show_graph
        elif key == ord("i"):
            show_info = not show_info
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("t"):
            traffic_light.force_next()
        elif key == ord("f"):
            prop = cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord("r"):
            roi_polygon = select_roi(frame.copy())
            cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("[ROI] Zone updated")
        elif key in (ord("+"), ord("=")):
            count_line_y = max(50, count_line_y - 10)
        elif key == ord("-"):
            count_line_y = min(frame_h - 50, count_line_y + 10)

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    elapsed = time.time() - session_start

    os.system('cls' if os.name == 'nt' else 'clear')
    print_session_summary(elapsed, frame_num, vehicle_db, total_pedestrians_seen,
                          count_up, count_down, skipped_outside, eco, max_density)
