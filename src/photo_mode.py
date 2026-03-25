import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from roi import select_roi, point_in_roi, draw_roi_overlay
from visualization import draw_box_corners, draw_label, draw_hud, draw_rounded_rect, C, FONT
from traffic_light import calc_green_time

MODEL_PATH = "yolov8l.pt"
CONFIDENCE = 0.15
IOU_THRESH = 0.45
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_PERSON = 0
ELDERLY_HEIGHT_RATIO = 0.18

CLS_COLORS = {0: C["cyan"], 1: (180, 130, 0), 2: C["green"], 3: C["orange"], 5: C["magenta"], 7: C["blue"]}

def analyze_photo(image_path: str):
    print(f"[PHOTO] Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not open: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"[PHOTO] {w}x{h}")

    roi = select_roi(img.copy())

    print("[PHOTO] Detection...")
    results = model.predict(source=img, conf=CONFIDENCE, iou=IOU_THRESH, verbose=False)
    detections = results[0].boxes
    coco_names = model.names

    counts = defaultdict(int)
    confidences = defaultdict(list)
    objects_detail = []
    skipped = 0

    img = draw_roi_overlay(img, roi)

    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        name = coco_names.get(cls_id, f"class_{cls_id}")

        if not point_in_roi(cx, cy, roi):
            skipped += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)
            cv2.putText(img, "OUTSIDE", (x1, y1 - 5), FONT, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
            continue

        counts[name] += 1
        confidences[name].append(round(conf, 3))
        objects_detail.append({"class": name, "confidence": round(conf, 3), "bbox": [x1, y1, x2, y2]})

        color = CLS_COLORS.get(cls_id, C["yellow"])
        thick = 3 if max(x2 - x1, y2 - y1) > 150 else 2
        draw_box_corners(img, x1, y1, x2, y2, color, thick)
        cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)
        draw_label(img, f"{name} {conf:.0%}", x1, y1, color, 0.5 if max(x2-x1, y2-y1) > 100 else 0.4)

    vehicle_types = ["car", "motorcycle", "bus", "truck", "bicycle"]
    total_v = sum(counts.get(v, 0) for v in vehicle_types)
    total_p = counts.get("person", 0)
    total_o = sum(counts.values()) - total_v - total_p

    hud = [
        ("IN ROAD ZONE",   str(sum(counts.values())), C["white"]),
        ("Skipped (outside)", str(skipped),            C["gray"]),
        ("Vehicles",       str(total_v),               C["green"]),
        ("  Cars",         str(counts.get("car", 0)),  C["green"]),
        ("  Trucks",       str(counts.get("truck", 0)),C["blue"]),
        ("  Buses",        str(counts.get("bus", 0)),  C["magenta"]),
        ("  Motos",        str(counts.get("motorcycle", 0)), C["orange"]),
        ("People",         str(total_p),               C["cyan"]),
        ("Other",          str(total_o),               C["yellow"]),
    ]
    hud = [x for x in hud if x[1] != "0" or x[0] in ("IN ROAD ZONE", "Vehicles", "People")]
    draw_hud(img, hud, x=10, y=10)

    ov = img.copy()
    cv2.rectangle(ov, (0, h - 40), (w, h), C["dark"], -1)
    cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
    roi_text = f"ROI: {len(roi) if roi is not None else 0} pts" if roi is not None else "ROI: full frame"
    cv2.putText(img, f"SMART TRAFFIC AI  |  {roi_text}  |  In zone: {sum(counts.values())}  Skipped: {skipped}  |  {Path(image_path).name}",
                (10, h - 14), FONT, 0.42, C["white"], 1, cv2.LINE_AA)

    has_elderly = False
    for det in detections:
        cls_id = int(det.cls[0])
        if cls_id == CLASS_PERSON:
            _, by1, _, by2 = map(int, det.xyxy[0])
            if (by2 - by1) / h > ELDERLY_HEIGHT_RATIO:
                has_elderly = True
                break

    timing = calc_green_time(total_v, total_p, avg_speed=0, has_elderly=has_elderly)
    v_green = timing["v_green"]
    p_green = timing["p_green"]
    walk_spd = "0.8" if has_elderly else "1.3"

    # console output simplified
    print()
    print("=" * 55)
    print(f"SMART TRAFFIC AI — Photo Analysis")
    print("=" * 55)
    print(f"File:          {Path(image_path).name}")
    print(f"Resolution:    {w}x{h}")
    print(f"In zone:       {sum(counts.values())}")
    print(f"Outside zone:  {skipped}")
    print("-" * 55)
    print(f"Vehicles:      {total_v} (cars:{counts.get('car',0)} trucks:{counts.get('truck',0)} buses:{counts.get('bus',0)} motos:{counts.get('motorcycle',0)})")
    print(f"Pedestrians:   {total_p}" + (" (elderly detected)" if has_elderly else ""))
    print("-" * 55)
    print(f"Recommended green time:")
    print(f"  Vehicles:    {v_green} sec")
    print(f"  Pedestrians: {p_green} sec")
    print(f"  Full cycle:  {timing['cycle']} sec")
    print(f"  Reason:      {timing['reason']}")
    print("=" * 55)

    # json output
    simple = {"cars": counts.get("car", 0), "trucks": counts.get("truck", 0), "buses": counts.get("bus", 0),
              "motorcycles": counts.get("motorcycle", 0), "people": total_p, "total_vehicles": total_v,
              "total_in_zone": sum(counts.values()), "skipped_outside": skipped,
              "green_vehicles_sec": v_green, "green_pedestrians_sec": p_green,
              "full_cycle_sec": timing["cycle"], "has_elderly": has_elderly}
    simple = {k: v for k, v in simple.items() if v or k in ("cars", "people", "total_in_zone", "green_vehicles_sec", "green_pedestrians_sec")}

    full = {"source": image_path, "resolution": f"{w}x{h}", "roi_points": roi.tolist() if roi is not None else None,
            "total_in_zone": sum(counts.values()), "skipped_outside": skipped, "summary": simple,
            "timing": timing,
            "counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
            "avg_confidence": {n: round(sum(c)/len(c), 3) for n, c in confidences.items()},
            "objects": objects_detail}

    jp = Path(image_path).stem + "_stats.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)


    print()
    print(f"  JSON:  {jp}")
    print()

    try:
        wn = "Smart Traffic AI | Photo | any key to close"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(wn, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(wn, img)
        print("[INFO] Press any key.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass
    return full
