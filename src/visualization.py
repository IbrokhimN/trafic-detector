import cv2
import numpy as np

C = {
    "green": (0, 220, 100), "yellow": (0, 220, 255), "red": (0, 60, 255),
    "cyan": (255, 220, 0), "white": (255, 255, 255), "black": (0, 0, 0),
    "dark": (30, 30, 30), "orange": (0, 140, 255), "magenta": (200, 0, 200),
    "blue": (255, 120, 0), "gray": (140, 140, 140),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

def speed_color(speed):
    if speed < 15:
        return C["green"]
    elif speed < 40:
        r = (speed - 15) / 25
        return tuple(int(C["green"][i] * (1 - r) + C["yellow"][i] * r) for i in range(3))
    return C["red"]

def draw_rounded_rect(img, pt1, pt2, color, alpha=0.82, radius=10):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy, sa in [(x1+radius,y1+radius,180),(x2-radius,y1+radius,270),(x1+radius,y2-radius,90),(x2-radius,y2-radius,0)]:
        cv2.ellipse(overlay, (cx, cy), (radius, radius), sa, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_traffic_light_widget(img, phase, timer, x, y):
    w_b, h_b = 50, 140
    draw_rounded_rect(img, (x, y), (x + w_b, y + h_b), (20, 20, 20), alpha=0.9, radius=8)
    positions = [(x + 25, y + 25), (x + 25, y + 65), (x + 25, y + 105)]
    cm = {"red": (0, 0, 200), "yellow": (0, 200, 200), "green": (0, 200, 0)}
    for i, s in enumerate(["red", "yellow", "green"]):
        c = cm[s] if s == phase["ns"] else (30, 30, 30)
        cv2.circle(img, positions[i], 14, c, -1)
        if s == phase["ns"]:
            cv2.circle(img, positions[i], 17, c, 2)
    cv2.putText(img, f"{timer}s", (x + 5, y + h_b + 22), FONT, 0.6, C["white"], 1, cv2.LINE_AA)
    cv2.putText(img, phase["name"][:8], (x - 5, y - 8), FONT, 0.4, C["cyan"], 1, cv2.LINE_AA)
    ped_x = x + w_b + 10
    ped_c = C["green"] if phase["ped"] == "green" else C["red"]
    cv2.rectangle(img, (ped_x, y + 40), (ped_x + 30, y + 100), (20, 20, 20), -1)
    cv2.circle(img, (ped_x + 15, y + 70), 10, ped_c, -1)

def draw_mini_graph(img, data, x, y, w, h, label="Vehicles"):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C["dark"], -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    if len(data) < 2:
        return
    mx = max(max(data), 1)
    pts = [(x + int(i / (len(data) - 1) * w), y + h - int(v / mx * (h - 24)) - 12) for i, v in enumerate(data)]
    cv2.fillPoly(img, [np.array(pts + [(x + w, y + h), (x, y + h)])], (40, 80, 40))
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], C["green"], 2, cv2.LINE_AA)
    cv2.putText(img, f"{label} | max:{mx} now:{data[-1]}", (x + 5, y + 14), FONT, 0.35, C["white"], 1, cv2.LINE_AA)

def draw_hud(img, metrics, x=10, y=10):
    lh, pad, w = 26, 10, 300
    h = pad * 2 + lh * len(metrics) + 4
    draw_rounded_rect(img, (x, y), (x + w, y + h), C["dark"], alpha=0.85, radius=10)
    cv2.putText(img, "SMART TRAFFIC AI", (x + pad, y + pad + 14), FONT, 0.5, C["cyan"], 1, cv2.LINE_AA)
    cv2.line(img, (x + pad, y + pad + 20), (x + w - pad, y + pad + 20), (60, 60, 60), 1)
    for i, (label, value, color) in enumerate(metrics):
        ty = y + pad + 40 + lh * i
        cv2.putText(img, label, (x + pad, ty), FONT, 0.42, C["gray"], 1, cv2.LINE_AA)
        cv2.putText(img, str(value), (x + 175, ty), FONT, 0.48, color, 1, cv2.LINE_AA)

def draw_box_corners(img, x1, y1, x2, y2, color, thick=2):
    corner = max(12, max(x2 - x1, y2 - y1) // 6)
    cv2.line(img, (x1, y1), (x1 + corner, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - corner, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + corner, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - corner, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner), color, thick, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

def draw_label(img, text, x1, y1, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
    cv2.putText(img, text, (x1 + 4, y1 - 5), FONT, scale, C["black"], 1, cv2.LINE_AA)
