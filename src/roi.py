import cv2
import numpy as np

C = {
    "cyan": (255, 220, 0),
    "white": (255, 255, 255),
    "yellow": (0, 220, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

roi_points = []

def _roi_mouse_callback(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

def select_roi(frame, window_name="Select Road Zone (ROI)"):
    global roi_points
    roi_points = []
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, _roi_mouse_callback)

    print("\n" + "=" * 55)
    print("  SELECT ROAD ZONE (ROI)")
    print("=" * 55)
    print("  Click corners of the road (minimum 3 points)")
    print("  ENTER — confirm zone")
    print("  R     — reset points")
    print("  ESC   — skip (use whole frame)")
    print("=" * 55 + "\n")

    while True:
        display = frame.copy()
        h, w = display.shape[:2]

        if len(roi_points) >= 3:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            dark = display.copy()
            dark = (dark * 0.3).astype(np.uint8)
            display = np.where(mask[:, :, None] == 255, display, dark)
            cv2.polylines(display, [pts], True, C["cyan"], 2, cv2.LINE_AA)
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (80, 60, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)

        for i, pt in enumerate(roi_points):
            cv2.circle(display, pt, 6, C["cyan"], -1, cv2.LINE_AA)
            cv2.circle(display, pt, 8, C["white"], 1, cv2.LINE_AA)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1] - 5),
                        FONT, 0.5, C["cyan"], 1, cv2.LINE_AA)
            if i > 0:
                cv2.line(display, roi_points[i - 1], pt, C["cyan"], 2, cv2.LINE_AA)

        cv2.putText(display, "Click corners of the road area", (10, 30),
                    FONT, 0.7, C["white"], 2, cv2.LINE_AA)
        cv2.putText(display, f"Points: {len(roi_points)}  |  ENTER=confirm  R=reset  ESC=skip",
                    (10, h - 15), FONT, 0.5, C["yellow"], 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:
            if len(roi_points) >= 3:
                cv2.destroyWindow(window_name)
                polygon = np.array(roi_points, dtype=np.int32)
                print(f"[ROI] Zone confirmed: {len(roi_points)} points")
                return polygon
            else:
                print("[ROI] Need at least 3 points!")
        elif key == ord("r") or key == ord("R"):
            roi_points = []
            print("[ROI] Points reset")
        elif key == 27:
            cv2.destroyWindow(window_name)
            print("[ROI] Skipped — whole frame used")
            return None

def point_in_roi(cx, cy, roi_polygon):
    if roi_polygon is None:
        return True
    return cv2.pointPolygonTest(roi_polygon.astype(np.float32), (float(cx), float(cy)), False) >= 0

def draw_roi_overlay(frame, roi_polygon):
    if roi_polygon is None:
        return frame
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    outside = mask == 0
    frame[outside] = (frame[outside] * 0.35).astype(np.uint8)
    cv2.polylines(frame, [roi_polygon], True, C["cyan"], 2, cv2.LINE_AA)
    return frame
