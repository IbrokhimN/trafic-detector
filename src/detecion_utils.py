from collections import defaultdict
from datetime import datetime

INCIDENT_SPEED_THRESHOLD = 2.0
INCIDENT_FRAMES = 15
CO2_PER_IDLE_SEC = 2.3
FUEL_PER_IDLE_SEC = 0.0008
CLASS_NAMES = {0: "person", 2: "car", 3: "moto", 5: "bus", 7: "truck"}

class IncidentDetector:
    def __init__(self):
        self.stopped = defaultdict(int)
        self.active = {}

    def update(self, track_id, speed, cx, cy):
        if speed < INCIDENT_SPEED_THRESHOLD:
            self.stopped[track_id] += 1
        else:
            self.stopped[track_id] = 0
            self.active.pop(track_id, None)
        if self.stopped[track_id] >= INCIDENT_FRAMES:
            if track_id not in self.active:
                self.active[track_id] = {"time": datetime.now().isoformat(), "pos": (cx, cy)}
        return track_id in self.active


class EcoEstimator:
    def __init__(self):
        self.idle_frames = 0
        self.seen = set()

    def update(self, track_id, speed):
        self.seen.add(track_id)
        if speed < 3:
            self.idle_frames += 1

    @property
    def co2_saved(self):
        return (self.idle_frames / 30) * CO2_PER_IDLE_SEC * 0.3

    @property
    def fuel_saved(self):
        return (self.idle_frames / 30) * FUEL_PER_IDLE_SEC * 0.3


class VehicleDB:
    def __init__(self):
        self.db = {}

    def update(self, track_id, cls_id, speed, pos):
        if track_id not in self.db:
            self.db[track_id] = {"class": CLASS_NAMES.get(cls_id, "?"), "speeds": [], "first": datetime.now().isoformat()}
        e = self.db[track_id]
        e["speeds"].append(round(speed, 1))
        if len(e["speeds"]) > 60:
            e["speeds"] = e["speeds"][-60:]

    @property
    def stats(self):
        r = defaultdict(int)
        for v in self.db.values():
            r[v["class"]] += 1
        return dict(r)
