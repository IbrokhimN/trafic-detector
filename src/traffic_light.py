import math

TRAFFIC_LIGHT_PHASES = [
    {"name": "NS_GREEN",  "ns": "green",  "ew": "red",    "ped": "red",   "base_duration": 30},
    {"name": "NS_YELLOW", "ns": "yellow", "ew": "red",    "ped": "red",   "base_duration": 4},
    {"name": "EW_GREEN",  "ns": "red",    "ew": "green",  "ped": "red",   "base_duration": 30},
    {"name": "EW_YELLOW", "ns": "red",    "ew": "yellow", "ped": "red",   "base_duration": 4},
    {"name": "PED_CROSS", "ns": "red",    "ew": "red",    "ped": "green", "base_duration": 15},
]

class TrafficLight:
    def __init__(self, phases=None):
        if phases is None:
            phases = TRAFFIC_LIGHT_PHASES
        self.phases = phases
        self.current_idx = 0
        self.timer = phases[0]["base_duration"]
        self.emergency_override = False

    @property
    def phase(self):
        return self.phases[self.current_idx]

    def tick(self, vehicle_density=0, pedestrians=0, has_elderly=False, emergency=False):
        if emergency and not self.emergency_override:
            self.emergency_override = True
            self.current_idx = 0
            self.timer = 10
            return True
        if self.emergency_override and not emergency:
            self.emergency_override = False
        self.timer -= 1
        if self.timer <= 0:
            self.current_idx = (self.current_idx + 1) % len(self.phases)
            phase = self.phases[self.current_idx]
            duration = phase["base_duration"]
            if "GREEN" in phase["name"]:
                duration += min(vehicle_density * 2, 20)
            if phase["name"] == "PED_CROSS":
                if has_elderly:
                    duration += 10
                if pedestrians == 0:
                    duration = 3
            self.timer = max(duration, 3)
            return True
        return False

    def force_next(self):
        self.current_idx = (self.current_idx + 1) % len(self.phases)
        self.timer = self.phases[self.current_idx]["base_duration"]


def calc_green_time(vehicle_count, ped_count, avg_speed, has_elderly=False, emergency=False,
                    crossing_distance=12.0, lanes_count=2):
    result = {
        "v_green": 0, "p_green": 0, "yellow": 3, "all_red": 2,
        "cycle": 0, "reason": "", "priority": "none",
        "v_clearance": 0.0, "p_crossing": 0.0,
        "throughput": 0.0, "efficiency": "",
    }

    if emergency:
        result["v_green"] = 60
        result["p_green"] = 0
        result["reason"] = "EMERGENCY VEHICLE DETECTED"
        result["priority"] = "emergency"
        result["cycle"] = 60 + result["yellow"] + result["all_red"]
        return result

    STARTUP_LOST = 3.0
    HEADWAY = 2.0
    MIN_GREEN_V = 7
    MAX_GREEN_V = 60

    if vehicle_count == 0:
        v_green = 0
        v_clearance = 0
    else:
        queue_per_lane = math.ceil(vehicle_count / max(lanes_count, 1))
        v_clearance = STARTUP_LOST + queue_per_lane * HEADWAY
        v_green = max(MIN_GREEN_V, min(round(v_clearance), MAX_GREEN_V))

    result["v_green"] = v_green
    result["v_clearance"] = round(v_clearance, 1)
    result["throughput"] = round(vehicle_count / max(v_green, 1) * 3600, 0) if v_green > 0 else 0

    PED_SPEED_NORMAL = 1.3
    PED_SPEED_ELDERLY = 0.8
    PED_REACTION = 3.0
    MIN_GREEN_P = 7
    MAX_GREEN_P = 40

    if ped_count == 0:
        p_green = 0
        p_crossing = 0
        result["reason"] = "No pedestrians"
        result["priority"] = "vehicles"
    else:
        speed = PED_SPEED_ELDERLY if has_elderly else PED_SPEED_NORMAL
        p_crossing = crossing_distance / speed
        crowd_extra = max(0, (ped_count - 5)) * 1.0
        p_total = PED_REACTION + p_crossing + crowd_extra
        p_green = max(MIN_GREEN_P, min(round(p_total), MAX_GREEN_P))
        result["p_crossing"] = round(p_crossing, 1)
        result["reason"] = f"Pedestrians: {ped_count}, speed {speed:.1f} m/s"
        result["priority"] = "pedestrians_elderly" if has_elderly else "balanced"

    result["p_green"] = p_green
    result["cycle"] = result["v_green"] + result["yellow"] + result["p_green"] + result["all_red"]

    if vehicle_count == 0 and ped_count == 0:
        result["reason"] = "No traffic"
        result["priority"] = "idle"
        result["efficiency"] = "—"
    elif v_green > 0 and p_green == 0:
        result["efficiency"] = f"Green for {vehicle_count} vehicles, {v_green}s"
    elif v_green == 0 and p_green > 0:
        result["efficiency"] = f"Green for {ped_count} pedestrians, {p_green}s"
    else:
        ratio = v_green / max(v_green + p_green, 1) * 100
        result["efficiency"] = f"Vehicles {ratio:.0f}% / Pedestrians {100-ratio:.0f}%"
    return result
