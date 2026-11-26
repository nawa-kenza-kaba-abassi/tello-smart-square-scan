"""
Tello EDU Smart Square Scan - course distribution baseline A.

Baseline source: tello_themed_gui_2_rev3.1.py.
This copy keeps the validated rev3.1 flight-control logic. It only corrects
student-facing documentation and the invalid CLI theme choices.

CURRENT BUILD
  - One defined theme: SMART SQUARE SCAN (Theme 1).
  - Fixed detector: yolov8n.pt, object detection.
  - Camera source: laptop webcam or Tello front video stream.
  - Main assessed route: A / B / B second scan / C / START.
  - Mission picker is locked to Smart Square Scan for the current theme.
  - Live path is a command-derived estimate. It is not measured XY position.

SAFETY
  - STOP requests a controlled SDK 'land' command. It does not cut the motors.
  - A missing or ambiguous command reply blocks later flight commands.
  - Valid and fresh telemetry is required before START.
  - Classroom temperature policy: START <= 75 C; landing request at 85 C.
  - Always confirm physical grounding before another flight.
"""

import argparse
import csv
import datetime
import errno
import math
import os
import platform
import socket
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

# Tame the Tello H.264 stream BEFORE cv2 is imported (must be env vars):
#  - larger UDP fifo + non-fatal overruns => far fewer "error while decoding
#    MB / bytestream" corrupted frames when Wi-Fi drops packets
#  - loglevel -8 (quiet) => no "non-existing PPS" / "no frame!" console spam
#    (those are normal while the decoder waits for the first keyframe)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "fifo_size;5000000|overrun_nonfatal;1")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

try:
    import cv2
    from PIL import Image, ImageTk
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# Ultralytics YOLO is optional; the GUI degrades gracefully without it.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

TELLO_IP   = '192.168.10.1'
TELLO_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
TELLO_ADDR = (TELLO_IP, TELLO_PORT)

MIN_BAT_FLY  = 20
MAX_TEMP_FLY = 75   # classroom-trial START limit from ref/ref1 (not manufacturer-certified)
MIN_TEMP_FLY = 0
CRITICAL_BAT  = 15
CRITICAL_TEMP = 85  # classroom-trial in-flight LAND limit from ref/ref1
TELEM_READY  = 3
MAX_TELEMETRY_AGE = 2.0
MAX_FLIGHT_MIN = 13.0
SEC_PER_PCT    = (MAX_FLIGHT_MIN * 60) / 100
OBSTACLE_THRESHOLD = 0.25

PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diagnosis', 'photos')
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diagnosis', 'videos')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diagnosis', 'logs')


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — YOLO DETECTOR  (fixed yolov8n.pt for this course baseline)
# ═══════════════════════════════════════════════════════════════

# Single fixed detector: this GUI only detects BOXES.
BOX_MODEL = 'yolov8n.pt'
BOX_TASK  = 'detect'


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — STUDENT-EDITABLE GUI/THEME SETTINGS
# Edit labels/colours/project wording here. Do not change the safety core below.
# ═══════════════════════════════════════════════════════════════

# Shared LIGHT base (white background, dark text). Accent colours per theme
# stay vivid; these neutrals are tuned to read well on white.
BG        = '#ffffff'   # main window background (white)
PANEL_BG  = '#eef1f5'   # section-header strips
CARD_BG   = '#f5f7fa'   # cards / inset panels
BORDER    = '#d3d9e0'   # light grey borders
HEADER_BG = '#f0f2f6'   # title bars / header strips (was near-black)
CANVAS_BG = '#f7f9fb'   # camera + flight-path canvases (was near-black)
LOG_BG    = '#f4f6f9'   # activity-log background
C_WHITE   = '#1a1f26'   # PRIMARY TEXT — now dark for white bg (name kept)
C_MUTED   = '#6b7683'   # secondary/label text
C_DIM     = '#9aa4b0'   # faint text / grid lines
C_BLUE    = '#0277bd'   # info blue (darkened for contrast on white)
C_GREEN   = '#0a9d5a'   # ok/good (darkened for contrast on white)
C_AMBER   = '#c77700'   # warning (darkened for contrast on white)
C_RED     = '#d32f2f'   # critical (darkened for contrast on white)
C_SUBTEXT = '#57606b'   # header question text

THEMES = {
    1: dict(
        name="SMART SQUARE SCAN",
        emoji="[SCAN]",
        question="Can a drone patrol a square path and scan each corner for a box?",
        accent='#7c5cff', accent2='#00e5ff', glow='#f0ecff',
        accent_text='#5a3fd6',
        default_task='detect', default_model='yolov8n.pt',
        props=["POINT A", "POINT B", "POINT B (2)"],
        waypoint_labels=["START", "POINT A", "POINT B", "POINT C", "START"],
        # This theme runs ONE dedicated mission (the smart square scan) and
        # locks the mission picker to it.
        locked_mission="Smart Square Scan (YOLO)",
    ),
}


# ═══════════════════════════════════════════════════════════════
# SMART SQUARE SCAN — STUDENT-EDITABLE route + scan plan
# Keep changes inside the approved classroom flight area and validate before live flight.
# ═══════════════════════════════════════════════════════════════
# Fly a CLOCKWISE square START -> A -> B -> C -> back to START, scanning for a
# small box at chosen waypoints with the live YOLO detector. Everything here is
# customisable: change distances, which points scan, and rotate direction.
#
# Each waypoint dict:
#   name       label shown in logs / scan-results panel
#   leg_cm     distance to travel to reach this point (0 = stay in place)
#   leg_dir    forward / back / left / right  (drone-relative)
#   scan       None | "left" | "right"  (rotate that way, scan, rotate back)
#   scan_deg   degrees to rotate before scanning
#   return_deg degrees to rotate back after scanning (0 = keep new heading)

SMART_SQUARE_SPEED    = 30    # cm/s (10-100)
SMART_SQUARE_SETTLE   = 1.5   # seconds to hold still before grabbing a frame
SMART_SQUARE_CONF     = 0.35  # min confidence to count as a detection
# COCO classes that plausibly match a "small square box" on a desk.
# Empty list = accept ANY detected object.
SMART_SQUARE_CLASSES  = ["suitcase", "book", "cell phone", "remote",
                         "box", "handbag", "tv", "laptop"]

SMART_SQUARE_WAYPOINTS = [
    {"name": "POINT A", "leg_cm": 60, "leg_dir": "forward",
     "scan": "left",  "scan_deg": 90, "return_deg": 90},

    {"name": "POINT B", "leg_cm": 60, "leg_dir": "right",
     "scan": "right", "scan_deg": 90, "return_deg": 90},

    {"name": "POINT B (2nd scan)", "leg_cm": 0, "leg_dir": "forward",
     "scan": "left",  "scan_deg": 90, "return_deg": 90},

    {"name": "POINT C", "leg_cm": 60, "leg_dir": "back",
     "scan": None,    "scan_deg": 0,  "return_deg": 0},

    {"name": "START",   "leg_cm": 60, "leg_dir": "left",
     "scan": None,    "scan_deg": 0,  "return_deg": 0},
]

FONT_MONO    = ('Courier', 10)
FONT_MONO_S  = ('Courier', 8)
FONT_MONO_L  = ('Courier', 13, 'bold')
FONT_MONO_XL = ('Courier', 22, 'bold')
FONT_LABEL   = ('Courier', 7)


def bat_color(pct):
    if pct >= 50: return C_GREEN
    if pct >= 20: return C_AMBER
    return C_RED


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DRONE PROXY  (same safe API as the original GUI)
# ═══════════════════════════════════════════════════════════════

class PositionTracker:
    """Dead-reckoning tracker: turns Tello SDK movement commands into a 2-D
    top-down path (x, y in cm) plus heading and altitude.

    The Tello has no absolute positioning, so we estimate position by adding up
    the movement commands we send — exactly what a flight-path diagram needs.
    Origin (0, 0) is the START point; +y is 'forward' at takeoff, heading is in
    degrees clockwise from that initial forward direction.
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.x = 0.0            # cm, right(+)/left(-)
            self.y = 0.0            # cm, forward(+)/back(-) at takeoff heading
            self.z = 0.0            # cm, altitude
            self.heading = 0.0      # degrees, clockwise from start-forward
            self.trail = [(0.0, 0.0)]   # list of (x, y) points
            self.airborne = False
            self.events = []        # (x, y, label) markers e.g. photo stops

    def snapshot(self):
        """Return a thread-safe copy for the drawing code."""
        with self._lock:
            return dict(x=self.x, y=self.y, z=self.z, heading=self.heading,
                        trail=list(self.trail), airborne=self.airborne,
                        events=list(self.events))

    def mark_event(self, label):
        with self._lock:
            self.events.append((self.x, self.y, label))

    def set_live_altitude(self, height_cm):
        """Sync Z to the drone's continuously-reported height (telemetry 'h').
        This makes the altitude move smoothly in real time during flight,
        instead of only stepping on up/down commands."""
        try:
            h = float(height_cm)
        except (TypeError, ValueError):
            return
        with self._lock:
            self.z = h

    def feed(self, command):
        """Update the estimated position from one SDK command string."""
        parts = command.strip().lower().split()
        if not parts:
            return
        cmd = parts[0]

        def arg(i, default=0.0):
            try:
                return float(parts[i])
            except (IndexError, ValueError):
                return default

        with self._lock:
            if cmd == 'takeoff':
                self.airborne = True
                self.z = 80.0                       # Tello rises to ~80 cm
            elif cmd in ('land', 'emergency'):
                self.airborne = False
                self.z = 0.0
            elif cmd in ('forward', 'back', 'left', 'right', 'up', 'down'):
                d = arg(1)
                self._move(cmd, d)
            elif cmd in ('cw', 'ccw'):
                deg = arg(1)
                self.heading += deg if cmd == 'cw' else -deg
                self.heading %= 360
            elif cmd == 'go':                       # go x y z speed (relative)
                self._move_xyz(arg(1), arg(2), arg(3))
            elif cmd == 'curve':                    # curve x1 y1 z1 x2 y2 z2 sp
                # end point is the second triple (x2,y2,z2), relative to drone
                self._move_xyz(arg(4), arg(5), arg(6))
            # speed / command / streamon etc. don't change position

            self.trail.append((self.x, self.y))
            if len(self.trail) > 4000:
                self.trail = self.trail[-4000:]

    def _move(self, direction, dist):
        """Move relative to current heading. Tello body frame:
        forward = +x_body, right = +y_body."""
        h = math.radians(self.heading)
        # unit vectors in world frame (y world = forward at heading 0)
        fwd = (math.sin(h), math.cos(h))     # forward direction
        rgt = (math.cos(h), -math.sin(h))    # right direction
        if direction == 'forward':
            self.x += fwd[0] * dist; self.y += fwd[1] * dist
        elif direction == 'back':
            self.x -= fwd[0] * dist; self.y -= fwd[1] * dist
        elif direction == 'right':
            self.x += rgt[0] * dist; self.y += rgt[1] * dist
        elif direction == 'left':
            self.x -= rgt[0] * dist; self.y -= rgt[1] * dist
        elif direction == 'up':
            self.z += dist
        elif direction == 'down':
            self.z = max(0.0, self.z - dist)

    def _move_xyz(self, fx, fy, fz):
        """Relative go/curve: fx forward, fy left(+), fz up (Tello convention)."""
        h = math.radians(self.heading)
        fwd = (math.sin(h), math.cos(h))
        lft = (-math.cos(h), math.sin(h))
        self.x += fwd[0] * fx + lft[0] * fy
        self.y += fwd[1] * fx + lft[1] * fy
        self.z += fz


class DroneProxy:
    # Action commands need longer reply windows because the Tello replies after
    # the action completes. Timeouts on action commands create an UNKNOWN state:
    # the command may have executed even though its acknowledgement was lost.
    ACTION_TIMEOUTS = (
        ('takeoff', 20), ('land', 20),
        ('up', 15), ('down', 15), ('left', 15), ('right', 15),
        ('forward', 15), ('back', 15), ('go', 15), ('curve', 15),
        ('cw', 10), ('ccw', 10), ('flip', 10),
    )
    ACTION_COMMANDS = {item[0] for item in ACTION_TIMEOUTS}

    def __init__(self, cmd_sock, tracker=None):
        self._sock     = cmd_sock
        self._lock     = threading.Lock()
        self.abort     = False
        self.uncertain = False
        self.flight_unknown = False
        self.last_problem = ""
        self._flying   = False
        self._fly_lock = threading.Lock()
        self._telem_data = {}
        self.tracker   = tracker      # optional PositionTracker for live path
        self.on_refused = None        # callback(command, reply) -> GUI log

    @property
    def flying(self):
        with self._fly_lock:
            return self._flying

    @flying.setter
    def flying(self, value):
        with self._fly_lock:
            self._flying = value
            self.flight_unknown = False

    def _timeout_for(self, command):
        verb = command.strip().lower().split()[0] if command.strip() else ''
        for prefix, timeout in self.ACTION_TIMEOUTS:
            if verb == prefix:
                return timeout
        return 5

    def send(self, command, delay=2):
        """Serialize replies and state changes; fail closed on a lost reply."""
        verb = command.strip().lower().split()[0] if command.strip() else ''
        landing = verb in ('land', 'emergency')
        with self._lock:
            if not landing and (self.abort or self.uncertain):
                return 'blocked'
            was_uncertain = self.uncertain
            if verb == 'takeoff':
                # The aircraft may lift off before its acknowledgement arrives.
                with self._fly_lock:
                    self.flight_unknown = True
            deadline = time.monotonic() + self._timeout_for(command)
            result = ''
            try:
                self._sock.setblocking(False)
                # Bound the flush so an unsolicited stream cannot delay STOP.
                for _ in range(100):
                    try:
                        self._sock.recvfrom(2048)
                    except (BlockingIOError, OSError):
                        break
                self._sock.settimeout(0.1)
                self._sock.sendto(command.encode('utf-8'), TELLO_ADDR)
                _DIAG_LOCAL.wire_times = getattr(_DIAG_LOCAL, 'wire_times', {})
                _DIAG_LOCAL.wire_times[id(self)] = {'send': time.monotonic_ns(), 'reply': None}
                _diag_emit('sdk_wire_sent', command=command)
                while time.monotonic() < deadline:
                    if self.abort and not landing:
                        result = 'cancelled'
                        break
                    try:
                        resp, address = self._sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    if address != TELLO_ADDR:
                        continue
                    result = resp.decode('utf-8', errors='replace').strip()
                    _DIAG_LOCAL.wire_times[id(self)]['reply'] = time.monotonic_ns()
                    _diag_emit('sdk_wire_reply', command=command, response=result)
                    break
            except OSError:
                result = ''
            finally:
                try:
                    self._sock.settimeout(5)
                except OSError:
                    pass

            result_lower = result.lower()
            refused = (result_lower.startswith('error')
                       or result_lower in ('unactive', 'unknown command', 'out of range'))
            valid_query = False
            if command.strip().lower() == 'battery?':
                try:
                    valid_query = 0 <= int(result) <= 100
                except ValueError:
                    pass
            expected = valid_query if command.endswith('?') else result_lower == 'ok'
            if not expected and not refused:
                # Queries/configuration can also leave a late reply in flight.
                self.uncertain = True
                self.abort = True
                self.last_problem = f"{command}: {result or 'no response'}"
                with self._fly_lock:
                    if self._flying or verb == 'takeoff':
                        self.flight_unknown = True
            elif refused:
                self.abort = True
                self.last_problem = f"{command}: {result}"
                if verb == 'takeoff':
                    self.flying = False
            elif result_lower == 'ok':
                if landing and was_uncertain:
                    result = result_lower = 'unconfirmed'
                    self.uncertain = True
                    self.abort = True
                    self.last_problem = "Landing acknowledgement may belong to an earlier command"
                else:
                    if verb == 'takeoff':
                        self.flying = True
                    if landing:
                        self.flying = False
                    if self.tracker is not None:
                        self.tracker.feed(command)
            # Certainty, flight state and path are committed BEFORE releasing
            # the command lock, so a queued landing sees the complete outcome.

        if (not expected or result_lower == 'unconfirmed') and self.on_refused:
            try:
                self.on_refused(command, result)
            except Exception:
                pass
        time.sleep(delay)
        return result


    @property
    def telem(self):
        return self._telem_data

    @telem.setter
    def telem(self, value):
        self._telem_data = value

    def emergency_land(self):
        """Request a controlled landing and report whether it was confirmed."""
        self.abort = True
        result = self.send('land', delay=0)
        return result.lower() == 'ok' and not self.uncertain


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MISSION LIBRARY  (choose-your-flight patterns)
# ═══════════════════════════════════════════════════════════════
# Each mission is a function taking the DroneProxy. All moves are kept
# between 20–50 cm so the drone stays inside the 2 x 2 m demo area.
# Every mission checks drone.abort so the STOP button always works.

def mission_hover(drone):
    """LEVEL 1 — take off, hover 5 s, land."""
    drone.send("takeoff", delay=5)
    if drone.abort: return
    for _ in range(10):                 # 10 x 0.5 s = 5 s
        if drone.abort: return
        time.sleep(0.5)
    drone.send("land", delay=3)


def mission_square(drone):
    """LEVEL 2 — fly a 40 cm square and return to start."""
    drone.send("speed 30", delay=0.5)
    drone.send("takeoff", delay=4)
    if drone.abort: return
    for direction in ["forward", "right", "back", "left"]:
        if drone.abort: return
        drone.send(f"{direction} 40", delay=4)
    drone.send("land", delay=3)


def mission_rotation_scan(drone):
    """LEVEL 3 — rotate 360 in 45 steps, pausing to 'scan' at each angle."""
    drone.send("takeoff", delay=4)
    if drone.abort: return
    steps = 360 // 45
    for step in range(steps):
        if drone.abort: return
        drone.send("cw 45", delay=1.5)
        time.sleep(1.5)
    drone.send("land", delay=3)


def mission_figure8(drone):
    """LEVEL 4 — two 60 cm-radius semicircles; legacy figure-8 name."""
    drone.send("takeoff", delay=5)
    if drone.abort: return
    drone.send("curve 60 60 0 120 0 0 30", delay=5)
    if drone.abort: return
    drone.send("curve -60 -60 0 -120 0 0 30", delay=5)
    drone.send("land", delay=3)


def mission_altitude_wave(drone):
    """LEVEL 5 — up/down wave, 3 cycles of 40 cm."""
    drone.send("takeoff", delay=5)
    if drone.abort: return
    for _ in range(3):
        if drone.abort: return
        drone.send("up 40", delay=3)
        if drone.abort: return
        drone.send("down 40", delay=3)
    drone.send("land", delay=3)


def mission_spiral(drone):
    """LEVEL 6 — ascending then descending helix."""
    drone.send("speed 40", delay=0.5)
    drone.send("takeoff", delay=4)
    if drone.abort: return
    for _ in range(8):                  # 8 x 45 = 360 ascending
        if drone.abort: return
        drone.send("up 20", delay=1)
        drone.send("forward 40", delay=1)
        drone.send("cw 45", delay=1)
    for _ in range(8):                  # mirror, descending
        if drone.abort: return
        drone.send("down 20", delay=1)
        drone.send("forward 40", delay=1)
        drone.send("ccw 45", delay=1)
    drone.send("land", delay=3)


def make_waypoint_mission(theme_id):
    """Build a theme-specific waypoint run: fly toward each prop, face it,
    pause 3 s so the camera captures it, rotate, and move to the next prop.
    Uses 40 cm forward legs and 20 cm backward offsets; no position feedback."""
    theme = THEMES[theme_id]
    props = theme["props"]

    def mission(drone):
        drone.send("speed 30", delay=0.5)
        drone.send("takeoff", delay=4)
        if drone.abort: return
        for i, prop in enumerate(props):
            if drone.abort: return
            print(f"  -> approaching {prop}")
            drone.send("forward 40", delay=4)     # fly toward the prop
            if drone.abort: return
            time.sleep(3)                          # hover so camera captures face-on
            if i < len(props) - 1:
                drone.send("cw 90", delay=2)       # rotate toward next prop
                drone.send("back 20", delay=2)     # small reposition
        # rotate once more and land at the current position
        drone.send("cw 90", delay=2)
        drone.send("land", delay=3)

    return mission


def make_smart_square_mission(panel):
    """Build the smart square-scan mission bound to a running panel.

    Flies a clockwise square and, at each configured waypoint, rotates and asks
    the panel to run a YOLO scan on the live camera frame. Results (found /
    label / confidence / colour) are pushed to the panel's SCAN RESULTS box and,
    after landing, a full SUMMARY + ANALYSIS is logged.
    """
    waypoints = SMART_SQUARE_WAYPOINTS

    def mission(drone):
        results = []
        panel.after(0, panel.clear_scan_results)

        drone.send(f"speed {SMART_SQUARE_SPEED}", delay=0.5)
        drone.send("takeoff", delay=5)
        if drone.abort:
            panel.after(0, lambda: panel.show_scan_summary(results, aborted=True))
            return

        for wp in waypoints:
            _diag_waypoint(wp["name"])
            if drone.abort:
                panel.after(0, lambda r=list(results):
                            panel.show_scan_summary(r, aborted=True))
                return

            # ── travel leg (skip if leg_cm == 0, e.g. a 2nd scan in place) ──
            if wp["leg_cm"] > 0:
                drone.send(f"{wp['leg_dir']} {wp['leg_cm']}", delay=5)
                if drone.abort:
                    panel.after(0, lambda r=list(results):
                                panel.show_scan_summary(r, aborted=True))
                    return

            # ── scan: rotate, settle, detect, rotate back ──
            if wp["scan"] in ("left", "right"):
                turn = "ccw" if wp["scan"] == "left" else "cw"
                drone.send(f"{turn} {wp['scan_deg']}", delay=2)
                if drone.abort:
                    panel.after(0, lambda r=list(results):
                                panel.show_scan_summary(r, aborted=True))
                    return

                time.sleep(SMART_SQUARE_SETTLE)
                res = panel.scan_at_waypoint(wp["name"], wp["scan"])
                results.append(res)

                if wp["return_deg"]:
                    back = "cw" if wp["scan"] == "left" else "ccw"
                    drone.send(f"{back} {wp['return_deg']}", delay=2)

        drone.send("land", delay=3)
        panel.after(0, lambda r=list(results):
                    panel.show_scan_summary(r, aborted=False))

    return mission


# Registry shown in the mission dropdown. Key = label, value = (fn, needs_theme)
MISSION_LIBRARY = {
    "Level 1 — Hover in Place":        (mission_hover, False),
    "Level 2 — Square (40 cm)":        (mission_square, False),
    "Level 3 — 360 Rotation Scan":     (mission_rotation_scan, False),
    "Level 4 — Figure-8":              (mission_figure8, False),
    "Level 5 — Altitude Wave":         (mission_altitude_wave, False),
    "Level 6 — Spiral Ascend/Descend": (mission_spiral, False),
    "Theme Waypoint Run (props)":      (None, True),   # built per-theme at start
    "Smart Square Scan (YOLO)":        (None, "smart_square"),  # built at start
}


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — YOLO DETECTOR WRAPPER
# ═══════════════════════════════════════════════════════════════

class YoloDetector:
    """Loads a YOLOv8 model and annotates frames.

    If ultralytics is unavailable or inference fails, the raw camera frame is
    returned with zero detections. YOLO failure is never hidden by another detector.
    """

    def __init__(self):
        self.model      = None
        self.model_name = None
        self.task       = None
        self.using_yolo = False
        self.last_error = ""

    def load(self, model_name, task):
        """Load weights. Returns (ok, message)."""
        self.last_error = ""
        self.model_name = model_name
        self.task       = task
        if not YOLO_AVAILABLE:
            self.using_yolo = False
            return (False,
                    "ultralytics not installed — camera preview remains available, "
                    "but YOLO detection is disabled. Run: pip install ultralytics")
        try:
            # YOLO() downloads the weights on first use if not present.
            self.model = YOLO(model_name)
            self.using_yolo = True
            return (True, f"Loaded {model_name} ({task}).")
        except Exception as err:
            self.model = None
            self.using_yolo = False
            return (False, f"Could not load {model_name}: {err}")

    def annotate(self, frame):
        """Return (annotated_frame, n_boxes). Draws ONLY box-like objects
        (classes in SMART_SQUARE_CLASSES); everything else is ignored so the
        preview shows exactly what the Smart Square Scan will count. Never
        raises."""
        if self.using_yolo and self.model is not None:
            try:
                results = self.model.predict(frame, conf=0.25, verbose=False)
                r = results[0]
                names = getattr(r, "names", {}) or {}
                out = frame.copy()
                n = 0
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        label = names.get(int(b.cls[0]), str(int(b.cls[0])))
                        if (SMART_SQUARE_CLASSES
                                and label not in SMART_SQUARE_CLASSES):
                            continue        # not a box-like object — hide it
                        conf = float(b.conf[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(out, f"BOX {conf:.0%} ({label})",
                                    (x1, max(14, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 200, 0), 2)
                        n += 1
                return out, n
            except Exception as err:
                self.last_error = f"YOLO inference failed: {err}"
                self.using_yolo = False
                return frame, 0
        return frame, 0

    @staticmethod
    def _edge_preview(frame):
        """Fallback: highlight large contours as 'obstacles'."""
        out  = frame.copy()
        fh, fw = frame.shape[:2]
        area_total = fh * fw
        grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (21, 21), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for c in contours:
            if cv2.contourArea(c) < area_total * 0.02:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if (w * h) > area_total * OBSTACLE_THRESHOLD:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(out, "OBJECT", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                n += 1
            else:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 229, 255), 1)
        return out, n


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — BATTERY GAUGE  (themed arc)
# ═══════════════════════════════════════════════════════════════

class BatteryGauge(tk.Canvas):
    def __init__(self, parent, accent, size=150, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=BG, highlightthickness=0, **kw)
        self.size   = size
        self.accent = accent
        self._draw(0)

    def set(self, pct):
        self._draw(pct)

    def _draw(self, pct):
        self.delete('all')
        s, pad = self.size, 16
        cx, cy, r = s / 2, s / 2, s / 2 - pad
        col = bat_color(pct)
        self.create_arc(pad, pad, s - pad, s - pad, start=-210, extent=240,
                        style='arc', outline=C_DIM, width=8)
        if pct > 0:
            self.create_arc(pad, pad, s - pad, s - pad, start=-210,
                            extent=(pct / 100) * 240, style='arc',
                            outline=col, width=8)
        for t in range(0, 101, 25):
            a = math.radians(-210 + (t / 100) * 240)
            self.create_line(cx + (r - 5) * math.cos(a), cy - (r - 5) * math.sin(a),
                             cx + (r + 2) * math.cos(a), cy - (r + 2) * math.sin(a),
                             fill=C_MUTED, width=1)
        self.create_text(cx, cy - 6, text=f'{pct}%', fill=col, font=FONT_MONO_XL)
        self.create_text(cx, cy + 16, text='BATTERY', fill=C_MUTED, font=FONT_LABEL)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — SCROLLABLE COLUMN HELPER
# ═══════════════════════════════════════════════════════════════

class ScrollColumn(tk.Frame):
    """A vertically scrollable container.

    Put widgets inside `.body`. A scrollbar appears automatically and the
    mouse wheel scrolls when the pointer is over the column. This guarantees
    that content is always reachable, even on small screens — fixing the
    'can't see info at the bottom' problem on fixed-height layouts.
    """

    def __init__(self, parent, fixed_width=None, **kw):
        super().__init__(parent, bg=BG, highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        self._fixed_width = fixed_width
        if fixed_width:
            # Hold this column to an exact pixel width regardless of the
            # natural size of the widgets packed inside it.
            self.configure(width=fixed_width)
            self.grid_propagate(False)
            self.pack_propagate(False)
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self._sb = tk.Scrollbar(self, orient='vertical',
                                command=self._canvas.yview,
                                troughcolor=CARD_BG, bg=BORDER)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        # `body` is where callers add their widgets.
        self.body = tk.Frame(self._canvas, bg=BG)
        self._win = self._canvas.create_window((0, 0), window=self.body,
                                               anchor='nw')

        # Keep the scrollregion in sync and make the inner frame match width.
        self.body.bind('<Configure>', self._on_body_config)
        self._canvas.bind('<Configure>', self._on_canvas_config)

        # Mouse-wheel scrolling only while the cursor is over this column.
        self._canvas.bind('<Enter>', self._bind_wheel)
        self._canvas.bind('<Leave>', self._unbind_wheel)

    def _on_body_config(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_config(self, e):
        # Make the inner body fill the canvas width (so fill='x' works inside).
        self._canvas.itemconfig(self._win, width=e.width)

    def _bind_wheel(self, _e):
        # Windows / macOS
        self._canvas.bind_all('<MouseWheel>', self._on_wheel)
        # Linux (X11) uses Button-4 / Button-5
        self._canvas.bind_all('<Button-4>', self._on_wheel)
        self._canvas.bind_all('<Button-5>', self._on_wheel)

    def _unbind_wheel(self, _e):
        self._canvas.unbind_all('<MouseWheel>')
        self._canvas.unbind_all('<Button-4>')
        self._canvas.unbind_all('<Button-5>')

    def _on_wheel(self, e):
        if getattr(e, 'num', None) == 4:
            self._canvas.yview_scroll(-1, 'units')
        elif getattr(e, 'num', None) == 5:
            self._canvas.yview_scroll(1, 'units')
        else:
            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — LIVE FLIGHT-PATH WINDOW
# ═══════════════════════════════════════════════════════════════

class FlightPathView(tk.Frame):
    """A reusable flight-path display (top view + altitude side view + telemetry).

    This is a Frame so it can be embedded directly in the main window OR placed
    inside a pop-out Toplevel. It reads live position from a PositionTracker and
    redraws ~10x/second while `start()` is active.
    """

    AREA_CM = 200          # 2 m x 2 m demo area, in cm
    MARGIN  = 24           # px border around the grid

    def __init__(self, parent, tracker, accent, compact=False, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.tracker  = tracker
        self.accent   = accent
        self.compact  = compact
        self._running = False
        self.follow_var = tk.BooleanVar(value=True)

        # top-view canvas (expands to fill)
        self.canvas = tk.Canvas(self, bg=CANVAS_BG, highlightthickness=1,
                                highlightbackground=accent)
        self.canvas.pack(fill='both', expand=True, padx=6, pady=(4, 4))

        # telemetry strip — explicit X / Y / Z position readout
        strip = tk.Frame(self, bg=BG)
        strip.pack(fill='x', padx=6)
        tk.Label(strip, text="POS", bg=BG, fg=C_MUTED,
                 font=('Courier', 8, 'bold')).pack(side='left', padx=(0, 4))
        self.x_lbl = tk.Label(strip, text="X:0", bg=BG, fg='#ff8a5c',
                              font=('Courier', 9, 'bold'))
        self.x_lbl.pack(side='left')
        self.y_lbl = tk.Label(strip, text="Y:0", bg=BG, fg=C_GREEN,
                              font=('Courier', 9, 'bold'))
        self.y_lbl.pack(side='left', padx=(6, 0))
        self.z_lbl = tk.Label(strip, text="Z:0", bg=BG, fg=C_BLUE,
                              font=('Courier', 9, 'bold'))
        self.z_lbl.pack(side='left', padx=(6, 0))
        self.hdg_lbl = tk.Label(strip, text="HDG:0°", bg=BG, fg=C_AMBER,
                                font=('Courier', 9, 'bold'))
        self.hdg_lbl.pack(side='left', padx=(10, 0))
        self.live_lbl = tk.Label(strip, text="● IDLE", bg=BG, fg=C_MUTED,
                                 font=('Courier', 8, 'bold'))
        self.live_lbl.pack(side='right')
        self._blink = False

        # units hint
        tk.Label(self, text="X = right(+)/left(-)   Y = fwd(+)/back(-)   "
                            "Z = altitude   (all cm)",
                 bg=BG, fg=C_DIM, font=FONT_LABEL, anchor='w').pack(
                     fill='x', padx=6)

        # side-view (altitude)
        tk.Label(self, text="SIDE VIEW · ALTITUDE", bg=BG, fg=C_MUTED,
                 font=FONT_LABEL, anchor='w').pack(fill='x', padx=6, pady=(4, 0))
        self.alt_canvas = tk.Canvas(self, bg=CANVAS_BG,
                                    height=60 if compact else 80,
                                    highlightthickness=1,
                                    highlightbackground=BORDER)
        self.alt_canvas.pack(fill='x', padx=6, pady=(2, 4))

        # controls
        ctl = tk.Frame(self, bg=BG)
        ctl.pack(fill='x', padx=6, pady=(0, 6))
        tk.Checkbutton(ctl, text="auto-fit", variable=self.follow_var,
                       bg=BG, fg=C_MUTED, selectcolor=CARD_BG,
                       activebackground=BG, activeforeground=accent,
                       font=FONT_MONO_S).pack(side='left')
        tk.Button(ctl, text="CLEAR", bg=CARD_BG, fg=C_AMBER, relief='flat',
                  font=('Courier', 8, 'bold'), cursor='hand2',
                  command=self.tracker.reset, highlightbackground=BORDER,
                  highlightthickness=1).pack(side='right')

    def start(self):
        if not self._running:
            self._running = True
            self._redraw()

    def stop(self):
        self._running = False

    def _redraw(self):
        if not self._running:
            return
        self._frame = getattr(self, '_frame', 0) + 1
        try:
            self._draw_top_view()
            self._draw_side_view()
        except tk.TclError:
            return
        self.after(100, self._redraw)

    def _draw_top_view(self):
        c = self.canvas
        c.delete('all')
        w = max(100, c.winfo_width())
        h = max(100, c.winfo_height())
        size = min(w, h) - 2 * self.MARGIN
        if size < 30:
            return
        ox = (w - size) / 2
        oy = (h - size) / 2
        snap = self.tracker.snapshot()
        half = self.AREA_CM / 2
        if self.follow_var.get() and snap['trail']:
            xs = [p[0] for p in snap['trail']] + [snap['x']]
            ys = [p[1] for p in snap['trail']] + [snap['y']]
            reach = max(half, max(abs(v) for v in xs + ys) + 20)
        else:
            reach = half
        scale = size / (2 * reach)

        def to_px(x, y):
            return (ox + size / 2 + x * scale, oy + size / 2 - y * scale)

        bx0, by0 = to_px(-half, half)
        bx1, by1 = to_px(half, -half)
        c.create_rectangle(bx0, by0, bx1, by1, outline=C_DIM, width=1)
        for g in range(-100, 101, 50):
            col = BORDER if g != 0 else '#b0b8c2'
            c.create_line(*to_px(g, half), *to_px(g, -half), fill=col)
            c.create_line(*to_px(-half, g), *to_px(half, g), fill=col)
        c.create_text(bx0 + 4, by1 - 9, text="2 m", fill=C_MUTED,
                      font=FONT_LABEL, anchor='w')

        sx, sy = to_px(0, 0)
        c.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill=C_GREEN, outline='')
        c.create_text(sx + 8, sy - 8, text="START", fill=C_GREEN,
                      font=FONT_LABEL, anchor='w')

        for ex, ey, label in snap['events']:
            px, py = to_px(ex, ey)
            c.create_rectangle(px - 3, py - 3, px + 3, py + 3,
                               outline=self.accent, width=1)
            c.create_text(px + 6, py, text=label, fill=self.accent,
                          font=FONT_LABEL, anchor='w')

        if len(snap['trail']) >= 2:
            pts = []
            for x, y in snap['trail']:
                pts.extend(to_px(x, y))
            c.create_line(*pts, fill=self.accent, width=2, smooth=True)

        dx, dy = to_px(snap['x'], snap['y'])
        col = self.accent if snap['airborne'] else C_MUTED
        r = 7
        c.create_oval(dx - r, dy - r, dx + r, dy + r, outline=col, width=2)
        c.create_oval(dx - 2, dy - 2, dx + 2, dy + 2, fill=col, outline='')
        hrad = math.radians(snap['heading'])
        c.create_line(dx, dy, dx + math.sin(hrad) * 20, dy - math.cos(hrad) * 20,
                      fill=col, width=2, arrow='last')

        self.x_lbl.configure(text=f"X:{snap['x']:+.0f}")
        self.y_lbl.configure(text=f"Y:{snap['y']:+.0f}")
        self.z_lbl.configure(text=f"Z:{snap['z']:.0f}")
        self.hdg_lbl.configure(text=f"HDG:{snap['heading']:.0f}°")
        # LIVE indicator: pulse green while airborne (blink ~ every 0.5s)
        if snap['airborne']:
            on = (getattr(self, '_frame', 0) // 5) % 2 == 0
            self.live_lbl.configure(text="● LIVE" if on else "○ LIVE",
                                    fg=C_GREEN)
        else:
            self.live_lbl.configure(text="● IDLE", fg=C_MUTED)

    def _draw_side_view(self):
        c = self.alt_canvas
        c.delete('all')
        w = max(100, c.winfo_width())
        h = max(50, c.winfo_height())
        snap = self.tracker.snapshot()
        floor = h - 14
        c.create_line(10, floor, w - 10, floor, fill=C_DIM)
        c.create_text(12, floor + 5, text="floor", fill=C_MUTED,
                      font=FONT_LABEL, anchor='w')
        max_alt = 200.0
        z = min(snap['z'], max_alt)
        drone_y = floor - (z / max_alt) * (floor - 12)
        dx = w / 2
        col = self.accent if snap['airborne'] else C_MUTED
        c.create_line(dx, floor, dx, drone_y, fill=BORDER, dash=(2, 2))
        c.create_oval(dx - 6, drone_y - 4, dx + 6, drone_y + 4,
                      outline=col, width=2)
        c.create_text(dx + 14, drone_y, text=f"{snap['z']:.0f} cm",
                      fill=col, font=FONT_LABEL, anchor='w')


class FlightPathWindow(tk.Toplevel):
    """Optional pop-out wrapper around FlightPathView (kept for a bigger view)."""

    def __init__(self, parent, tracker, theme):
        super().__init__(parent)
        self._parent = parent
        accent = theme["accent"]
        accent_text = theme.get("accent_text", accent)
        self.title(f"LIVE FLIGHT PATH — THEME {theme.get('_id','')} "
                   f"{theme['name']}")
        self.configure(bg=BG)
        self.geometry("560x680")
        self.minsize(360, 460)
        self.resizable(True, True)
        hdr = tk.Frame(self, bg=HEADER_BG, highlightbackground=accent,
                       highlightthickness=1)
        hdr.pack(fill='x')
        tk.Label(hdr, text="  LIVE FLIGHT PATH  ·  TOP VIEW (2 x 2 m)",
                 bg=HEADER_BG, fg=accent_text,
                 font=('Courier', 11, 'bold')).pack(side='left', pady=6)
        self.view = FlightPathView(self, tracker, accent_text, compact=False)
        self.view.pack(fill='both', expand=True, padx=6, pady=6)
        self.view.start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.view.stop()
        if hasattr(self._parent, "_on_path_window_closed"):
            self._parent._on_path_window_closed()
        self.destroy()


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — MAIN GUI WINDOW
# ═══════════════════════════════════════════════════════════════

class TelloThemedPanel(tk.Tk):

    def __init__(self, theme_id=1, group="GROUP 07", students=None):
        super().__init__()
        self.theme_id = theme_id
        self.theme    = THEMES[theme_id]
        self.accent   = self.theme["accent"]
        self.accent_text = self.theme.get("accent_text", self.theme["accent"])
        self.group    = group
        self.students = students or [
            "Mission Designer  - Student A",
            "Safety Officer    - Student B",
            "Data Analyst      - Student C",
            "Camera Operator   - Student D",
            "Engineering Anal. - Student E",
        ]

        self.title(f"TELLO MISSION CONTROL — THEME {theme_id} — {self.theme['name']}")
        self.configure(bg=BG)
        self.resizable(True, True)
        # Fit the laptop screen: size to the available area (minus a margin for
        # the OS taskbar/dock), and keep a small min so it works even on
        # 1366x768 laptops. All four columns scroll, so content stays reachable
        # at any size.
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            # leave room for taskbar/dock and window chrome
            avail_w = sw - 40
            avail_h = sh - 90
            w = min(1680, avail_w)
            h = min(940, avail_h)
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2 - 20)
            self.geometry(f"{w}x{h}+{x}+{y}")
            # min size never larger than the screen itself
            self.minsize(min(1040, avail_w), min(520, avail_h))
        except Exception:
            self.geometry("1360x760")
            self.minsize(1040, 520)

        # ── runtime state ──
        self._drone        = None
        self._cmd_sock     = None
        self._state_sock   = None
        self._connected    = False
        self._connection = None
        self._connection_lock = threading.Lock()
        self._closing = False
        self._stop_flag    = False
        self._telem        = {}
        self._telem_received = 0.0
        self._telem_ready  = False
        self._pkt_count    = 0
        self._stale_telem_landing = False
        self._landing_requested = False
        self._landing_in_progress = False
        self._landing_unconfirmed = False
        self._bat_history  = []
        self._flight_start = None
        self._mission_thread = None
        self._activity_log = []

        # camera / yolo
        self._cam_on       = False
        self._camera_session = None
        self._camera_restart = False
        self._cap          = None
        self._cam_lock     = threading.Lock()
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._recording    = False
        self._video_writer = None
        self.detector      = YoloDetector()
        self._model_loaded = False

        # live flight-path tracking
        self.tracker       = PositionTracker()
        self._path_window  = None

        # selections
        self.sel_task    = tk.StringVar(value='detect')      # locked: box detection
        self.sel_model   = tk.StringVar(value='yolov8n.pt')  # locked: single model
        # If this theme locks to one mission, default the picker to it.
        _locked = self.theme.get("locked_mission")
        _default_mission = _locked or list(MISSION_LIBRARY.keys())[0]
        self.sel_mission = tk.StringVar(value=_default_mission)
        # Camera source: which feed YOLO runs on.
        self.sel_camera  = tk.StringVar(value="Laptop Webcam")

        # scan-results state (used by the Smart Square Scan mission)
        self._scan_results = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # Style ttk widgets (comboboxes) to match the light theme.
        try:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('TCombobox',
                            fieldbackground=CARD_BG, background=CARD_BG,
                            foreground=C_WHITE, arrowcolor=self.accent_text,
                            bordercolor=BORDER, lightcolor=BORDER,
                            darkcolor=BORDER)
            style.map('TCombobox',
                      fieldbackground=[('readonly', CARD_BG)],
                      foreground=[('readonly', C_WHITE)])
        except Exception:
            pass

        self._build_header()
        self._build_titlebar()

        # Four columns: telemetry | camera+setup | flight path | controls+log.
        # All stretch vertically; the flight path is now ON the main screen.
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=170)   # left telemetry (fixed)
        body.grid_columnconfigure(1, weight=4, minsize=360)   # camera + setup
        body.grid_columnconfigure(2, weight=3, minsize=240)   # flight path
        body.grid_columnconfigure(3, weight=1, minsize=340)   # controls + log (wide, stretches)

        self._build_left_col(body)     # telemetry + props
        self._build_mid_col(body)      # camera + YOLO + mission picker
        self._build_path_col(body)     # embedded live flight path
        self._build_right_col(body)    # status + controls + log

    def _build_header(self):
        glow = self.theme["glow"]
        hdr = tk.Frame(self, bg=glow)
        hdr.pack(fill='x')
        bar = tk.Frame(self, bg=self.accent, height=3)
        bar.pack(fill='x')

        left = tk.Frame(hdr, bg=glow)
        left.pack(side='left', padx=16, pady=8)
        tk.Label(left, text="PROGRAMMING TECHNIQUES  ·  DJI TELLO MISSION",
                 bg=glow, fg=C_MUTED, font=FONT_LABEL).pack(anchor='w')
        tk.Label(left, text=f"{self.theme['emoji']}  THEME {self.theme_id} — {self.theme['name']}",
                 bg=glow, fg=self.accent_text, font=('Courier', 18, 'bold')).pack(anchor='w')
        tk.Label(left, text=self.theme["question"],
                 bg=glow, fg=C_SUBTEXT, font=FONT_MONO_S).pack(anchor='w')

        # group + students box
        gbox = tk.Frame(hdr, bg=HEADER_BG, highlightbackground=BORDER,
                        highlightthickness=1)
        gbox.pack(side='right', padx=16, pady=8)
        tk.Label(gbox, text=self.group, bg=HEADER_BG, fg=self.accent_text,
                 font=('Courier', 12, 'bold')).pack(anchor='w', padx=10, pady=(4, 0))
        for s in self.students:
            tk.Label(gbox, text=s, bg=HEADER_BG, fg=C_WHITE,
                     font=FONT_LABEL).pack(anchor='w', padx=10)
        tk.Frame(gbox, bg=HEADER_BG, height=4).pack()

    def _build_titlebar(self):
        tb = tk.Frame(self, bg=HEADER_BG, highlightbackground=self.accent,
                      highlightthickness=1)
        tb.pack(fill='x')
        tk.Label(tb, text="  TELLO FLIGHT CONTROL  ·  YOLOV8",
                 bg=HEADER_BG, fg=self.accent_text,
                 font=('Courier', 12, 'bold')).pack(side='left', pady=6)
        self.conn_lbl = tk.Label(tb, text='OFFLINE', bg=HEADER_BG, fg=C_MUTED,
                                 font=('Courier', 9, 'bold'))
        self.conn_lbl.pack(side='right', padx=(0, 14))
        self.conn_dot = tk.Label(tb, text='●', bg=HEADER_BG, fg=C_MUTED,
                                 font=('Courier', 12))
        self.conn_dot.pack(side='right')

    def _section_hdr(self, parent, text):
        f = tk.Frame(parent, bg=PANEL_BG, highlightbackground=BORDER,
                     highlightthickness=1)
        f.pack(fill='x', pady=(0, 4))
        tk.Label(f, text=f'  {text}', bg=PANEL_BG, fg=C_MUTED,
                 font=FONT_LABEL, anchor='w').pack(fill='x', pady=4)

    def _card(self, parent, label, initial, color):
        f = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER,
                     highlightthickness=1)
        f.pack(fill='x', padx=10, pady=2, ipadx=4, ipady=2)
        tk.Label(f, text=label, bg=CARD_BG, fg=C_MUTED,
                 font=FONT_LABEL).pack(anchor='w', padx=6, pady=(2, 0))
        val = tk.Label(f, text=initial, bg=CARD_BG, fg=color, font=FONT_MONO_L)
        val.pack(anchor='w', padx=6, pady=(0, 2))
        return val

    # ── LEFT COLUMN: telemetry + props ────────────────────────

    def _build_left_col(self, parent):
        sc = ScrollColumn(parent, fixed_width=170)
        sc.grid(row=0, column=0, sticky='nsew', padx=(8, 4), pady=8)
        col = sc.body

        self._section_hdr(col, "POWER & TELEMETRY")
        self.gauge = BatteryGauge(col, self.accent, size=130)
        self.gauge.pack(pady=(4, 6), padx=6)

        self.eft_card    = self._card(col, "EST. FLIGHT TIME", "waiting", C_GREEN)
        self.timer_card  = self._card(col, "TIME IN AIR",      "00:00",   self.accent_text)
        self.height_card = self._card(col, "HEIGHT",           "waiting", C_BLUE)
        self.temp_card   = self._card(col, "TEMPERATURE",      "waiting", C_AMBER)
        self.drain_card  = self._card(col, "DRAIN RATE",       "waiting", C_MUTED)

        self.rtf_banner = tk.Label(col, text="[ NOT CONNECTED ]", bg=CARD_BG,
                                   fg=C_MUTED, font=('Courier', 9, 'bold'),
                                   anchor='center', pady=6)
        self.rtf_banner.pack(fill='x', padx=10, pady=(6, 6))

        # props expected in the demo area (from the theme)
        self._section_hdr(col, "PROPS IN AREA")
        for i, p in enumerate(self.theme["props"]):
            row = tk.Frame(col, bg=CARD_BG, highlightbackground=BORDER,
                           highlightthickness=1)
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=f'  {i+1}. {p}', bg=CARD_BG, fg=C_WHITE,
                     font=FONT_MONO_S, anchor='w').pack(side='left',
                     fill='x', expand=True, padx=4, pady=4)
            tk.Label(row, text='OK', bg=CARD_BG, fg=C_GREEN,
                     font=FONT_LABEL).pack(side='right', padx=8)

    # ── MIDDLE COLUMN: camera + YOLO selector + mission picker ─

    def _build_mid_col(self, parent):
        sc = ScrollColumn(parent)
        sc.grid(row=0, column=1, sticky='nsew', padx=4, pady=8)
        col = sc.body

        self._section_hdr(col, "LIVE CAMERA · YOLOV8 DETECTION")
        # Camera canvas scales with the window. We track its real size and
        # resize frames to fit, so the feed grows/shrinks as you resize.
        self.cam_w, self.cam_h = 480, 300
        self.cam_canvas = tk.Canvas(col, width=self.cam_w, height=self.cam_h,
                                    bg=CANVAS_BG, highlightthickness=1,
                                    highlightbackground=self.accent)
        self.cam_canvas.pack(fill='x', expand=True, padx=8, pady=(4, 2))
        self.cam_canvas.bind('<Configure>', self._on_cam_resize)
        self._show_cam_placeholder()

        self.obstacle_lbl = tk.Label(col, text="[ CAMERA INACTIVE ]", bg=BG,
                                     fg=C_MUTED, font=FONT_MONO, anchor='center')
        self.obstacle_lbl.pack(fill='x', padx=8, pady=(2, 2))

        # camera control buttons
        cam_row = tk.Frame(col, bg=BG)
        cam_row.pack(pady=(2, 6))
        for attr, text, fg, cmd in [
            ('cam_on_btn',  'CAM ON',  self.accent, self._camera_on),
            ('cam_off_btn', 'CAM OFF', C_RED,       self._camera_off),
            ('snap_btn',    'PHOTO',   C_WHITE,     self._snap_photo),
            ('record_btn',  'REC',     '#cc44cc',   self._toggle_record),
        ]:
            b = tk.Button(cam_row, text=text, bg=CARD_BG, fg=fg, relief='flat',
                          padx=10, pady=5, font=('Courier', 9, 'bold'),
                          cursor='hand2', state='disabled', command=cmd,
                          highlightbackground=BORDER, highlightthickness=1)
            b.pack(side='left', padx=3)
            setattr(self, attr, b)
        self.cam_on_btn.configure(state='normal')

        # ── PRE-FLIGHT SETUP: YOLO model + camera test + mission ──
        self._section_hdr(col, "PRE-FLIGHT SETUP")
        setup = tk.Frame(col, bg=BG)
        setup.pack(fill='x', padx=10, pady=(0, 12))
        setup.grid_columnconfigure(0, weight=1)
        setup.grid_columnconfigure(1, weight=1)

        # (1) YOLO task + model selectors
        tk.Label(setup, text="1 · DETECTOR (fixed)", bg=BG, fg=C_MUTED,
                 font=FONT_LABEL).grid(row=0, column=0, columnspan=2,
                                       sticky='w', pady=(2, 0))
        tk.Label(setup, text="YOLOv8n  ·  BOX DETECTION ONLY",
                 bg=CARD_BG, fg=self.accent_text, font=('Courier', 9, 'bold'),
                 anchor='w', padx=8, pady=5, highlightbackground=BORDER,
                 highlightthickness=1).grid(row=1, column=0, columnspan=2,
                                            sticky='we')

        # (2) camera source: laptop webcam OR drone camera
        tk.Label(setup, text="2 · CAMERA SOURCE (run YOLO on)", bg=BG,
                 fg=C_MUTED, font=FONT_LABEL).grid(
                     row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.camera_combo = ttk.Combobox(
            setup, textvariable=self.sel_camera,
            values=["Laptop Webcam", "Drone Camera (Tello)"],
            state='readonly', font=FONT_MONO_S)
        self.camera_combo.grid(row=3, column=0, columnspan=2, sticky='we')
        self.camera_combo.bind('<<ComboboxSelected>>', self._on_camera_change)

        # (3) test camera + yolo
        self.test_btn = tk.Button(setup, text="3 · TEST CAMERA + LOAD YOLO",
                                  bg=CARD_BG, fg=self.accent_text, relief='flat',
                                  padx=8, pady=6, font=('Courier', 9, 'bold'),
                                  cursor='hand2', command=self._test_camera_yolo,
                                  highlightbackground=BORDER, highlightthickness=1)
        self.test_btn.grid(row=4, column=0, columnspan=2, sticky='we', pady=(8, 2))

        self.model_status = tk.Label(setup, text="Model not loaded yet.",
                                     bg=BG, fg=C_MUTED, font=FONT_MONO_S,
                                     anchor='w', justify='left', wraplength=360)
        self.model_status.grid(row=5, column=0, columnspan=2, sticky='we')

        # (4) mission picker
        tk.Label(setup, text="4 · CHOOSE MISSION", bg=BG, fg=C_MUTED,
                 font=FONT_LABEL).grid(row=6, column=0, columnspan=2,
                 sticky='w', pady=(8, 0))
        locked = self.theme.get("locked_mission")
        self.mission_combo = ttk.Combobox(
            setup, textvariable=self.sel_mission,
            values=[locked] if locked else list(MISSION_LIBRARY.keys()),
            state='disabled' if locked else 'readonly', font=FONT_MONO_S)
        self.mission_combo.grid(row=7, column=0, columnspan=2, sticky='we')

        if locked:
            tk.Label(setup,
                     text=f"This theme runs one mission: {locked}",
                     bg=BG, fg=self.accent_text, font=FONT_LABEL,
                     wraplength=360, justify='left').grid(
                         row=8, column=0, columnspan=2, sticky='w', pady=(4, 0))
            tk.Label(setup, text="5 · press START MISSION (right panel) to fly",
                     bg=BG, fg=C_DIM, font=FONT_LABEL).grid(
                         row=9, column=0, columnspan=2, sticky='w', pady=(6, 0))
        else:
            tk.Label(setup, text="5 · press START MISSION (right panel) to fly",
                     bg=BG, fg=C_DIM, font=FONT_LABEL).grid(
                         row=8, column=0, columnspan=2, sticky='w', pady=(8, 0))

        # ── SCAN RESULTS panel (populated by the Smart Square Scan mission) ──
        self._build_scan_results_panel(col)

    def _build_scan_results_panel(self, col):
        """A live table that shows what each waypoint scan detected."""
        self._section_hdr(col, "SCAN RESULTS · YOLO BOX DETECTION")
        wrap = tk.Frame(col, bg=BG)
        wrap.pack(fill='x', padx=10, pady=(0, 12))

        # column headers
        head = tk.Frame(wrap, bg=CARD_BG, highlightbackground=BORDER,
                        highlightthickness=1)
        head.pack(fill='x')
        for text, w in [("WAYPOINT", 16), ("SCAN", 6), ("FOUND", 6),
                        ("LABEL", 12), ("CONF", 6), ("COLOUR", 8)]:
            tk.Label(head, text=text, bg=CARD_BG, fg=C_MUTED, font=FONT_LABEL,
                     width=w, anchor='w').pack(side='left', padx=2, pady=3)

        self.scan_rows_frame = tk.Frame(wrap, bg=BG)
        self.scan_rows_frame.pack(fill='x')

        self.scan_placeholder = tk.Label(
            self.scan_rows_frame,
            text="No scans yet — run the Smart Square Scan mission.",
            bg=BG, fg=C_DIM, font=FONT_MONO_S, anchor='w', justify='left')
        self.scan_placeholder.pack(fill='x', pady=6)

        # summary/analysis line
        self.scan_summary = tk.Label(
            wrap, text="", bg=BG, fg=self.accent_text, font=FONT_MONO_S,
            anchor='w', justify='left', wraplength=380)
        self.scan_summary.pack(fill='x', pady=(4, 0))

    # ── scan-results helpers (called from the mission thread via .after) ──

    _COLOUR_SWATCH = {
        "red": "#e53935", "orange": "#fb8c00", "yellow": "#fdd835",
        "green": "#43a047", "cyan": "#00acc1", "blue": "#1e88e5",
        "purple": "#8e24aa", "white": "#cfd8dc", "black": "#37474f",
        "grey": "#90a4ae",
    }

    def clear_scan_results(self):
        self._scan_results = []
        for w in self.scan_rows_frame.winfo_children():
            w.destroy()
        self.scan_placeholder = tk.Label(
            self.scan_rows_frame, text="Scanning in progress…",
            bg=BG, fg=C_DIM, font=FONT_MONO_S, anchor='w')
        self.scan_placeholder.pack(fill='x', pady=6)
        self.scan_summary.configure(text="")

    def _add_scan_row(self, res):
        """Render one result row in the SCAN RESULTS table."""
        if self.scan_placeholder is not None:
            self.scan_placeholder.destroy()
            self.scan_placeholder = None
        row = tk.Frame(self.scan_rows_frame, bg=BG)
        row.pack(fill='x', pady=1)
        found = res.get("found")
        cells = [
            (res.get("waypoint", "-"), 16, C_WHITE),
            (res.get("scan_dir", "-"), 6, C_MUTED),
            ("YES" if found else "no", 6, C_GREEN if found else C_RED),
            (res.get("label") or "-", 12, C_WHITE),
            (f"{res.get('confidence', 0):.0%}" if found else "-", 6, C_MUTED),
        ]
        for text, w, fg in cells:
            tk.Label(row, text=str(text), bg=BG, fg=fg, font=FONT_MONO_S,
                     width=w, anchor='w').pack(side='left', padx=2)
        # colour cell with a little swatch
        colour = res.get("colour")
        cframe = tk.Frame(row, bg=BG)
        cframe.pack(side='left', padx=2)
        if found and colour:
            sw = self._COLOUR_SWATCH.get(colour, "#bbbbbb")
            tk.Frame(cframe, bg=sw, width=10, height=10,
                     highlightbackground=BORDER,
                     highlightthickness=1).pack(side='left', padx=(0, 3))
        tk.Label(cframe, text=colour or "-", bg=BG,
                 fg=C_WHITE if found else C_DIM,
                 font=FONT_MONO_S, anchor='w').pack(side='left')

    def scan_at_waypoint(self, waypoint, scan_dir):
        """Grab the current live frame, run YOLO, return a result dict.

        Called from the mission thread. Reuses the panel's loaded detector and
        the most recent camera frame (turn CAM ON before flying, or run the
        Smart Square Scan with the laptop webcam for a dry run).
        """
        self.after(0, lambda: self.log(
            f"Scanning for box at {waypoint} (rotate {scan_dir})…", 'info'))

        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        res = {"waypoint": waypoint, "scan_dir": scan_dir, "found": False,
               "label": None, "confidence": 0.0, "colour": None,
               "note": "no camera frame — is CAM ON?"}

        if frame is not None:
            res = self._detect_box(frame)
            res["waypoint"] = waypoint
            res["scan_dir"] = scan_dir

        self._scan_results.append(res)
        self.after(0, lambda r=res: [
            self._add_scan_row(r),
            self.log(
                f"{waypoint}: " + ("FOUND " + str(r['label']) +
                f" ({r['confidence']:.0%}, {r['colour']})" if r["found"]
                else "no box detected"),
                'ok' if r["found"] else 'warn'),
        ])
        return res

    def _detect_box(self, frame):
        """Run the loaded YOLO model on a frame; return a result dict with the
        best matching detection plus its dominant colour. Never raises."""
        res = {"found": False, "label": None, "confidence": 0.0,
               "colour": None, "note": "detector not using YOLO"}
        det = self.detector
        if not (det.using_yolo and det.model is not None):
            res["note"] = "YOLO not loaded — press TEST CAMERA + LOAD YOLO first"
            return res
        try:
            out = det.model.predict(frame, conf=SMART_SQUARE_CONF, verbose=False)
            r = out[0]
            best = None
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for b in boxes:
                    conf = float(b.conf[0])
                    label = names.get(int(b.cls[0]), str(int(b.cls[0])))
                    if conf < SMART_SQUARE_CONF:
                        continue
                    if SMART_SQUARE_CLASSES and label not in SMART_SQUARE_CLASSES:
                        continue
                    if best is None or conf > best["confidence"]:
                        best = {"label": label, "confidence": round(conf, 3),
                                "xyxy": b.xyxy[0].tolist()}
            if best is None:
                res["note"] = "no matching box detected"
                return res
            colour = self._dominant_colour(frame, best["xyxy"])
            return {"found": True, "label": best["label"],
                    "confidence": best["confidence"], "colour": colour,
                    "note": "detected"}
        except Exception as e:
            res["note"] = f"scan error: {e}"
            return res

    @staticmethod
    def _dominant_colour(frame, box):
        """Estimate the dominant colour name inside a bounding box (HSV)."""
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return "unknown"
            crop = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hh, ss, vv = hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean()
            if ss < 40 and vv > 180: return "white"
            if vv < 60:              return "black"
            if ss < 40:              return "grey"
            hue = hh * 2
            for limit, name in [(15, "red"), (45, "orange"), (70, "yellow"),
                                (170, "green"), (200, "cyan"), (260, "blue"),
                                (320, "purple"), (360, "red")]:
                if hue < limit:
                    return name
            return "unknown"
        except Exception:
            return "unknown"

    def show_scan_summary(self, results, aborted=False):
        """Compute and display the post-flight SUMMARY + ANALYSIS."""
        if not results:
            self.scan_summary.configure(
                text="No scans were performed." if not aborted
                else "Mission aborted before any scan.")
            return
        found = [r for r in results if r.get("found")]
        colours = [r["colour"] for r in found if r.get("colour")]
        parts = []
        parts.append(("ABORTED — " if aborted else "") +
                     f"{len(found)}/{len(results)} scans found a box.")
        if found:
            best = max(found, key=lambda r: r["confidence"])
            avg = sum(r["confidence"] for r in found) / len(found)
            parts.append(f"Best: {best['label']} {best['confidence']:.0%} "
                         f"@ {best['waypoint']}.  Avg conf {avg:.0%}.")
        if colours:
            from collections import Counter
            uniq = ", ".join(sorted(set(colours)))
            top, n = Counter(colours).most_common(1)[0]
            parts.append(f"Colours: {uniq}.  Most common: {top} ({n}x).")
        rate = len(found) / len(results)
        verdict = ("strong" if rate >= 0.75 else
                   "partial" if rate >= 0.4 else "weak")
        parts.append(f"Verdict: {verdict} success ({rate:.0%} hit rate).")
        self.scan_summary.configure(text="  ".join(parts))
        for line in parts:
            self.log("SUMMARY · " + line, 'ok' if found else 'warn')
        self._save_scan_summary_csv(results, parts, aborted)

    def _save_scan_summary_csv(self, results, summary_lines, aborted):
        """Write per-waypoint scan results + the summary/analysis to a CSV
        in logs/ (visible via the FILES button)."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            fname = (datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                     + '_scan_summary.csv')
            fpath = os.path.join(LOG_DIR, fname)
            with open(fpath, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['mission', 'Smart Square Scan (BOX detection)'])
                w.writerow(['datetime',
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                w.writerow(['aborted', 'YES' if aborted else 'NO'])
                w.writerow([])
                w.writerow(['waypoint', 'scan_dir', 'box_found', 'label',
                            'confidence', 'colour', 'note'])
                for r in results:
                    w.writerow([r.get('waypoint', ''), r.get('scan_dir', ''),
                                'YES' if r.get('found') else 'NO',
                                r.get('label') or '',
                                f"{r.get('confidence', 0.0):.2f}",
                                r.get('colour') or '',
                                r.get('note') or ''])
                w.writerow([])
                w.writerow(['SUMMARY / ANALYSIS'])
                for line in summary_lines:
                    w.writerow([line])
            self.log(f"Summary saved: {fpath}", 'ok')
        except Exception as err:
            self.log(f"Could not save summary CSV: {err}", 'warn')

    # ── PATH COLUMN: embedded live flight path ────────────────

    def _build_path_col(self, parent):
        col = tk.Frame(parent, bg=BG, highlightbackground=BORDER,
                       highlightthickness=1)
        col.grid(row=0, column=2, sticky='nsew', padx=4, pady=8)
        self._section_hdr(col, "LIVE FLIGHT PATH · TOP VIEW (2x2 m)")
        # The embedded flight-path view (same drawing as the pop-out window).
        self.path_view = FlightPathView(col, self.tracker, self.accent_text,
                                        compact=True)
        self.path_view.pack(fill='both', expand=True, padx=6, pady=(4, 8))
        self.path_view.start()      # begins the ~10 fps live redraw loop

    # ── RIGHT COLUMN: status + controls + activity log ────────

    def _build_right_col(self, parent):
        col = tk.Frame(parent, bg=BG, highlightbackground=BORDER,
                       highlightthickness=1, width=340)
        col.grid(row=0, column=3, sticky='nsew', padx=(4, 8), pady=8)
        col.grid_propagate(False)
        col.pack_propagate(False)

        self._section_hdr(col, "FLIGHT CONTROL")

        status_frame = tk.Frame(col, bg=CARD_BG, highlightbackground=BORDER,
                                highlightthickness=1)
        status_frame.pack(fill='x', padx=10, pady=(0, 8))
        tk.Label(status_frame, text='STATUS', bg=CARD_BG, fg=C_MUTED,
                 font=FONT_LABEL).pack(anchor='w', padx=8, pady=(5, 0))
        self.status_main = tk.Label(status_frame, text="NOT CONNECTED",
                                    bg=CARD_BG, fg=C_MUTED,
                                    font=('Courier', 12, 'bold'), anchor='w',
                                    wraplength=320)
        self.status_main.pack(fill='x', padx=8, pady=(2, 0))
        self.status_sub = tk.Label(status_frame, text="Connect the drone to begin",
                                   bg=CARD_BG, fg=C_DIM, font=FONT_MONO_S,
                                   anchor='w', wraplength=320)
        self.status_sub.pack(fill='x', padx=8, pady=(0, 8))

        def hud_btn(text, fg, cmd, state='normal'):
            return tk.Button(col, text=text, bg=CARD_BG, fg=fg,
                             relief='flat', padx=0, pady=6,
                             font=('Courier', 12, 'bold'), cursor='hand2',
                             state=state, command=cmd,
                             highlightbackground=BORDER, highlightthickness=1)

        self.connect_btn = hud_btn("[ CONNECT ]", self.accent_text, self._toggle_connect)
        self.connect_btn.pack(fill='x', padx=10, pady=(0, 4))
        self.start_btn = hud_btn("> START MISSION", C_GREEN, self._do_start,
                                 state='disabled')
        self.start_btn.pack(fill='x', padx=10, pady=(0, 4))
        self.stop_btn = hud_btn("[] EMERGENCY STOP", C_RED, self._do_stop,
                                state='disabled')
        self.stop_btn.pack(fill='x', padx=10, pady=(0, 4))
        self.path_btn = hud_btn("◎ ENLARGE PATH", self.accent_text, self._open_path_window)
        self.path_btn.pack(fill='x', padx=10, pady=(0, 4))
        self.files_btn = hud_btn("FILES", C_BLUE, self._open_downloads)
        self.files_btn.pack(fill='x', padx=10, pady=(0, 10))

        tk.Label(col, text='ACTIVITY LOG', bg=BG, fg=C_MUTED,
                 font=FONT_LABEL).pack(anchor='w', padx=10)
        log_frame = tk.Frame(col, bg=BG)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(2, 8))
        sb = tk.Scrollbar(log_frame, troughcolor=CARD_BG, bg=BORDER)
        sb.pack(side='right', fill='y')
        self.log_box = tk.Text(log_frame, height=8, bg=LOG_BG, fg=C_MUTED,
                               font=('Courier', 10), relief='flat',
                               insertbackground=self.accent_text, state='disabled',
                               yscrollcommand=sb.set, highlightbackground=BORDER,
                               highlightthickness=1)
        self.log_box.pack(fill='both', expand=True)
        sb.config(command=self.log_box.yview)
        self.log_box.tag_configure('info',  foreground=C_MUTED)
        self.log_box.tag_configure('ok',    foreground=C_GREEN)
        self.log_box.tag_configure('warn',  foreground=C_AMBER)
        self.log_box.tag_configure('error', foreground=C_RED)

    # ─────────────────────────────────────────────────────────
    # LOGGING + STATUS
    # ─────────────────────────────────────────────────────────

    def log(self, message, level='info'):
        ts = time.strftime('%H:%M:%S')
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f"[{ts}] {message}\n", level)
        self.log_box.see('end')
        self.log_box.configure(state='disabled')
        self._activity_log.append((ts, message))

    def _set_status(self, main, sub='', color=None):
        self.status_main.configure(text=main, fg=color if color else C_WHITE)
        self.status_sub.configure(text=sub)

    def _on_cam_resize(self, event):
        """Track the camera canvas size so frames scale to fill it."""
        # Keep a 4:3 feel but follow the actual canvas width/height.
        self.cam_w = max(160, event.width)
        self.cam_h = max(120, event.height)
        if not self._cam_on:
            self._show_cam_placeholder()

    def _show_cam_placeholder(self):
        self.cam_canvas.delete('all')
        w = getattr(self, 'cam_w', 480)
        h = getattr(self, 'cam_h', 300)
        cx, cy = w / 2, h / 2
        self.cam_canvas.create_rectangle(0, 0, w, h, fill=CANVAS_BG,
                                         tags='placeholder')
        self.cam_canvas.create_text(cx, cy - 10, text="AWAITING STREAM",
                                    fill=C_MUTED, font=('Courier', 11, 'bold'),
                                    tags='placeholder')
        self.cam_canvas.create_text(cx, cy + 10,
                                    text="press TEST or CAM ON to start",
                                    fill=C_DIM, font=('Courier', 9),
                                    tags='placeholder')

    # ─────────────────────────────────────────────────────────
    # FEATURE 1 — YOLO MODEL SELECTION
    # ─────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────
    # CAMERA SOURCE SELECTION  (laptop webcam vs drone camera)
    # ─────────────────────────────────────────────────────────

    def _use_drone_camera(self):
        """True if the user picked the drone's camera as the YOLO source."""
        return self.sel_camera.get().lower().startswith("drone")

    def _on_camera_change(self, _event=None):
        src = self.sel_camera.get()
        self.log(f"Camera source set to: {src}", 'info')
        if self._use_drone_camera() and not self._connected:
            self.log("Drone camera selected — connect the Tello first.", 'warn')
        restart = self._cam_on or self._camera_restart
        self._camera_off()
        self._camera_restart = restart
        if restart and self._camera_session is None:
            self._camera_restart = False
            self._camera_on()

    # ─────────────────────────────────────────────────────────
    # FEATURE 2 — TEST CAMERA + LOAD YOLO (preview before flight)
    # ─────────────────────────────────────────────────────────

    def _test_camera_yolo(self):
        """Load the chosen model and start the SELECTED camera source so the
        user can preview live detections before flying. The source (laptop
        webcam or drone camera) is taken from the CAMERA SOURCE dropdown."""
        if not CAMERA_AVAILABLE:
            self.log("Install: pip install opencv-python pillow", 'error')
            self.model_status.configure(
                text="opencv/pillow missing — cannot preview.", fg=C_RED)
            return

        # Guard: drone camera chosen but not connected.
        if self._use_drone_camera() and not self._connected:
            self.model_status.configure(
                text="Drone camera selected — connect the Tello first, "
                     "or switch CAMERA SOURCE to Laptop Webcam.", fg=C_AMBER)
            self.log("Cannot test drone camera: Tello not connected.", 'warn')
            return

        model_name = self.sel_model.get()
        task       = self.sel_task.get()
        source     = self.sel_camera.get()
        self.test_btn.configure(state='disabled', text="LOADING MODEL...")
        self.model_status.configure(
            text=f"Loading {model_name} for {source} ...", fg=C_AMBER)
        self.log(f"Loading YOLO {model_name} ({task}) on {source}...", 'info')

        def worker():
            ok, msg = self.detector.load(model_name, task)
            self._model_loaded = ok
            level = 'ok' if ok else 'warn'
            self.after(0, lambda: [
                self.model_status.configure(
                    text=msg, fg=C_GREEN if ok else C_AMBER),
                self.log(msg, level),
                self.test_btn.configure(state='normal',
                                        text="3 · TEST CAMERA + LOAD YOLO"),
            ])
            # Start the preview on the chosen source.
            self.after(0, self._camera_on)

        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────
    # CAMERA
    # ─────────────────────────────────────────────────────────

    def _open_capture(self, session):
        """Worker only: arm the UDP decoder before enabling the drone stream."""
        if not session['drone_source']:
            # DirectShow avoids the Windows MSMF source-switching warning.
            return (cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system() == 'Windows'
                    else cv2.VideoCapture(0))
        if not self._video_port_free():
            raise RuntimeError(f"UDP {VIDEO_PORT} is occupied; close the other video receiver")
        cancelled = threading.Event()
        outcome = []
        def enable_stream():
            # VideoCapture.open waits for packets. Observe its bound UDP socket
            # from this separate worker instead of sending before it listens.
            deadline = time.monotonic() + 5.0
            try:
                while self._camera_alive(session) and not cancelled.is_set():
                    if self._video_receiver_bound():
                        if not cancelled.is_set() and self._camera_alive(session):
                            reply = session['drone'].send('streamon', delay=0)
                            if reply.lower() != 'ok':
                                raise RuntimeError(f"streamon failed: {reply or 'no response'}; reconnect")
                            session['stream_started'] = True
                            outcome.append(None)
                        return
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Video receiver did not bind within 5 seconds')
                    cancelled.wait(0.02)
            except Exception as err:
                outcome.append(str(err))
        starter = threading.Thread(target=enable_stream, daemon=True)
        starter.start()
        cap = None
        try:
            cap = cv2.VideoCapture(
                f'udp://@0.0.0.0:{VIDEO_PORT}', cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
        except Exception:
            cancelled.set()
            starter.join(6.0)
            raise
        cancelled.set()
        starter.join(6.0)
        if starter.is_alive() or not outcome or outcome[0] is not None:
            if cap is not None:
                cap.release()
            raise RuntimeError(outcome[0] if outcome and outcome[0] else
                               'Video start cancelled or receiver unavailable')
        return cap

    @staticmethod
    def _video_port_free():
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('0.0.0.0', VIDEO_PORT))
            return True
        except OSError as err:
            if err.errno == errno.EADDRINUSE or getattr(err, 'winerror', None) in (10048, 10013):
                return False
            raise
        finally:
            probe.close()

    @staticmethod
    def _video_receiver_bound():
        # Observe the UDP table without binding a probe while FFmpeg opens;
        # a competing probe socket can itself make the decoder bind fail.
        if platform.system() == 'Linux':
            with open('/proc/net/udp', encoding='ascii') as table:
                for row in list(table)[1:]:
                    if int(row.split()[1].rsplit(':', 1)[1], 16) == VIDEO_PORT:
                        return True
            return False
        result = subprocess.run(['netstat', '-an', '-p', 'udp'],
                                capture_output=True, text=True, timeout=2,
                                check=True, creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                                          if platform.system() == 'Windows' else 0))
        for row in result.stdout.splitlines():
            fields = row.split()
            if fields and fields[0].lower().startswith('udp'):
                # Windows local address is column 2; macOS is column 4.
                address = fields[1] if platform.system() == 'Windows' else fields[3]
                if address.rsplit(':' if ':' in address else '.', 1)[-1] == str(VIDEO_PORT):
                    return True
        return False

    def _camera_alive(self, session):
        if self._closing or session['stop'].is_set():
            return False
        if session['drone_source']:
            connection = session['connection']
            return (connection is self._connection and self._connected
                    and not connection['stop'].is_set())
        return True

    def _camera_message(self, session, kind, text):
        with self._cam_lock:
            session['messages'].append((kind, text))

    def _camera_on(self):
        if not CAMERA_AVAILABLE or self._closing or self._cam_on:
            return
        if self._camera_session is not None:
            self._camera_restart = True
            return
        drone_source = self._use_drone_camera()
        if drone_source and not (self._connected and self._drone):
            self.log('Drone camera selected but Tello is not connected. Connect first.', 'warn')
            self.obstacle_lbl.configure(text='[ CAMERA SOURCE UNAVAILABLE ]', fg=C_AMBER)
            return
        if drone_source and (self._drone.abort or self._drone.uncertain):
            self.log('Camera blocked: command state requires reconnect.', 'error')
            return
        session = dict(drone_source=drone_source, drone=self._drone,
                       connection=self._connection, stop=threading.Event(),
                       done=threading.Event(), messages=[], stream_started=False)
        self._camera_session = session
        self._camera_restart = False
        self._cam_on = True
        with self._frame_lock:
            self._latest_frame = None
        self._show_cam_placeholder()
        self.obstacle_lbl.configure(text='[ CAMERA STARTING ]', fg=C_AMBER)
        self.cam_on_btn.configure(state='disabled')
        self.cam_off_btn.configure(state='normal')
        threading.Thread(target=self._camera_loop, args=(session,), daemon=True).start()
        self.after(40, lambda: self._camera_poll(session))

    def _camera_poll(self, session):
        """GUI only: consume worker messages without waiting on video I/O."""
        if self._closing or session is not self._camera_session:
            return
        with self._cam_lock:
            messages, session['messages'] = session['messages'], []
        for kind, message in messages:
            if session['stop'].is_set():
                continue
            self.log(message, 'error' if kind == 'error' else 'warn' if kind == 'lost' else 'ok')
            if kind in ('lost', 'error'):
                self._show_cam_placeholder()
                self.obstacle_lbl.configure(text='[ STREAM LOST ]', fg=C_AMBER)
            elif kind == 'started':
                self.obstacle_lbl.configure(text='[ PREVIEW RUNNING ]', fg=C_AMBER)
                for button in (self.snap_btn, self.record_btn):
                    button.configure(state='normal')
        if session['done'].is_set():
            self._cam_on = False
            self._camera_session = None
            with self._frame_lock:
                self._latest_frame = None
            if self._recording:
                self._stop_recording()
            self._show_cam_placeholder()
            self.cam_on_btn.configure(state='normal')
            for button in (self.cam_off_btn, self.snap_btn, self.record_btn):
                button.configure(state='disabled')
            self.obstacle_lbl.configure(text='[ CAMERA INACTIVE — CHECK LOG ]', fg=C_AMBER)
            if self._camera_restart:
                self._camera_restart = False
                self._camera_on()
            return
        if self._cam_on and not session['stop'].is_set():
            self._update_camera_frame()
        self.after(40, lambda: self._camera_poll(session))

    def _camera_off(self):
        self._camera_restart = False
        session = self._camera_session
        if session is not None:
            session['stop'].set()
        was_on = self._cam_on
        self._cam_on = False
        if self._recording:
            self._stop_recording()
        with self._frame_lock:
            self._latest_frame = None
        self._show_cam_placeholder()
        self.obstacle_lbl.configure(text='[ CAMERA STOPPING ]' if session else
                                    '[ CAMERA INACTIVE ]', fg=C_MUTED)
        self.cam_on_btn.configure(state='disabled' if session else 'normal')
        for button in (self.cam_off_btn, self.snap_btn, self.record_btn):
            button.configure(state='disabled')
        if was_on:
            self.log('Camera stopping; decoder cleanup runs in the background.', 'warn')

    def _camera_loop(self, session):
        """Own capture/release in one worker; retry a healthy stream only once."""
        cap = None
        try:
            for attempt in range(2):
                if not self._camera_alive(session):
                    return
                if session['drone_source'] and not attempt:
                    reply = session['drone'].send('streamoff', delay=0)
                    if reply.lower() != 'ok':
                        raise RuntimeError(f"Initial streamoff failed: {reply or 'no response'}; reconnect")
                if attempt:
                    drone = session['drone']
                    if (not session['drone_source'] or drone.abort or drone.uncertain
                            or not self._telem_received
                            or time.monotonic() - self._telem_received > MAX_TELEMETRY_AGE):
                        raise RuntimeError('Video recovery stopped: connection is not healthy; reconnect')
                    reply = drone.send('streamoff', delay=0)
                    if reply.lower() != 'ok':
                        raise RuntimeError(f"streamoff failed: {reply or 'no response'}; reconnect")
                    session['stream_started'] = False
                cap = self._open_capture(session)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    if attempt == 0 and session['drone_source']:
                        self._camera_message(session, 'lost', 'No decoded video; one bounded restart will be attempted.')
                        continue
                    raise RuntimeError('Could not open the selected camera')
                with self._cam_lock:
                    self._cap = cap
                last_frame = time.monotonic()
                announced = False
                _diag_camera_tick = time.monotonic()
                _diag_frame_count = 0
                while self._camera_alive(session):
                    ret, frame = cap.read()
                    if not self._camera_alive(session):
                        break
                    if not ret or frame is None:
                        if time.monotonic() - last_frame >= 3.0:
                            with self._frame_lock:
                                self._latest_frame = None
                            self._camera_message(session, 'lost', 'Video stream lost; clearing stale frame.')
                            break
                        session['stop'].wait(0.03)
                        continue
                    last_frame = time.monotonic()
                    _diag_frame_count += 1
                    if last_frame - _diag_camera_tick >= 1.0:
                        _diag_emit('camera_frame_rate',
                                   source='Tello' if session['drone_source'] else 'Laptop',
                                   frames=_diag_frame_count,
                                   fps=round(_diag_frame_count / max(0.001,last_frame - _diag_camera_tick), 2),
                                   frame_w=frame.shape[1], frame_h=frame.shape[0])
                        _diag_camera_tick = last_frame
                        _diag_frame_count = 0
                    if not announced:
                        source = 'Drone camera (Tello stream)' if session['drone_source'] else 'Laptop webcam'
                        self._camera_message(session, 'started', f'Camera started — {source}; decoded frame received.')
                        announced = True
                    with self._frame_lock:
                        if self._camera_alive(session):
                            self._latest_frame = frame.copy()
                    if self._recording and self._video_writer:
                        with self._cam_lock:
                            if self._video_writer:
                                self._video_writer.write(frame)
                    session['stop'].wait(0.01)
                cap.release()
                cap = None
                with self._cam_lock:
                    self._cap = None
                if not self._camera_alive(session):
                    return
                if not session['drone_source'] or attempt == 1:
                    raise RuntimeError('Camera stopped receiving frames; check connection and restart manually')
        except Exception as err:
            self._camera_message(session, 'error', str(err))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception as err:
                    self._camera_message(session, 'error', f'Camera release failed: {err}')
            with self._cam_lock:
                self._cap = None
            drone = session['drone']
            # Never send a cleanup command on a replacement connection or
            # reset uncertainty to make video recovery appear successful.
            if (session['stream_started'] and session['drone_source']
                    and not self._closing and session['connection'] is self._connection
                    and self._connected and not drone.abort and not drone.uncertain):
                try:
                    reply = drone.send('streamoff', delay=0)
                    if reply.lower() != 'ok':
                        self._camera_message(session, 'error',
                            f"Camera cleanup streamoff failed: {reply or 'no response'}; reconnect")
                except Exception as err:
                    self._camera_message(session, 'error', f'Camera cleanup failed: {err}')
            session['done'].set()

    def _schedule_frame_update(self):
        if self._cam_on and not self._stop_flag:
            self._update_camera_frame()
            self.after(40, self._schedule_frame_update)   # ~25 fps display

    def _update_camera_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        annotated, n = self.detector.annotate(frame)
        error = self.detector.last_error
        if error:
            if self._model_loaded:
                self.log(error, 'error')
            self._model_loaded = False
            self.model_status.configure(text=error + " — reload YOLO", fg=C_RED)
            self.obstacle_lbl.configure(text="[ YOLO INFERENCE FAILED — reload model ]", fg=C_RED)
        elif self.detector.using_yolo:
            self.obstacle_lbl.configure(
                text=f"[ BOX DETECTION · {n} box(es) found ]",
                fg=C_GREEN if n else C_MUTED)
        else:
            self.obstacle_lbl.configure(
                text="[ YOLO NOT LOADED — press TEST CAMERA + LOAD YOLO ]",
                fg=C_AMBER)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        # Scale to the current canvas size (keeps aspect ratio, letterboxed).
        cw = max(160, int(self.cam_canvas.winfo_width()))
        ch = max(120, int(self.cam_canvas.winfo_height()))
        fh, fw = rgb.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(rgb, (nw, nh))
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.cam_canvas.delete('frame')
        self.cam_canvas.create_image(cw // 2, ch // 2, anchor='center',
                                     image=photo, tags='frame')
        self.cam_canvas._photo = photo

    # ─────────────────────────────────────────────────────────
    # PHOTO + VIDEO
    # ─────────────────────────────────────────────────────────

    def _snap_photo(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            self.log("No frame — is the camera on?", 'warn')
            return
        os.makedirs(PHOTO_DIR, exist_ok=True)
        fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
        cv2.imwrite(os.path.join(PHOTO_DIR, fname), frame)
        self.log(f"Photo saved: {fname}", 'ok')
        # Drop a marker on the live flight path at the current position.
        try:
            self.tracker.mark_event("PHOTO")
        except Exception:
            pass

    def _toggle_record(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            self.log("No frame — is the camera on?", 'warn')
            return
        os.makedirs(VIDEO_DIR, exist_ok=True)
        fname  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.avi'
        h, w   = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        with self._cam_lock:
            self._video_writer = cv2.VideoWriter(os.path.join(VIDEO_DIR, fname),
                                                 fourcc, 25.0, (w, h))
            self._recording = True
        self.record_btn.configure(text='STOP REC', fg=C_RED)
        self.log(f"Recording: {fname}", 'ok')

    def _stop_recording(self):
        with self._cam_lock:
            self._recording = False
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None
        self.record_btn.configure(text='REC', fg='#cc44cc')
        self.log("Recording stopped.", 'warn')

    # ─────────────────────────────────────────────────────────
    # CONNECT / DISCONNECT
    # ─────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if self._connected:
            self._do_disconnect()
        else:
            self._do_connect()

    def _do_connect(self):
        # A previous connection must finish cleanup before a new bind is allowed.
        if self._connection is not None or self._closing:
            return
        cmd_sock = state_sock = None
        try:
            cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Windows: fail visibly if another controller owns the command port.
            # Other platforms use the default non-reusable bind.
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            cmd_sock.bind(('', 9000))
            cmd_sock.settimeout(5)
            state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            state_sock.bind(('', STATE_PORT))
            state_sock.settimeout(1)
        except OSError as err:
            for sock in (cmd_sock, state_sock):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self.log(f"Socket error: {err}", 'error')
            self._set_status("CONNECT FAILED", "Check WiFi and close other controllers", C_RED)
            self.connect_btn.configure(text="[ CONNECT ]", state='normal')
            return

        drone = DroneProxy(cmd_sock, tracker=self.tracker)
        session = dict(command=cmd_sock, state=state_sock, drone=drone,
                       stop=threading.Event(), ready=threading.Event(),
                       threads=[], cleaning=False)
        with self._connection_lock:
            self._connection = session
            self._cmd_sock, self._state_sock, self._drone = cmd_sock, state_sock, drone
            self._telem = {}
            self._telem_received = 0.0
            self._telem_ready = False
            self._pkt_count = 0
        self._landing_requested = False
        self._landing_in_progress = False
        self._landing_unconfirmed = False
        self._stale_telem_landing = False
        self._connected = False
        self._stop_flag = False
        self._tick_on = False
        drone.on_refused = lambda c, r: self._connection_after(
            session, lambda: self.log(
                f"{('COMMAND TIMEOUT' if not r else 'COMMAND BLOCKED' if r == 'blocked' else 'COMMAND FAILED')} '{c}' — reply: '{r or 'no response'}'"
                + (" (battery too low?)" if c == 'takeoff' else ""), 'error'))
        self.connect_btn.configure(text="[ CONNECTING... ]", state='disabled')
        self.start_btn.configure(state='disabled')
        self._set_status("CONNECTING", "Sending SDK command to drone", C_AMBER)
        try:
            session['threads'] = [
                threading.Thread(target=self._state_loop, args=(session,), daemon=True),
                threading.Thread(target=self._handshake, args=(session,), daemon=True),
            ]
            for thread in session['threads']:
                thread.start()
        except RuntimeError as err:
            self._begin_connection_cleanup(session, f"Connection worker failed: {err}", True)

    def _connection_after(self, session, callback, delay=0):
        """Run a queued UI update only for the live connection that created it."""
        if (session is None or self._closing or session is not self._connection
                or session['stop'].is_set()):
            return
        def current_only():
            if (not self._closing and session is self._connection
                    and not session['stop'].is_set()):
                callback()
        try:
            self.after(delay, current_only)
        except (tk.TclError, RuntimeError):
            pass  # The GUI may be closing while a worker finishes.

    def _begin_connection_cleanup(self, session, message, failed=False):
        """Stop this session and close its sockets before permitting reconnect."""
        if session is not self._connection or session['cleaning']:
            return
        with self._connection_lock:
            session['cleaning'] = True
            session['stop'].set()
            session['drone'].abort = True
            self._connected = False
            self._stop_flag = True
        self.connect_btn.configure(text="[ CLEANING UP... ]", state='disabled')
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='disabled')
        if self._cam_on:
            self._camera_off()
        for sock in (session['command'], session['state']):
            try:
                sock.close()
            except OSError:
                pass
        self._finish_connection_cleanup(session, message, failed)

    def _finish_connection_cleanup(self, session, message, failed):
        if self._closing or session is not self._connection:
            return
        # Poll without blocking Tk. Closed-socket receives have a finite timeout.
        if any(thread.is_alive() for thread in session['threads']):
            self.after(25, lambda: self._finish_connection_cleanup(session, message, failed))
            return
        with self._connection_lock:
            self._connection = None
            self._cmd_sock = self._state_sock = self._drone = None
            self._telem = {}
            self._telem_received = 0.0
            self._telem_ready = False
            self._pkt_count = 0
        self._tick_on = False
        self._landing_requested = False
        self._landing_in_progress = False
        self._landing_unconfirmed = False
        self.conn_dot.configure(fg=C_MUTED)
        self.conn_lbl.configure(text='OFFLINE', fg=C_MUTED)
        self.connect_btn.configure(text="[ CONNECT ]", state='normal')
        self.rtf_banner.configure(text="[ NOT CONNECTED ]", bg=CARD_BG, fg=C_MUTED)
        self._set_status("CONNECT FAILED" if failed else "DISCONNECTED", message,
                         C_RED if failed else C_MUTED)
        self.log(message, 'error' if failed else 'warn')
        # _stop_flag stays True until the next connection creates fresh workers.
        self._save_activity_log()

    def _handshake(self, session):
        drone, stop = session['drone'], session['stop']
        def fail(message):
            self._connection_after(session, lambda: self._begin_connection_cleanup(
                session, message, True))
        try:
            resp = drone.send("command", delay=1)
            if stop.is_set():
                return
            if resp.lower() != 'ok':
                fail(f"Handshake failed: '{resp or 'no response'}'. Check Tello WiFi.")
                return
            def connected():
                self._connected = True
                self.conn_dot.configure(fg=C_GREEN)
                self.conn_lbl.configure(text='ONLINE', fg=C_GREEN)
                self.connect_btn.configure(text="[ DISCONNECT ]", state='normal')
                self.stop_btn.configure(state='normal')
                self.log("SDK connected.", 'ok')
                self._set_status("CONNECTED", "Waiting for telemetry...", C_GREEN)
                self._start_tick()
            bat_str = drone.send("battery?", delay=0.5)
            if stop.is_set():
                return
            if drone.uncertain or drone.abort:
                fail("Battery query failed; reconnect before flight.")
                return
            self._connection_after(session, lambda: self.log(f"Battery: {bat_str}%", 'ok'))
            deadline = time.monotonic() + 6.0
            while not session['ready'].is_set() and time.monotonic() < deadline:
                if stop.wait(0.1):
                    return
            if stop.is_set():
                return
            if not session['ready'].is_set():
                fail("Validated telemetry was not received within 6 seconds.")
                return
            self._connection_after(session, lambda: (connected(), self._show_ready_state()))
        except Exception as err:
            fail(f"Connection failed: {err}")

    def _preflight_problem(self):
        """Return a single reason that flight must not start, or None if ready."""
        if not self._connected or not self._drone:
            return "drone is not connected"
        if self._landing_in_progress:
            return "landing is in progress"
        if self._landing_unconfirmed:
            return "landing is unconfirmed; verify grounded state and reconnect"
        if self._mission_thread and self._mission_thread.is_alive():
            return "the previous mission is still stopping"
        if not self._telem_ready or not self._telem:
            return "validated telemetry is not ready"
        if (not self._telem_received
                or time.monotonic() - self._telem_received > MAX_TELEMETRY_AGE):
            return "telemetry is missing or stale"
        if getattr(self._drone, 'uncertain', False):
            return "command state is uncertain; disconnect and reconnect"
        if self._drone.last_problem:
            return f"previous error: {self._drone.last_problem}; disconnect and reconnect"
        if self._drone.flight_unknown:
            return "flight state is unknown; verify grounded state and reconnect"
        if self._drone.flying:
            return "drone is already airborne"
        try:
            bat = int(self._telem['bat'])
            height = int(self._telem['h'])
            templ = int(self._telem['templ'])
            temph = int(self._telem['temph'])
        except (KeyError, TypeError, ValueError):
            return "telemetry values are incomplete or invalid"
        if bat < MIN_BAT_FLY:
            return f"battery {bat}% < {MIN_BAT_FLY}%"
        if temph > MAX_TEMP_FLY:
            return f"internal temperature {temph}C > {MAX_TEMP_FLY}C start limit"
        if templ < 0:
            return "reported internal temperature is invalid"
        if height > 10:
            return f"reported height {height} cm indicates the drone is not grounded"
        return None

    def _show_ready_state(self):
        """Enable START only when the complete pre-flight gate passes."""
        problem = self._preflight_problem()
        if problem:
            self.log(f"NOT READY: {problem}.", 'error')
            self._set_status("NOT READY", problem, C_RED)
            self.start_btn.configure(state='disabled')
            self._connection_after(self._connection, self._recheck_battery_gate, 5000)
        else:
            self.log("Telemetry live — pre-flight checks passed.", 'ok')
            self.start_btn.configure(state='normal')
            self._set_status("READY TO FLY", "Pick a mission, then START", C_GREEN)

    def _recheck_battery_gate(self):
        """Periodic re-check of the unified pre-flight gate."""
        if self._stop_flag or not self._connected:
            return
        if self._drone and self._drone.flying:
            return
        problem = self._preflight_problem()
        if problem is None:
            self.log("Pre-flight conditions satisfied — flight enabled.", 'ok')
            self.start_btn.configure(state='normal')
            self._set_status("READY TO FLY", "Pick a mission, then START", C_GREEN)
        else:
            self.start_btn.configure(state='disabled')
            self._set_status("NOT READY", problem, C_RED)
            self._connection_after(self._connection, self._recheck_battery_gate, 5000)

    def _do_disconnect(self):
        if self._connection is not None:
            self._begin_connection_cleanup(self._connection, "Disconnected. Press CONNECT to reconnect.")

    # ─────────────────────────────────────────────────────────
    # FEATURE 3 + 4 — CHOOSE MISSION + START / STOP
    # ─────────────────────────────────────────────────────────

    def _resolve_mission(self):
        """Turn the dropdown choice into a callable mission function."""
        label = self.sel_mission.get()
        fn, needs_theme = MISSION_LIBRARY[label]
        if needs_theme == "smart_square":
            fn = make_smart_square_mission(self)
        elif needs_theme:
            fn = make_waypoint_mission(self.theme_id)
        return label, fn

    def _do_start(self):
        problem = self._preflight_problem()
        if problem:
            self.log(f"BLOCKED: {problem}.", 'error')
            self._set_status("NOT READY", problem, C_RED)
            self.start_btn.configure(state='disabled')
            return
        label, mission_fn = self._resolve_mission()
        self._drone.abort = False
        self._drone.last_problem = ""
        self._landing_requested = False
        self._landing_unconfirmed = False
        # Reset the flight path trail for this run. The embedded path panel is
        # always live, so no need to open a window (ENLARGE PATH is optional).
        self.tracker.reset()
        self.start_btn.configure(state='disabled')
        self.mission_combo.configure(state='disabled')
        self._set_status("MISSION RUNNING", label, self.accent_text)
        self.log(f"Mission started: {label}", 'ok')
        self.log(f"Detector: {self.sel_model.get()} ({self.sel_task.get()})", 'info')

        drone = self._drone
        def run():
            try:
                mission_fn(drone)
            except Exception as err:
                drone.abort = True
                drone.last_problem = f"Mission error: {err}"
                self.after(0, lambda e=err: self.log(f"Mission error: {e}", 'error'))
            finally:
                self.after(0, lambda d=drone: self._on_mission_done(d))

        self._mission_thread = threading.Thread(target=run, daemon=True)
        self._mission_thread.start()

    # ─────────────────────────────────────────────────────────
    # LIVE FLIGHT-PATH WINDOW
    # ─────────────────────────────────────────────────────────

    def _open_path_window(self):
        """Open (or focus) the live flight-path window."""
        if self._path_window is not None:
            try:
                self._path_window.lift()
                self._path_window.focus_force()
                return
            except tk.TclError:
                self._path_window = None
        theme = dict(self.theme)
        theme["_id"] = self.theme_id
        self._path_window = FlightPathWindow(self, self.tracker, theme)

    def _on_path_window_closed(self):
        self._path_window = None

    def _restore_mission_combo(self):
        """Re-enable the mission picker, keeping it locked if the theme requires."""
        state = 'disabled' if self.theme.get("locked_mission") else 'readonly'
        self.mission_combo.configure(state=state)

    def _on_mission_done(self, drone=None):
        if drone is not None and drone is not self._drone:
            return  # Ignore completion from a disconnected session.
        if self._landing_in_progress:
            self._save_activity_log()
            return
        if self._landing_unconfirmed:
            self.start_btn.configure(state='disabled')
            self._restore_mission_combo()
            self._set_status("LANDING UNCONFIRMED", "Verify grounded state, then reconnect", C_RED)
            self._save_activity_log()
            return
        if self._drone and (self._drone.flying or self._drone.flight_unknown):
            self._request_landing("Mission ended without confirmed landing")
            self._save_activity_log()
            return
        if self._drone and self._drone.uncertain:
            self.start_btn.configure(state='disabled')
            self._restore_mission_combo()
            self._set_status("RECONNECT REQUIRED",
                             "Command outcome is uncertain; verify the drone is grounded", C_RED)
            self.log("Mission stopped with uncertain command state — verify the drone is grounded and reconnect.", 'error')
        elif self._drone and self._drone.abort:
            self._restore_mission_combo()
            self._show_ready_state()
            self.log("Mission aborted: " + (self._drone.last_problem or "STOP requested"), 'warn')
        else:
            self._restore_mission_combo()
            self._set_status("MISSION COMPLETE", "Landing acknowledged; check readiness", C_GREEN)
            self._sync_start_gate()
            self.log("Mission finished.", 'ok')
        self._save_activity_log()


    def _do_stop(self):
        self.rtf_banner.configure(text="[ !! EMERGENCY STOP !! ]",
                                  bg='#fdeaea', fg=C_RED)
        self._request_landing("STOP — controlled landing requested immediately", retry=True)

    def _request_landing(self, reason, retry=False):
        """One landing exchange shared by STOP, watchdog and mission recovery."""
        if self._landing_in_progress or (self._landing_requested and not retry):
            return
        self._landing_requested = True
        self._landing_in_progress = True
        drone = self._drone
        if drone:
            drone.abort = True  # Cancel a pending action before starting the worker.
        self.log(reason, 'error')
        self._set_status("STOPPING", "Landing command requested...", C_RED)
        self.start_btn.configure(state='disabled')
        threading.Thread(target=self._stop_thread, args=(drone,), daemon=True).start()

    def _stop_thread(self, drone=None):
        confirmed = False
        try:
            if drone:
                confirmed = drone.emergency_land()
        except Exception as err:
            if drone:
                drone.uncertain = True
                drone.last_problem = f"Landing error: {err}"
        self.after(0, lambda d=drone, ok=confirmed: self._finish_landing(d, ok))

    def _finish_landing(self, drone, confirmed):
        if drone is not self._drone:
            return  # A previous session must not change the new session's UI.
        self._landing_in_progress = False
        self._landing_unconfirmed = not confirmed
        self._restore_mission_combo()
        if confirmed:
            self.log("Landing acknowledged by the Tello; visually confirm grounded.", 'warn')
            self._set_status("STOPPED", "Landing acknowledged; mission aborted.", C_AMBER)
            self._sync_start_gate()
        else:
            self.log("Landing was requested but NOT confirmed. Verify the drone is grounded, then disconnect and reconnect.", 'error')
            self._set_status("LANDING UNCONFIRMED", "Verify grounded state, then reconnect", C_RED)
            self.start_btn.configure(state='disabled')
        self._save_activity_log()

    def _sync_start_gate(self):
        """Keep the main status and START consistent with live readiness."""
        problem = self._preflight_problem()
        self.start_btn.configure(state='normal' if problem is None else 'disabled')
        if (self._connected and self._drone and not self._drone.flying
                and not self._drone.flight_unknown and not self._landing_in_progress
                and not self._landing_unconfirmed
                and not (self._mission_thread and self._mission_thread.is_alive())):
            self._set_status('READY TO FLY' if problem is None else 'NOT READY',
                             'Pick a mission, then START' if problem is None else problem,
                             C_GREEN if problem is None else C_RED)



    # ─────────────────────────────────────────────────────────
    # TELEMETRY STATE LOOP + UI REFRESH
    # ─────────────────────────────────────────────────────────

    def _state_loop(self, session):
        # Local references can never switch to the next connection's socket.
        state_sock, stop = session['state'], session['stop']
        while not stop.is_set():
            try:
                data, address = state_sock.recvfrom(1024)
                if stop.is_set() or address[0] != TELLO_IP:
                    continue
                record = {}
                for pair in data.decode('ascii').strip().split(';'):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        record[k.strip()] = v.strip()
                try:
                    values = {key: int(record[key])
                              for key in ('bat', 'h', 'templ', 'temph')}
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0 <= values['bat'] <= 100
                        and 0 <= values['h'] <= 1000
                        and -20 <= values['templ'] <= values['temph'] <= 120):
                    continue
                with self._connection_lock:
                    if session is not self._connection or stop.is_set():
                        return
                    self._telem = record
                    self._telem_received = time.monotonic()
                    self._stale_telem_landing = False
                    session['drone'].telem = record
                    self._pkt_count += 1
                    if not self._telem_ready and self._pkt_count >= TELEM_READY:
                        self._telem_ready = True
                        session['ready'].set()
            except (socket.timeout, UnicodeDecodeError):
                pass
            except OSError:
                if not stop.is_set():
                    def state_error():
                        if self._connected:
                            # Preserve the command channel for the landing watchdog.
                            self._telem_received = 0.0
                            self.log("Telemetry receive failed; data is now stale.", 'error')
                            self._refresh_telem()
                        else:
                            self._begin_connection_cleanup(
                                session, "Telemetry socket closed unexpectedly; reconnect.", True)
                    self._connection_after(session, state_error)
                break

    def _start_tick(self):
        """Start one telemetry refresh loop for this connection only."""
        session = self._connection
        if session is None or session['stop'].is_set() or self._tick_on:
            return
        self._tick_on = True
        self._tick()

    def _tick(self):
        """Refresh telemetry display; an old session cannot restart this loop."""
        session = self._connection
        if session is None or session['stop'].is_set() or not self._connected:
            self._tick_on = False
            return
        self._refresh_telem()
        self._connection_after(session, self._tick, 500)

    def _refresh_telem(self):
        self._sync_start_gate()
        t = self._telem
        if not t:
            self._update_rtf_banner(None, None, None)
            return
        telem_age = (time.monotonic() - self._telem_received
                     if self._telem_received else float('inf'))
        if telem_age > MAX_TELEMETRY_AGE:
            if self._drone and (self._drone.flying or self._drone.flight_unknown):
                self._request_landing(f"AUTO LAND: telemetry stale for {telem_age:.1f} s")
            self._update_rtf_banner(None, None, None)
            return
        try:
            bat = int(t.get('bat', 0))
        except ValueError:
            bat = 0
        self.gauge.set(bat)
        self.height_card.configure(text=f"{t.get('h', '--')} cm")
        # Sync the flight-path altitude (Z) to the drone's live reported height.
        if 'h' in t:
            self.tracker.set_live_altitude(t.get('h'))

        tl, th = t.get('templ'), t.get('temph')
        if tl and th:
            self.temp_card.configure(text=f"{tl}-{th} C")
        elif tl:
            self.temp_card.configure(text=f"{tl} C")

        now = time.time()
        self._bat_history.append((now, bat))
        self._bat_history = [(ts, b) for ts, b in self._bat_history
                             if now - ts <= 60]
        measured = None
        if len(self._bat_history) >= 2:
            ots, ob = self._bat_history[0]
            nts, nb = self._bat_history[-1]
            em = (nts - ots) / 60
            if em >= 0.05:
                measured = (ob - nb) / em
                self.drain_card.configure(text=f"{measured:.1f}%/min",
                                          fg=C_RED if measured > 3 else C_MUTED)
            else:
                self.drain_card.configure(text="< 3s data")
        else:
            self.drain_card.configure(text="wait...")

        eta = bat / measured if (measured and measured > 0) \
              else bat * (MAX_FLIGHT_MIN / 100)
        self.eft_card.configure(text=f"{int(eta):02d}:{int((eta % 1) * 60):02d}",
                                fg=bat_color(bat))

        if self._drone and self._drone.flying:
            if self._flight_start is None:
                self._flight_start = now
            el = int(now - self._flight_start)
            self.timer_card.configure(text=f"{el // 60:02d}:{el % 60:02d}")
        else:
            self._flight_start = None

        self._update_rtf_banner(bat, tl, th)

        # auto-land safety
        if self._drone and (self._drone.flying or self._drone.flight_unknown):
            if bat <= CRITICAL_BAT:
                self._request_landing(f"AUTO LAND: battery {bat}%")
            try:
                if th and int(th) >= CRITICAL_TEMP:
                    self._request_landing(f"AUTO LAND: temp {th}C")
            except ValueError:
                pass

    def _update_rtf_banner(self, bat, templ, temph):
        if self._landing_unconfirmed or (self._drone and self._drone.flight_unknown):
            self.rtf_banner.configure(text="[ FLIGHT / LANDING UNCONFIRMED — RECONNECT ]",
                                      bg='#fdeaea', fg=C_RED)
            return
        if self._landing_in_progress:
            self.rtf_banner.configure(text="[ LANDING REQUESTED ]", bg='#fdeaea', fg=C_RED)
            return
        if self._drone and self._drone.flying:
            self.rtf_banner.configure(text="[ ^ IN FLIGHT ]", bg='#e8f4fb',
                                      fg=self.accent_text)
            return
        problem = self._preflight_problem()
        if problem:
            self.rtf_banner.configure(text=f"[ X NOT READY — {problem} ]",
                                      bg='#fdeaea', fg=C_RED)
        else:
            self.rtf_banner.configure(text="[ READY TO FLY ]", bg='#eafaf1',
                                      fg=C_GREEN)

    # ─────────────────────────────────────────────────────────
    # DOWNLOADS PANEL + LOG EXPORT + SHUTDOWN
    # ─────────────────────────────────────────────────────────

    def _open_downloads(self):
        win = tk.Toplevel(self)
        win.title("FILES — photos / videos / logs")
        win.configure(bg=BG)
        win.geometry("560x460")
        win.minsize(420, 320)
        win.resizable(True, True)
        tk.Label(win, text="[ SAVED FILES ]", bg=PANEL_BG, fg=self.accent_text,
                 font=('Courier', 11, 'bold')).pack(fill='x', pady=6)
        frame = tk.Frame(win, bg=BG)
        frame.pack(fill='both', expand=True, padx=12, pady=6)
        any_files = False
        for folder, label, ext in [(PHOTO_DIR, '[ PHOTOS ]', '.jpg'),
                                   (VIDEO_DIR, '[ VIDEOS ]', '.avi'),
                                   (LOG_DIR, '[ LOGS ]', '.csv')]:
            if not os.path.isdir(folder):
                continue
            files = sorted([f for f in os.listdir(folder) if f.endswith(ext)],
                           reverse=True)
            if not files:
                continue
            any_files = True
            tk.Label(frame, text=label, bg=BG, fg=self.accent_text,
                     font=('Courier', 10, 'bold')).pack(anchor='w', pady=(8, 2))
            for fname in files:
                fpath = os.path.abspath(os.path.join(folder, fname))
                row = tk.Frame(frame, bg=CARD_BG, highlightbackground=BORDER,
                               highlightthickness=1)
                row.pack(fill='x', pady=2)
                tk.Label(row, text=f'  {fname}', bg=CARD_BG, fg=C_WHITE,
                         font=FONT_MONO_S, anchor='w').pack(side='left',
                         fill='x', expand=True, padx=6, pady=5)
                tk.Button(row, text='OPEN', bg=PANEL_BG, fg=self.accent_text,
                          relief='flat', font=FONT_LABEL, cursor='hand2',
                          command=lambda p=fpath: self._open_file(p)).pack(
                              side='left', padx=6, pady=4)
        if not any_files:
            tk.Label(frame, text="NO FILES SAVED YET\n\nPhotos, videos and logs\n"
                     "appear here after a flight.", bg=BG, fg=C_MUTED,
                     font=FONT_MONO, justify='center').pack(pady=40)

    @staticmethod
    def _open_file(path):
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception:
            pass

    def _save_activity_log(self):
        if not self._activity_log:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        fname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_activity.csv'
        with open(os.path.join(LOG_DIR, fname), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time', 'event'])
            w.writerows(self._activity_log)
        self._activity_log = []

    def _on_close(self):
        self._closing = True
        with self._connection_lock:
            if self._connection is not None:
                self._connection['stop'].set()
        self._stop_flag = True
        if self._drone:
            self._drone.abort = True
        if self._recording:
            self._stop_recording()
        self._cam_on = False
        if self._camera_session is not None:
            self._camera_session['stop'].set()
        with self._frame_lock:
            self._latest_frame = None
        for sock in (self._cmd_sock, self._state_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._save_activity_log()
        self.destroy()


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — THEME PICKER + ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def pick_theme_dialog():
    """Small launcher window to choose a theme if none was given on the CLI."""
    chosen = {'id': None}
    root = tk.Tk()
    root.title("CHOOSE THEME")
    root.configure(bg=BG)
    tk.Label(root, text="DJI TELLO — CHOOSE PROJECT THEME", bg=BG,
             fg=C_WHITE, font=('Courier', 14, 'bold')).pack(padx=30, pady=(20, 10))
    for tid, th in THEMES.items():
        def choose(t=tid):
            chosen['id'] = t
            root.destroy()
        b = tk.Button(root, text=f"  {th['emoji']}  THEME {tid} — {th['name']}",
                      bg=CARD_BG, fg=th['accent'], relief='flat', anchor='w',
                      padx=16, pady=10, font=('Courier', 11, 'bold'),
                      cursor='hand2', command=choose,
                      highlightbackground=BORDER, highlightthickness=1)
        b.pack(fill='x', padx=30, pady=4)
    tk.Frame(root, bg=BG, height=12).pack()
    root.mainloop()
    return chosen['id']


def run(theme_id=None, group="GROUP 07", students=None):
    """Launch the themed mission control panel."""
    if theme_id is None:
        theme_id = pick_theme_dialog()
    if theme_id is None:
        return                              # user closed the picker
    app = TelloThemedPanel(theme_id=theme_id, group=group, students=students)
    app.mainloop()



# ============================================================================
# DIAGNOSTIC EDITION: passive measurements only.  No flight-control changes.
# All four diagnostic applications embed this same instrumentation block.
# ============================================================================
import atexit as _diag_atexit
import json as _diag_json
import queue as _diag_queue
import sys as _diag_sys
import hashlib as _diag_hashlib
from pathlib import Path as _DiagPath

_DIAG_RECORDER = None
_DIAG_LOCAL = threading.local()
_DIAG_SOURCE_SHA256 = "55dc18afc6955af9d1d77dbf8019c18705923a78f9db20f057bd24a33b4bd415"
_DIAG_VERSION = "1.0.1"
_DIAG_FIELDS = ("pitch", "roll", "yaw", "vgx", "vgy", "vgz", "h", "tof",
                "agx", "agy", "agz", "bat", "templ", "temph", "baro",
                "mid", "x", "y", "z")


def _diag_emit(event_type, **details):
    """Non-blocking, best-effort evidence collection; never controls the drone."""
    rec = _DIAG_RECORDER
    if rec is not None:
        try:
            rec.add(event_type, **details)
        except Exception:
            pass


def _diag_waypoint(name):
    _DIAG_LOCAL.waypoint = str(name)
    _diag_emit('waypoint_enter', waypoint=name)


class _DiagnosisRecorder:
    """Async append-only timeline and CSVs in Diagnosis/<script>/<run_id>."""

    def __init__(self):
        from datetime import datetime
        self.script = _DiagPath(__file__).stem
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
        self.folder = _DiagPath(__file__).resolve().parent / 'Diagnosis' / self.script / self.run_id
        self.folder.mkdir(parents=True, exist_ok=False)
        self.start_ns = time.monotonic_ns()
        self.queue = _diag_queue.Queue(maxsize=16384)
        self._queue_lock = threading.Lock()
        self._closed = False
        self._lost = 0
        self.last_telem_ns = None
        self._summary = {'commands': 0, 'refused': 0, 'timeouts': 0,
                         'telemetry_packets': 0, 'scans': 0, 'camera_events': 0}
        info = {'diagnostic_version': _DIAG_VERSION, 'script': self.script,
                'run_id': self.run_id, 'source_sha256': _DIAG_SOURCE_SHA256,
                'python': _diag_sys.version, 'working_directory': os.getcwd(),
                'diagnosis_folder': str(self.folder),
                'utc_start': datetime.now().astimezone().isoformat(),
                'note': 'Passive observation. SDK command timing includes original delay. '
                        'XY tracker coordinates are commanded estimates, not measured positions.'}
        (self.folder/'run_info.json').write_text(_diag_json.dumps(info, indent=2), encoding='utf-8')
        self._worker = threading.Thread(target=self._write_loop, name='DiagnosisWriter', daemon=True)
        self._worker.start()
        self.add('diagnostic_start', **info)

    def add(self, event_type, **details):
        if self._closed:
            return
        ns = time.monotonic_ns()
        obj = {'wall_time': datetime.datetime.now().astimezone().isoformat(timespec='milliseconds'),
               'monotonic_ns': ns, 'elapsed_ms': round((ns-self.start_ns)/1e6, 3),
               'event': event_type, 'thread': threading.current_thread().name,
               'waypoint': getattr(_DIAG_LOCAL, 'waypoint', ''), **details}
        with self._queue_lock:
            if self._closed:
                return
            try:
                self.queue.put_nowait(obj)
            except _diag_queue.Full:
                self._lost += 1

    def _write_loop(self):
        from contextlib import ExitStack
        cmd_fields = ['wall_time','elapsed_ms','waypoint','command','response',
                      'status','wire_ms','total_ms','original_delay_s',
                      'abort_before','abort_after','uncertain_before','uncertain_after',
                      'flight_unknown_before','flight_unknown_after','flying_before',
                      'flying_after','telem_age_before_ms','telem_age_after_ms',
                      'before_telemetry','after_telemetry','before_path','after_path','exception']
        telem_fields = ['wall_time','elapsed_ms','waypoint',*_DIAG_FIELDS]
        scan_fields = ['wall_time','elapsed_ms','waypoint','scan_dir','found',
                       'label','confidence','note','image','duration_ms','model_ready']
        camera_fields = ['wall_time','elapsed_ms','event','kind','message','source','frames','fps','frame_w','frame_h','drone_source']
        with ExitStack() as stack:
            tl=stack.enter_context(open(self.folder/'timeline.jsonl','a',encoding='utf-8', buffering=1))
            cf=stack.enter_context(open(self.folder/'commands.csv','w',encoding='utf-8-sig',newline=''))
            tf=stack.enter_context(open(self.folder/'telemetry.csv','w',encoding='utf-8-sig',newline=''))
            sf=stack.enter_context(open(self.folder/'scans.csv','w',encoding='utf-8-sig',newline=''))
            vf=stack.enter_context(open(self.folder/'camera.csv','w',encoding='utf-8-sig',newline=''))
            cw,tw,sw,vw=csv.DictWriter(cf,fieldnames=cmd_fields),csv.DictWriter(tf,fieldnames=telem_fields),csv.DictWriter(sf,fieldnames=scan_fields),csv.DictWriter(vf,fieldnames=camera_fields)
            for w in (cw,tw,sw,vw): w.writeheader()
            while True:
                event=self.queue.get()
                if event is None:
                    self.queue.task_done()
                    break
                try:
                    tl.write(_diag_json.dumps(event,ensure_ascii=False,default=str)+'\n')
                    kind=event['event']
                    if kind=='sdk_command_end':
                        self._summary['commands']+=1
                        if event.get('status')=='REFUSED': self._summary['refused']+=1
                        if event.get('status')=='NO_RESPONSE': self._summary['timeouts']+=1
                        cw.writerow({k:(_diag_json.dumps(event.get(k),ensure_ascii=False) if isinstance(event.get(k),(dict,list)) else event.get(k,'')) for k in cmd_fields})
                        cf.flush()
                    elif kind=='telemetry':
                        self._summary['telemetry_packets']+=1
                        tw.writerow({k:event.get(k,'') for k in telem_fields})
                        tf.flush()
                    elif kind=='scan_end':
                        self._summary['scans']+=1
                        sw.writerow({k:event.get(k,'') for k in scan_fields})
                        sf.flush()
                    elif kind.startswith('camera_'):
                        self._summary['camera_events']+=1
                        vw.writerow({k:event.get(k,'') for k in camera_fields})
                        vf.flush()
                except Exception:
                    self._lost+=1
                finally:
                    self.queue.task_done()
            cf.flush();tf.flush();sf.flush();vf.flush();tl.flush()

    def close(self):
        # Close producers before the sentinel. No event may be queued after it.
        with self._queue_lock:
            if self._closed:
                return
            self._closed=True
            try:
                self.queue.put_nowait({
                    'wall_time':datetime.datetime.now().astimezone().isoformat(timespec='milliseconds'),
                    'monotonic_ns':time.monotonic_ns(),'elapsed_ms':round((time.monotonic_ns()-self.start_ns)/1e6,3),
                    'event':'diagnostic_end','thread':threading.current_thread().name,'waypoint':''})
            except _diag_queue.Full:
                self._lost+=1
            try:
                self.queue.put_nowait(None)
            except _diag_queue.Full:
                try:self.queue.get_nowait();self.queue.task_done();self._lost+=1;self.queue.put_nowait(None)
                except Exception:pass
        self._worker.join(timeout=5.0)
        summary=dict(self._summary)
        summary.update(run_id=self.run_id, dropped_events=self._lost,
                       writer_alive=self._worker.is_alive())
        try:(self.folder/'summary.json').write_text(_diag_json.dumps(summary,indent=2),encoding='utf8')
        except Exception:pass


def _diag_close():
    rec=_DIAG_RECORDER
    if rec is not None:
        try:rec.close()
        except Exception:pass

_diag_atexit.register(_diag_close)


def _diag_flags(drone):
    try:
        age=(time.monotonic_ns()-_DIAG_RECORDER.last_telem_ns)/1e6 if (
             _DIAG_RECORDER and _DIAG_RECORDER.last_telem_ns) else None
        return {'abort':bool(drone.abort),'uncertain':bool(drone.uncertain),
                'flight_unknown':bool(drone.flight_unknown),'flying':bool(drone.flying),
                'telemetry_age_ms':round(age,1) if age is not None else None,
                'telemetry':dict(drone.telem) if drone.telem else {},
                'path':drone.tracker.snapshot() if drone.tracker is not None else None}
    except Exception:
        return {}


# Instrument the existing SDK interface without changing any response/timeout logic.
_DIAG_ORIGINAL_SEND=DroneProxy.send

def _diag_send(self,command,delay=2):
    start=time.monotonic_ns()
    before=_diag_flags(self)
    _diag_emit('sdk_command_begin',command=command,original_delay_s=delay,
               telemetry=before.get('telemetry'),flags={k:v for k,v in before.items() if k not in ('telemetry','path')})
    result=None;exc=None
    try:
        result=_DIAG_ORIGINAL_SEND(self,command,delay=delay)
        return result
    except Exception as e:
        exc=repr(e)
        raise
    finally:
        try:
            elapsed=(time.monotonic_ns()-start)/1e6
            after=_diag_flags(self)
            raw=getattr(_DIAG_LOCAL,'wire_times',{}).pop(id(self),None)
            wire_ms=(raw['reply']-raw['send'])/1e6 if raw and raw.get('reply') is not None else None
            response=(result if result is not None else '')
            normalized=str(response).lower()
            status=('EXCEPTION' if exc else 'BLOCKED' if normalized=='blocked' else
                    'NO_RESPONSE' if normalized=='' else
                    'REFUSED' if normalized.startswith('error') or normalized in ('unactive','unknown command','out of range') else
                    'CANCELLED' if normalized=='cancelled' else 'OK' if normalized=='ok' else
                    'VALUE' if str(command).strip().endswith('?') and not after.get('abort') else 'OTHER')
            _diag_emit('sdk_command_end',command=command,response=response,status=status,
                       original_delay_s=delay,wire_ms=round(wire_ms,3) if wire_ms is not None else None,
                       total_ms=round(elapsed,3),exception=exc,
                       abort_before=before.get('abort'),abort_after=after.get('abort'),
                       uncertain_before=before.get('uncertain'),uncertain_after=after.get('uncertain'),
                       flight_unknown_before=before.get('flight_unknown'),flight_unknown_after=after.get('flight_unknown'),
                       flying_before=before.get('flying'),flying_after=after.get('flying'),
                       telem_age_before_ms=before.get('telemetry_age_ms'),
                       telem_age_after_ms=after.get('telemetry_age_ms'),
                       before_telemetry=before.get('telemetry'),after_telemetry=after.get('telemetry'),
                       before_path=before.get('path'),after_path=after.get('path'))
        except Exception:
            # Diagnostic exceptions must never change original SDK command results.
            pass

DroneProxy.send=_diag_send

_DIAG_ORIGINAL_TELEM=DroneProxy.telem

def _diag_telem_set(self,value):
    _DIAG_ORIGINAL_TELEM.fset(self,value)
    rec=_DIAG_RECORDER
    if rec is not None:
        rec.last_telem_ns=time.monotonic_ns()
        try:_diag_emit('telemetry',**{key:value.get(key,'') for key in _DIAG_FIELDS})
        except Exception:pass

DroneProxy.telem=property(_DIAG_ORIGINAL_TELEM.fget,_diag_telem_set)

_DIAG_ORIGINAL_INIT=TelloThemedPanel.__init__
def _diag_panel_init(self,*args,**kwargs):
    global _DIAG_RECORDER
    if _DIAG_RECORDER is None:
        try:
            _DIAG_RECORDER=_DiagnosisRecorder()
        except Exception as err:
            # Data logging is optional; failure must not disable any original GUI feature.
            print(f"[Diagnosis] recorder unavailable: {err!r}", file=_diag_sys.stderr)
    try:
        _DIAG_ORIGINAL_INIT(self,*args,**kwargs)
        _diag_emit('gui_initialized',theme_id=self.theme_id,group=self.group)
    except Exception:
        _diag_emit('gui_init_error',exception=_diag_sys.exc_info()[1].__repr__())
        raise
TelloThemedPanel.__init__=_diag_panel_init

_DIAG_ORIGINAL_LOG=TelloThemedPanel.log
def _diag_gui_log(self,message,level='info'):
    _diag_emit('gui_log',level=level,message=message)
    return _DIAG_ORIGINAL_LOG(self,message,level)
TelloThemedPanel.log=_diag_gui_log

_DIAG_ORIGINAL_SCAN=TelloThemedPanel.scan_at_waypoint
def _diag_scan(self,waypoint,scan_dir):
    start=time.monotonic_ns()
    _diag_emit('scan_begin',waypoint=waypoint,scan_dir=scan_dir,
               model_ready=bool(self.detector.using_yolo and self.detector.model is not None),
               camera_on=bool(self._cam_on))
    result=None
    try:
        result=_DIAG_ORIGINAL_SCAN(self,waypoint,scan_dir)
        return result
    finally:
        try:
            fields=result if isinstance(result,dict) else {}
            _diag_emit('scan_end',waypoint=waypoint,scan_dir=scan_dir,
                       duration_ms=round((time.monotonic_ns()-start)/1e6,3),
                       model_ready=bool(self.detector.using_yolo and self.detector.model is not None),
                       found=fields.get('found',False), label=fields.get('label'),
                       confidence=fields.get('confidence'),note=fields.get('note'),
                       image=fields.get('image'))
        except Exception:
            pass
TelloThemedPanel.scan_at_waypoint=_diag_scan

_DIAG_ORIGINAL_CAMERA_MESSAGE=TelloThemedPanel._camera_message
def _diag_camera_message(self,session,kind,message):
    _diag_emit('camera_message',kind=kind,message=message,drone_source=session.get('drone_source'))
    return _DIAG_ORIGINAL_CAMERA_MESSAGE(self,session,kind,message)
TelloThemedPanel._camera_message=_diag_camera_message

_DIAG_ORIGINAL_CAMERA_ON=TelloThemedPanel._camera_on
_DIAG_ORIGINAL_CAMERA_OFF=TelloThemedPanel._camera_off
def _diag_camera_on(self):
    try:
        _diag_emit('camera_on_requested',drone_source=self._use_drone_camera(),connected=self._connected)
    except Exception:
        pass
    return _DIAG_ORIGINAL_CAMERA_ON(self)
def _diag_camera_off(self):
    _diag_emit('camera_off_requested')
    return _DIAG_ORIGINAL_CAMERA_OFF(self)
TelloThemedPanel._camera_on=_diag_camera_on
TelloThemedPanel._camera_off=_diag_camera_off

_DIAG_ORIGINAL_YOLO_LOAD=YoloDetector.load
def _diag_yolo_load(self,*args,**kwargs):
    start=time.monotonic_ns()
    _diag_emit('yolo_load_begin',args=[str(a) for a in args])
    try:
        result=_DIAG_ORIGINAL_YOLO_LOAD(self,*args,**kwargs)
        try:
            _diag_emit('yolo_load_end',result=str(result),using_yolo=self.using_yolo,
                       model_ready=self.model is not None,
                       duration_ms=round((time.monotonic_ns()-start)/1e6,3))
        except Exception:
            pass
        return result
    except Exception as e:
        _diag_emit('yolo_load_error',exception=repr(e))
        raise
YoloDetector.load=_diag_yolo_load

_DIAG_ORIGINAL_START=TelloThemedPanel._do_start
def _diag_start(self):
    try:
        _diag_emit('mission_start_request',model_ready=bool(self.detector.using_yolo and self.detector.model is not None),
                   camera_source=self.sel_camera.get(),mission=self.sel_mission.get(),
                   drone_flags=_diag_flags(self._drone) if self._drone else None)
    except Exception:
        pass
    return _DIAG_ORIGINAL_START(self)
TelloThemedPanel._do_start=_diag_start

_DIAG_ORIGINAL_DONE=TelloThemedPanel._on_mission_done
def _diag_done(self,*args,**kwargs):
    _diag_emit('mission_worker_done',drone_flags=_diag_flags(self._drone) if self._drone else None)
    return _DIAG_ORIGINAL_DONE(self,*args,**kwargs)
TelloThemedPanel._on_mission_done=_diag_done

_DIAG_ORIGINAL_CLOSE=TelloThemedPanel._on_close
def _diag_panel_close(self):
    _diag_emit('gui_close_requested')
    try:return _DIAG_ORIGINAL_CLOSE(self)
    finally:_diag_close()
TelloThemedPanel._on_close=_diag_panel_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tello themed mission control.")
    parser.add_argument('--theme', type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Theme number (1-5). Omit to show the picker.")
    parser.add_argument('--group', type=str, default="GROUP 07",
                        help="Group number shown in the header.")
    args = parser.parse_args()
    run(theme_id=args.theme, group=args.group)
