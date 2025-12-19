"""
VisionGuard (Accident Detection Only) - Enhanced Heuristics Demo

This single-file demo focuses ONLY on accident detection.

What changed vs the earlier overlap-only rule:
- Still uses YOLOv8 (Ultralytics) for vehicle detection.
- Still uses a lightweight centroid tracker (fast).
- Accident decision is now based on *kinematics + proximity*:
    (A) "pre-impact risk": vehicles get close AND time-to-collision (TTC) becomes small (or IoU overlaps),
    (B) "impact signature": strong deceleration and/or abrupt direction change near that moment,
    (C) "post-impact confirmation": both vehicles slow/stop and remain close for a short time.

This greatly reduces false alarms from simple box overlaps and detects many real crashes
even when boxes don't overlap much (e.g., missed detections, partial occlusion, jitter).

Notes
- This is still rule-based (not a trained crash classifier).
- All speeds/accelerations are in pixels/second and pixels/second^2 (camera-dependent).
- dt is derived from video FPS (keeps original video speed).

Requirements
- Python 3.10+ (works with Python 3.14)
- pip install ultralytics opencv-python numpy

Example
  python visionguard_accident.py --source data/accident_demo.mp4
  python visionguard_accident.py --source 0 --save --output out.mp4
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


# --------------------------- Utility classes ---------------------------------


class CentroidTracker:
    """
    Very simple centroid-based tracker that assigns integer IDs to detections
    based on nearest-neighbour matching between consecutive frames.
    """

    def __init__(self, max_distance: float = 60.0, max_missing: int = 15):
        self.next_id = 0
        self.tracks: dict[int, dict] = {}  # track_id -> state dict
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)

    def _create_track(
        self,
        centroid: tuple[float, float],
        bbox: tuple[float, float, float, float],
        cls_id: int,
    ) -> int:
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "cls_id": int(cls_id),
            "missed": 0,
            "history": deque(maxlen=32),
        }
        return track_id

    def update(self, detections: list[dict]) -> dict[int, dict]:
        """
        detections: list of dicts with keys: 'bbox', 'cls_id'
            bbox = (x1, y1, x2, y2) in float pixels
        Returns dict: track_id -> state dict (augmented with history)
        """
        if not detections and not self.tracks:
            return self.tracks

        if len(self.tracks) == 0:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                self._create_track((cx, cy), det["bbox"], det["cls_id"])
            return self.tracks

        # Compute centroids of detections
        det_centroids: list[tuple[float, float]] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det_centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        unmatched_detections = set(range(len(detections)))

        # Match each existing track to nearest detection
        for track_id, state in list(self.tracks.items()):
            best_det_idx = None
            best_dist = None

            tx, ty = state["centroid"]
            for det_idx in list(unmatched_detections):
                cx, cy = det_centroids[det_idx]
                dist = math.hypot(cx - tx, cy - ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            if best_det_idx is not None and best_dist is not None and best_dist <= self.max_distance:
                unmatched_detections.remove(best_det_idx)
                det = detections[best_det_idx]
                cx, cy = det_centroids[best_det_idx]
                state["missed"] = 0
                state["centroid"] = (cx, cy)
                state["bbox"] = det["bbox"]
                state["cls_id"] = int(det["cls_id"])
                state["history"].append((cx, cy))
            else:
                state["missed"] += 1

        # Remove stale tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["missed"] > self.max_missing:
                del self.tracks[track_id]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self._create_track((cx, cy), det["bbox"], det["cls_id"])

        return self.tracks


# --------------------------- Accident detection logic -------------------------


def iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, (x12 - x11)) * max(0.0, (y12 - y11))
    area2 = max(0.0, (x22 - x21)) * max(0.0, (y22 - y21))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (radians)."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


@dataclass
class AccidentParams:
    # Backward compatible settings (used as secondary heuristics)
    speed_drop_ratio: float = 0.5  # drop by 50%
    min_speed: float = 5.0         # px/s before checking speed drops
    iou_thresh: float = 0.05       # "strong overlap" threshold

    # New kinematics/proximity parameters (primary heuristics)
    ttc_thresh: float = 1.0        # seconds; smaller = more strict
    proximity_factor: float = 1.20 # * average bbox diagonal
    min_closing_speed: float = 25.0  # px/s; ignore tiny closing speeds
    hard_decel: float = 250.0        # px/s^2; strong deceleration magnitude
    dir_change_deg: float = 35.0     # degrees; abrupt direction change
    stop_speed: float = 15.0         # px/s; "almost stopped"
    contact_frames: int = 2          # consecutive frames for "contact/risk"
    post_stop_frames: int = 12       # frames to look for "stop + stay close" after contact
    sustain_frames: int = 6          # frames of sustained stop+close required
    iou_soft: float = 0.01           # small overlap hint (helps with jitter)
    cooldown_seconds: float = 2.0    # do not retrigger too often for same pair


class AccidentDetector:
    """
    Detects accident events from tracked vehicles using kinematics + proximity.

    Returns: list of (track_id1, track_id2) that triggered in the current frame.
    """

    # COCO ids used by YOLOv8n.pt: bicycle=1, car=2, motorbike=3, bus=5, truck=7
    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

    def __init__(self, params: AccidentParams):
        self.params = params

        # Per-track kinematics state
        self.track_states = defaultdict(
            lambda: {
                "prev_centroid": None,
                "prev_speed": 0.0,
                "prev_dir": 0.0,
                "vel": (0.0, 0.0),
                "speeds": deque(maxlen=12),
                "accels": deque(maxlen=12),
                "dirs": deque(maxlen=12),
            }
        )

        # Per-pair state: candidate contact -> confirmed accident
        # key: (min_id, max_id)
        self.pair_states = defaultdict(
            lambda: {
                "contact": 0,
                "post": 0,
                "sustain": 0,
                "impact_seen": False,
                "last_trigger_frame": -10_000_000,
            }
        )

    def _bbox_diag(self, b: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = b
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        return float(math.hypot(w, h))

    def update(self, tracks: dict[int, dict], dt: float, frame_idx: int) -> list[tuple[int, int]]:
        if dt <= 0:
            dt = 1e-3

        p = self.params
        dir_change_rad = float(p.dir_change_deg) * math.pi / 180.0
        cooldown_frames = int(max(1.0, float(p.cooldown_seconds) / dt))

        vehicle_ids = [
            tid for tid, st in tracks.items()
            if int(st.get("cls_id", -1)) in self.VEHICLE_CLASS_IDS
        ]
        accidents: list[tuple[int, int]] = []

        # --- Update per-track kinematics ---
        for tid in vehicle_ids:
            cx, cy = tracks[tid]["centroid"]
            ts = self.track_states[tid]
            prev_c = ts["prev_centroid"]

            if prev_c is not None:
                vx = (float(cx) - float(prev_c[0])) / dt
                vy = (float(cy) - float(prev_c[1])) / dt
                speed = float(math.hypot(vx, vy))
                prev_speed = float(ts["prev_speed"])
                accel = (speed - prev_speed) / dt
                direction = float(math.atan2(vy, vx)) if speed > 1e-3 else float(ts["prev_dir"])
            else:
                vx, vy = 0.0, 0.0
                speed = 0.0
                accel = 0.0
                direction = 0.0

            ts["vel"] = (vx, vy)
            ts["speeds"].append(speed)
            ts["accels"].append(accel)
            ts["dirs"].append(direction)

            ts["prev_centroid"] = (float(cx), float(cy))
            ts["prev_speed"] = float(speed)
            ts["prev_dir"] = float(direction)

        # --- Pairwise accident scoring ---
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                id1 = vehicle_ids[i]
                id2 = vehicle_ids[j]
                key = (id1, id2) if id1 < id2 else (id2, id1)
                ps = self.pair_states[key]

                box1 = tracks[id1]["bbox"]
                box2 = tracks[id2]["bbox"]
                c1 = tracks[id1]["centroid"]
                c2 = tracks[id2]["centroid"]

                dx = float(c2[0] - c1[0])
                dy = float(c2[1] - c1[1])
                dist = float(math.hypot(dx, dy))

                diag = 0.5 * (self._bbox_diag(box1) + self._bbox_diag(box2))
                proximity = float(p.proximity_factor) * diag

                # Relative closing speed along the line of centers
                v1 = self.track_states[id1]["vel"]
                v2 = self.track_states[id2]["vel"]
                rvx = float(v2[0] - v1[0])
                rvy = float(v2[1] - v1[1])

                if dist > 1e-3:
                    ux, uy = dx / dist, dy / dist
                    closing_speed = -(rvx * ux + rvy * uy)  # positive means approaching
                else:
                    closing_speed = 0.0

                ttc = (dist / closing_speed) if closing_speed > 1e-3 else float("inf")
                ov = iou(box1, box2)

                # Candidate "contact/risk" conditions
                near = dist < proximity
                risky = (near and (ttc < p.ttc_thresh) and (closing_speed > p.min_closing_speed))
                overlap_hint = (ov >= p.iou_thresh) or (ov >= p.iou_soft and near)

                potential_contact = risky or overlap_hint

                if potential_contact:
                    ps["contact"] = int(ps["contact"]) + 1
                else:
                    # decay contact counter (prevents flicker)
                    ps["contact"] = max(0, int(ps["contact"]) - 1)

                # When contact is stable for a few frames, open a post-impact confirmation window.
                if ps["contact"] >= int(p.contact_frames):
                    ps["post"] = int(p.post_stop_frames)

                # Impact signature: hard decel OR abrupt direction change around contact
                ts1 = self.track_states[id1]
                ts2 = self.track_states[id2]

                a1 = float(ts1["accels"][-1]) if ts1["accels"] else 0.0
                a2 = float(ts2["accels"][-1]) if ts2["accels"] else 0.0
                s1 = float(ts1["speeds"][-1]) if ts1["speeds"] else 0.0
                s2 = float(ts2["speeds"][-1]) if ts2["speeds"] else 0.0

                # Direction change vs recent median direction (robust)
                if len(ts1["dirs"]) >= 4:
                    base1 = float(np.median(list(ts1["dirs"])[:-1]))
                    dchg1 = _angle_diff(float(ts1["dirs"][-1]), base1)
                else:
                    dchg1 = 0.0
                if len(ts2["dirs"]) >= 4:
                    base2 = float(np.median(list(ts2["dirs"])[:-1]))
                    dchg2 = _angle_diff(float(ts2["dirs"][-1]), base2)
                else:
                    dchg2 = 0.0

                hard_decel = (a1 < -p.hard_decel and s1 > p.min_speed) or (a2 < -p.hard_decel and s2 > p.min_speed)
                hard_turn = (dchg1 > dir_change_rad and s1 > p.min_speed) or (dchg2 > dir_change_rad and s2 > p.min_speed)

                # Backward-compatible "sudden brake" heuristic (helps in some videos)
                sudden_brake = False
                if len(ts1["speeds"]) >= 6:
                    half = max(1, len(ts1["speeds"]) // 2)
                    prev_mean = float(np.mean(list(ts1["speeds"])[:half]))
                    sudden_brake = sudden_brake or (prev_mean > p.min_speed and s1 < prev_mean * (1.0 - p.speed_drop_ratio))
                if len(ts2["speeds"]) >= 6:
                    half = max(1, len(ts2["speeds"]) // 2)
                    prev_mean = float(np.mean(list(ts2["speeds"])[:half]))
                    sudden_brake = sudden_brake or (prev_mean > p.min_speed and s2 < prev_mean * (1.0 - p.speed_drop_ratio))

                if ps["contact"] >= int(p.contact_frames) and (hard_decel or hard_turn or sudden_brake):
                    ps["impact_seen"] = True

                # Post-impact confirmation window: both slow/stop and remain close.
                if ps["post"] > 0:
                    ps["post"] = int(ps["post"]) - 1
                    both_slow = (s1 < p.stop_speed) and (s2 < p.stop_speed)
                    still_close = dist < (proximity * 1.4)
                    if both_slow and still_close:
                        ps["sustain"] = int(ps["sustain"]) + 1
                    else:
                        ps["sustain"] = 0
                else:
                    ps["sustain"] = 0
                    ps["impact_seen"] = False  # window expired

                # Trigger
                if (
                    ps["impact_seen"]
                    and ps["sustain"] >= int(p.sustain_frames)
                    and (frame_idx - int(ps["last_trigger_frame"])) >= cooldown_frames
                ):
                    accidents.append((id1, id2))
                    ps["last_trigger_frame"] = int(frame_idx)

                    # Reset to avoid immediate re-triggering within same stop
                    ps["contact"] = 0
                    ps["post"] = 0
                    ps["sustain"] = 0
                    ps["impact_seen"] = False

        return accidents


# --------------------------- Main pipeline -----------------------------------


COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
]


class VisionGuardAccidentDemo:
    def __init__(
        self,
        source,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        save_output: bool = False,
        output_path: str = "visionguard_output.mp4",
        display: bool = True,
        tracker_max_distance: float = 60.0,
        tracker_max_missing: int = 15,
        accident_params: AccidentParams | None = None,
    ):
        if YOLO is None:  # pragma: no cover
            raise ImportError("ultralytics is not installed. Install it with: pip install ultralytics")

        self.source = source
        self.model = YOLO(model_path)
        self.conf = float(conf)

        self.tracker = CentroidTracker(max_distance=tracker_max_distance, max_missing=tracker_max_missing)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0
        self.fps = fps
        self.dt = 1.0 / self.fps

        self.acc_detector = AccidentDetector(accident_params or AccidentParams())

        self.save_output = bool(save_output)
        self.output_path = str(output_path)
        self.display = bool(display)
        self.writer = None

    def _ensure_writer(self, frame_shape):
        if not self.save_output or self.writer is not None:
            return
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

    def run(self):
        frame_idx = 0
        accident_latched_frames = 0  # keep label visible briefly after trigger
        involved_latched: set[int] = set()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1

            # Inference
            results = self.model(frame, verbose=False, conf=self.conf)[0]

            detections: list[dict] = []
            if results.boxes is not None:
                boxes = results.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, confs):
                    if float(s) < self.conf:
                        continue
                    if c < 0:
                        continue
                    detections.append({"bbox": (float(x1), float(y1), float(x2), float(y2)), "cls_id": int(c)})

            # Update tracks
            tracks = self.tracker.update(detections)

            annotated = frame.copy()

            # Draw tracks
            for tid, st in tracks.items():
                x1, y1, x2, y2 = map(int, st["bbox"])
                cls_id = int(st["cls_id"])
                label = COCO_CLASS_NAMES[cls_id] if 0 <= cls_id < len(COCO_CLASS_NAMES) else str(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"ID {tid} {label}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Accident logic (enhanced)
            accidents = self.acc_detector.update(tracks, self.dt, frame_idx)
            if accidents:
                accident_latched_frames = int(self.fps * 1.0)  # keep visible ~1 sec
                involved_latched = set()
                for a, b in accidents:
                    involved_latched.add(a)
                    involved_latched.add(b)

            if accident_latched_frames > 0:
                accident_latched_frames -= 1

                for tid in involved_latched:
                    if tid not in tracks:
                        continue
                    x1, y1, x2, y2 = map(int, tracks[tid]["bbox"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

                cv2.putText(
                    annotated,
                    "ACCIDENT DETECTED",
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Info text
            cv2.putText(
                annotated,
                f"Frame {frame_idx} | FPS={self.fps:.1f}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Writer / display
            self._ensure_writer(annotated.shape)
            if self.writer is not None:
                self.writer.write(annotated)

            if self.display:
                cv2.imshow("VisionGuard - Accident Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.display:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard demo (accident detection only).")
    parser.add_argument("--source", type=str, default="data/accident_demo.mp4",
                        help="Path to input video file or camera index (e.g., 0).")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model path (e.g., yolov8n.pt, yolov8s.pt, custom.pt).")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save annotated output video.")
    parser.add_argument("--output", type=str, default="visionguard_output.mp4",
                        help="Where to save annotated video if --save is given.")
    parser.add_argument("--no-display", action="store_true", help="Do not show GUI window.")
    parser.add_argument("--max-distance", type=float, default=60.0, help="Tracker matching distance (pixels).")
    parser.add_argument("--max-missing", type=int, default=15, help="Tracker max missing frames before drop.")

    # Accident parameters
    parser.add_argument("--iou-thresh", type=float, default=0.05, help="IoU threshold to consider strong overlap.")
    parser.add_argument("--iou-soft", type=float, default=0.01, help="Small IoU hint (helps with jitter).")

    parser.add_argument("--min-speed", type=float, default=5.0, help="Min speed (px/s) before checking speed drops.")
    parser.add_argument("--speed-drop-ratio", type=float, default=0.5,
                        help="Required drop ratio, e.g. 0.5 => 50%% drop.")

    parser.add_argument("--ttc-thresh", type=float, default=1.0, help="Time-to-collision threshold (seconds).")
    parser.add_argument("--proximity-factor", type=float, default=1.20,
                        help="Proximity threshold as factor * avg bbox diagonal.")
    parser.add_argument("--min-closing-speed", type=float, default=25.0,
                        help="Min closing speed along center line (px/s) to consider TTC meaningful.")
    parser.add_argument("--hard-decel", type=float, default=250.0,
                        help="Hard deceleration threshold (px/s^2).")
    parser.add_argument("--dir-change-deg", type=float, default=35.0,
                        help="Direction change threshold (deg).")
    parser.add_argument("--stop-speed", type=float, default=15.0,
                        help="Speed below which a vehicle is considered almost stopped (px/s).")
    parser.add_argument("--contact-frames", type=int, default=2,
                        help="Consecutive frames required to open post-impact window.")
    parser.add_argument("--post-stop-frames", type=int, default=12,
                        help="Frames after contact to look for stop+close confirmation.")
    parser.add_argument("--sustain-frames", type=int, default=6,
                        help="Frames of sustained stop+close required to confirm accident.")
    parser.add_argument("--cooldown-seconds", type=float, default=2.0,
                        help="Cooldown per pair to avoid repeated triggers.")
    return parser.parse_args()


def main():
    args = parse_args()

    # If source is a digit, treat it as camera index
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    demo = VisionGuardAccidentDemo(
        source=source,
        model_path=args.model,
        conf=args.conf,
        save_output=args.save,
        output_path=args.output,
        display=not args.no_display,
        tracker_max_distance=args.max_distance,
        tracker_max_missing=args.max_missing,
        accident_params=AccidentParams(
            speed_drop_ratio=float(args.speed_drop_ratio),
            min_speed=float(args.min_speed),
            iou_thresh=float(args.iou_thresh),
            ttc_thresh=float(args.ttc_thresh),
            proximity_factor=float(args.proximity_factor),
            min_closing_speed=float(args.min_closing_speed),
            hard_decel=float(args.hard_decel),
            dir_change_deg=float(args.dir_change_deg),
            stop_speed=float(args.stop_speed),
            contact_frames=int(args.contact_frames),
            post_stop_frames=int(args.post_stop_frames),
            sustain_frames=int(args.sustain_frames),
            iou_soft=float(args.iou_soft),
            cooldown_seconds=float(args.cooldown_seconds),
        ),
    )
    demo.run()


if __name__ == "__main__":
    main()
