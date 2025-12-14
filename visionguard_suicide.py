"""
VisionGuard demo script for Digital Image Processing project.
Single-file implementation with three detection modes:
1. Accident detection
2. Suicide-related behavior detection
3. Theft/robbery detection & Robbery Behavior Potential (RBP)

This script is intended as a teaching / demo example, not a production system.
"""

import argparse
import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np

try:
    # Ultralytics YOLOv8 (install with: pip install ultralytics)
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    # The script will still import; user must install ultralytics to run the model.


# --------------------------- Utility classes ---------------------------------


class CentroidTracker:
    """
    Very simple centroid-based tracker that assigns integer IDs to detections
    based on nearest-neighbour matching between consecutive frames.
    """

    def __init__(self, max_distance=50.0, max_missing=30):
        self.next_id = 0
        self.tracks = {}  # track_id -> state dict
        self.max_distance = max_distance
        self.max_missing = max_missing

    def _create_track(self, centroid, bbox, cls_id):
        track_id = self.next_id
        self.next_id += 1
        self.tracks[track_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "cls_id": cls_id,
            "missed": 0,
            "history": deque(maxlen=32),
        }
        return track_id

    def update(self, detections):
        """
        detections: list of dicts with keys: 'bbox', 'cls_id'
            bbox = (x1, y1, x2, y2)
        Returns dict: track_id -> state dict (augmented with history)
        """
        if len(self.tracks) == 0:
            # Initialize tracks for all detections
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                self._create_track((cx, cy), det["bbox"], det["cls_id"])
            return self.tracks

        # Compute centroids of detections
        det_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            det_centroids.append((cx, cy))

        # Keep track of which detections have been matched
        unmatched_detections = set(range(len(detections)))
        updated_ids = set()

        # First, attempt to match each existing track to the nearest detection
        for track_id, state in list(self.tracks.items()):
            best_det_idx = None
            best_dist = None
            for det_idx in list(unmatched_detections):
                cx, cy = det_centroids[det_idx]
                tx, ty = state["centroid"]
                dist = math.hypot(cx - tx, cy - ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            if best_det_idx is not None and best_dist is not None and best_dist <= self.max_distance:
                # Match found
                unmatched_detections.remove(best_det_idx)
                det = detections[best_det_idx]
                cx, cy = det_centroids[best_det_idx]
                state["missed"] = 0
                state["centroid"] = (cx, cy)
                state["bbox"] = det["bbox"]
                state["cls_id"] = det["cls_id"]
                state["history"].append((cx, cy))
                updated_ids.add(track_id)
            else:
                # No match; increase missed counter
                state["missed"] += 1

        # Remove tracks that have been missing for too long
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


# --------------------------- Detection logic ----------------------------------


def iou(b1, b2):
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    area1 = abs(area1)
    area2 = abs(area2)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class AccidentDetector:
    def __init__(self, fps, speed_drop_ratio=0.5, min_speed=5.0, iou_thresh=0.05):
        self.fps = fps
        self.speed_drop_ratio = speed_drop_ratio
        self.min_speed = min_speed
        self.iou_thresh = iou_thresh
        # Per-track info
        self.track_states = defaultdict(
            lambda: {"prev_centroid": None, "prev_speed": 0.0, "speeds": deque(maxlen=10)}
        )

    def update(self, tracks, dt):
        """
        tracks: dict track_id -> state (must contain 'centroid' and 'bbox', 'cls_id')
        dt: seconds between frames
        Returns: list of accident events as (track_id1, track_id2)
        """
        vehicle_ids = [tid for tid, st in tracks.items() if st["cls_id"] in (1, 2, 3, 5, 7)]
        # In COCO: bicycle=1, car=2, motorbike=3, bus=5, truck=7
        accidents = []

        # Update speeds
        for tid in vehicle_ids:
            st = tracks[tid]
            cx, cy = st["centroid"]
            ts = self.track_states[tid]
            prev_c = ts["prev_centroid"]
            if prev_c is not None and dt > 0:
                dist = math.hypot(cx - prev_c[0], cy - prev_c[1])
                speed = dist / dt  # pixels per second
            else:
                speed = 0.0
            ts["speeds"].append(speed)
            ts["prev_speed"] = speed
            ts["prev_centroid"] = (cx, cy)

        # Check for collisions between pairs
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                id1 = vehicle_ids[i]
                id2 = vehicle_ids[j]
                st1 = tracks[id1]
                st2 = tracks[id2]
                box1 = st1["bbox"]
                box2 = st2["bbox"]
                ov = iou(box1, box2)
                if ov < self.iou_thresh:
                    continue

                ts1 = self.track_states[id1]
                ts2 = self.track_states[id2]
                if len(ts1["speeds"]) < 2 or len(ts2["speeds"]) < 2:
                    continue

                # Sudden speed drop for at least one vehicle
                v1 = np.mean(list(ts1["speeds"])[: len(ts1["speeds"]) // 2])
                v2 = np.mean(list(ts2["speeds"])[: len(ts2["speeds"]) // 2])
                last_v1 = ts1["speeds"][-1]
                last_v2 = ts2["speeds"][-1]

                sudden_brake1 = v1 > self.min_speed and last_v1 < v1 * (1.0 - self.speed_drop_ratio)
                sudden_brake2 = v2 > self.min_speed and last_v2 < v2 * (1.0 - self.speed_drop_ratio)
                if sudden_brake1 or sudden_brake2:
                    accidents.append((id1, id2))
        return accidents


class SuicideRiskDetector:
    """
    Detect suicide-related risk behavior near a hazard zone using person tracks.

    A track is considered "risky" if:
    - The person spends at least `dwell_threshold_sec` seconds continuously inside
      the hazard ROI, AND
    - They have entered the ROI at least `min_entries` times (to distinguish
      pacing near the edge from just passing through once).

    Note: hazard_roi_norm is given in normalized coordinates (x1, y1, x2, y2)
    with values in [0, 1], and is converted to pixel coordinates each frame.
    """

    def __init__(
        self,
        fps,
        hazard_roi_norm=(0.05, 0.6, 0.95, 0.98),
        dwell_threshold_sec=4.0,
        min_entries=1,
    ):
        self.fps = fps
        self.hazard_roi_norm = hazard_roi_norm
        # Number of frames a person must remain in the hazard zone
        self.dwell_threshold_frames = int(dwell_threshold_sec * fps)
        # Minimum number of separate entries into the zone (1 = at least once)
        self.min_entries = min_entries
        # Per-track state: dwell time, entry count, last inside status
        self.track_states = defaultdict(
            lambda: {"dwell_frames": 0, "inside": False, "entries": 0, "last_inside": False}
        )

    def update(self, tracks, frame_shape):
        h, w = frame_shape[:2]
        x1n, yn, x2n, y2n = self.hazard_roi_norm
        x1 = int(x1n * w)
        y1 = int(yn * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)
        hazard_box = (x1, y1, x2, y2)

        events = []
        for tid, st in tracks.items():
            # Only consider person class from YOLO (COCO id 0)
            if st["cls_id"] != 0:
                continue

            cx, cy = st["centroid"]
            inside = x1 <= cx <= x2 and y1 <= cy <= y2
            ts = self.track_states[tid]

            # Update dwell time and entry count
            ts["inside"] = inside
            if inside:
                ts["dwell_frames"] += 1
            if inside and not ts["last_inside"]:
                # New entry into hazard zone
                ts["entries"] += 1
            ts["last_inside"] = inside

            # Check risk condition
            if ts["dwell_frames"] >= self.dwell_threshold_frames and ts["entries"] >= self.min_entries:
                events.append(tid)

        return hazard_box, events


class RobberyRiskDetector:
    """
    Simple rule-based Robbery Behavior Potential (RBP) estimator.
    """

    def __init__(
        self,
        fps,
        hv_roi_norm=(0.2, 0.3, 0.8, 0.8),
        exit_roi_norm=(0.35, 0.0, 0.65, 0.25),
        loiter_time_sec=10.0,
    ):
        self.fps = fps
        self.hv_roi_norm = hv_roi_norm
        self.exit_roi_norm = exit_roi_norm
        self.loiter_frames = int(loiter_time_sec * fps)
        self.track_states = defaultdict(
            lambda: {
                "hv_dwell": 0,
                "exit_dwell": 0,
                "visited_hv": False,
                "entered_exit_after_hv": False,
                "rbp": 0.0,
            }
        )

    def _roi_box(self, roi_norm, frame_shape):
        h, w = frame_shape[:2]
        x1n, yn, x2n, y2n = roi_norm
        return (int(x1n * w), int(yn * h), int(x2n * w), int(y2n * h))

    def update(self, tracks, frame_shape):
        hv_box = self._roi_box(self.hv_roi_norm, frame_shape)
        exit_box = self._roi_box(self.exit_roi_norm, frame_shape)

        # Count group size near exit for this frame
        exit_people = 0
        for tid, st in tracks.items():
            if st["cls_id"] != 0:
                continue
            cx, cy = st["centroid"]
            x1e, y1e, x2e, y2e = exit_box
            if x1e <= cx <= x2e and y1e <= cy <= y2e:
                exit_people += 1

        suspicious_tracks = []

        for tid, st in tracks.items():
            if st["cls_id"] != 0:
                continue
            cx, cy = st["centroid"]
            ts = self.track_states[tid]

            # Update HV dwell
            x1h, y1h, x2h, y2h = hv_box
            in_hv = x1h <= cx <= x2h and y1h <= cy <= y2h
            if in_hv:
                ts["hv_dwell"] += 1
                ts["visited_hv"] = True

            # Update exit dwell
            x1e, y1e, x2e, y2e = exit_box
            in_exit = x1e <= cx <= x2e and y1e <= cy <= y2e
            if in_exit:
                ts["exit_dwell"] += 1
                if ts["visited_hv"]:
                    ts["entered_exit_after_hv"] = True

            # Compute simple RBP score [0,1]
            loiter_score = min(1.0, ts["hv_dwell"] / float(self.loiter_frames)) if self.loiter_frames > 0 else 0.0
            group_score = min(1.0, exit_people / 5.0)  # 0 when alone, ~1 when crowd >=5
            exit_score = 1.0 if ts["entered_exit_after_hv"] else 0.0

            rbp = 0.5 * loiter_score + 0.3 * exit_score + 0.2 * group_score
            ts["rbp"] = rbp

            if rbp >= 0.6:
                suspicious_tracks.append((tid, rbp))

        return hv_box, exit_box, suspicious_tracks


# --------------------------- Main VisionGuard pipeline -----------------------


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
    # the list continues in the real COCO, but we only use first few
]


class VisionGuardDemo:
    def __init__(self, mode, source, save_output=False, output_path="output.mp4", display=True):
        self.mode = mode
        self.source = source
        self.save_output = save_output
        self.output_path = output_path
        self.display = display

        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Install it with 'pip install ultralytics' before running this script."
            )

        self.model = YOLO("yolov8n.pt")  # small model, good enough for demo
        self.tracker = CentroidTracker(max_distance=60.0, max_missing=15)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0  # fallback
        self.fps = fps

        self.acc_detector = AccidentDetector(fps=self.fps)
        self.suicide_detector = SuicideRiskDetector(fps=self.fps)
        self.robbery_detector = RobberyRiskDetector(fps=self.fps)

        self.writer = None

    def run(self):
        frame_idx = 0
        prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1
            now = time.time()
            dt = max(1e-3, now - prev_time)
            prev_time = now

            # Inference
            results = self.model(frame, verbose=False)[0]
            detections = []

            if results.boxes is not None:
                boxes = results.boxes
                # xyxy shape: (N,4)
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
                    if s < 0.4:
                        continue
                    if c >= len(COCO_CLASS_NAMES):
                        continue
                    detections.append({"bbox": (float(x1), float(y1), float(x2), float(y2)), "cls_id": int(c)})

            # Update tracks
            tracks = self.tracker.update(detections)

            # Draw detections
            annotated = frame.copy()
            for tid, st in tracks.items():
                x1, y1, x2, y2 = map(int, st["bbox"])
                cls_id = st["cls_id"]
                label = COCO_CLASS_NAMES[cls_id] if cls_id < len(COCO_CLASS_NAMES) else str(cls_id)
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

            if self.mode == "accident":
                accidents = self.acc_detector.update(tracks, dt)
                for id1, id2 in accidents:
                    for tid in (id1, id2):
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

            elif self.mode == "suicide":
                hazard_box, risky_ids = self.suicide_detector.update(tracks, frame.shape)
                x1, y1, x2, y2 = hazard_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    annotated,
                    "HAZARD ZONE",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                for tid in risky_ids:
                    if tid not in tracks:
                        continue
                    x1, y1, x2, y2 = map(int, tracks[tid]["bbox"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if risky_ids:
                    cv2.putText(
                        annotated,
                        "SUICIDE RISK BEHAVIOR",
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            elif self.mode == "robbery":
                hv_box, exit_box, suspicious_tracks = self.robbery_detector.update(tracks, frame.shape)

                # Draw ROIs
                x1h, y1h, x2h, y2h = hv_box
                cv2.rectangle(annotated, (x1h, y1h), (x2h, y2h), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    "HIGH VALUE AREA",
                    (x1h, y1h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                x1e, y1e, x2e, y2e = exit_box
                cv2.rectangle(annotated, (x1e, y1e), (x2e, y2e), (255, 0, 255), 2)
                cv2.putText(
                    annotated,
                    "EXIT",
                    (x1e, y1e + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                for tid, rbp in suspicious_tracks:
                    if tid not in tracks:
                        continue
                    x1, y1, x2, y2 = map(int, tracks[tid]["bbox"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"RBP={rbp:.2f}",
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if suspicious_tracks:
                    cv2.putText(
                        annotated,
                        "ROBBERY RISK",
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # FPS text / frame index
            cv2.putText(
                annotated,
                f"Frame {frame_idx}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Initialize writer once we know frame size
            if self.save_output and self.writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

            if self.writer is not None:
                self.writer.write(annotated)

            if self.display:
                cv2.imshow("VisionGuard Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

        self.cap.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard demo: accident, suicide risk, robbery detection.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accident", "suicide", "robbery"],
        default="accident",
        help="Which detector to run.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/accident_demo.mp4",
        help="Path to input video file or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visionguard_output.mp4",
        help="Where to save annotated video if --save is given.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not show GUI window (useful on headless machines).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source
    try:
        # If source is a digit, treat as camera index
        source_int = int(source)
        source = source_int
    except ValueError:
        pass

    vg = VisionGuardDemo(
        mode=args.mode,
        source=source,
        save_output=args.save,
        output_path=args.output,
        display=not args.no_display,
    )
    vg.run()


if __name__ == "__main__":
    main()
