"""
VisionGuard (Accident Detection Only) - Demo Script

This is a cleaned, single-file demo focused ONLY on accident detection:
- Detect vehicles with YOLOv8 (Ultralytics)
- Track with a lightweight centroid tracker
- Trigger "ACCIDENT DETECTED" when:
  (1) two vehicle boxes overlap (IoU >= threshold) AND
  (2) at least one vehicle shows a sudden speed drop

Notes
- This is a rule-based demo, not a production-grade accident detector.
- Speed is measured in pixels/second using frame-to-frame centroid movement.
- For stability, dt is derived from the video FPS (not wall-clock time).

Requirements
- Python 3.10+ (works with Python 3.14)
- pip install ultralytics opencv-python numpy

Example
  python visionguard_accident_only.py --source data/accident_demo.mp4
  python visionguard_accident_only.py --source 0 --save --output out.mp4
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

    def _create_track(self, centroid: tuple[float, float], bbox: tuple[float, float, float, float], cls_id: int) -> int:
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


@dataclass
class AccidentParams:
    speed_drop_ratio: float = 0.5  # 0.5 means "drop by 50%"
    min_speed: float = 5.0         # pixels/sec threshold before we consider drops meaningful
    iou_thresh: float = 0.05       # overlap threshold


class AccidentDetector:
    """
    Detects "accident events" by overlap + sudden speed drop.
    """

    # COCO ids used by YOLOv8n.pt: bicycle=1, car=2, motorbike=3, bus=5, truck=7
    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

    def __init__(self, params: AccidentParams):
        self.params = params
        self.track_states = defaultdict(
            lambda: {"prev_centroid": None, "speeds": deque(maxlen=10)}
        )

    def update(self, tracks: dict[int, dict], dt: float) -> list[tuple[int, int]]:
        """
        tracks: dict track_id -> state (must contain 'centroid', 'bbox', 'cls_id')
        dt: seconds per frame
        Returns: list of accident events as (track_id1, track_id2)
        """
        if dt <= 0:
            dt = 1e-3

        vehicle_ids = [tid for tid, st in tracks.items() if int(st.get("cls_id", -1)) in self.VEHICLE_CLASS_IDS]
        accidents: list[tuple[int, int]] = []

        # Update speeds
        for tid in vehicle_ids:
            cx, cy = tracks[tid]["centroid"]
            ts = self.track_states[tid]
            prev_c = ts["prev_centroid"]
            if prev_c is not None:
                dist = math.hypot(cx - prev_c[0], cy - prev_c[1])
                speed = dist / dt  # pixels per second
            else:
                speed = 0.0
            ts["speeds"].append(float(speed))
            ts["prev_centroid"] = (float(cx), float(cy))

        # Check collisions between pairs
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                id1 = vehicle_ids[i]
                id2 = vehicle_ids[j]
                box1 = tracks[id1]["bbox"]
                box2 = tracks[id2]["bbox"]

                if iou(box1, box2) < self.params.iou_thresh:
                    continue

                ts1 = self.track_states[id1]
                ts2 = self.track_states[id2]
                if len(ts1["speeds"]) < 4 or len(ts2["speeds"]) < 4:
                    continue

                # Compare "older average" vs last speed (simple sudden drop heuristic)
                half1 = max(1, len(ts1["speeds"]) // 2)
                half2 = max(1, len(ts2["speeds"]) // 2)
                prev_mean1 = float(np.mean(list(ts1["speeds"])[:half1]))
                prev_mean2 = float(np.mean(list(ts2["speeds"])[:half2]))
                last1 = ts1["speeds"][-1]
                last2 = ts2["speeds"][-1]

                sudden_brake1 = prev_mean1 > self.params.min_speed and last1 < prev_mean1 * (1.0 - self.params.speed_drop_ratio)
                sudden_brake2 = prev_mean2 > self.params.min_speed and last2 < prev_mean2 * (1.0 - self.params.speed_drop_ratio)

                if sudden_brake1 or sudden_brake2:
                    accidents.append((id1, id2))

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
            raise ImportError(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )

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
                    detections.append(
                        {"bbox": (float(x1), float(y1), float(x2), float(y2)), "cls_id": int(c)}
                    )

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

            # Accident logic
            accidents = self.acc_detector.update(tracks, self.dt)
            if accidents:
                accident_latched_frames = int(self.fps * 1.0)  # keep visible ~1 sec

            if accident_latched_frames > 0:
                accident_latched_frames -= 1

                # Highlight boxes involved in current frame accidents
                involved = set()
                for a, b in accidents:
                    involved.add(a)
                    involved.add(b)

                for tid in involved:
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
    parser.add_argument(
        "--source",
        type=str,
        default="data/accident_demo.mp4",
        help="Path to input video file or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path (e.g., yolov8n.pt, yolov8s.pt, custom.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold.",
    )
    parser.add_argument("--save", action="store_true", help="Save annotated output video.")
    parser.add_argument(
        "--output",
        type=str,
        default="visionguard_output.mp4",
        help="Where to save annotated video if --save is given.",
    )
    parser.add_argument("--no-display", action="store_true", help="Do not show GUI window.")
    parser.add_argument("--max-distance", type=float, default=60.0, help="Tracker matching distance (pixels).")
    parser.add_argument("--max-missing", type=int, default=15, help="Tracker max missing frames before drop.")
    parser.add_argument("--iou-thresh", type=float, default=0.05, help="IoU threshold to consider overlap/collision.")
    parser.add_argument("--min-speed", type=float, default=5.0, help="Min speed (px/s) before checking speed drops.")
    parser.add_argument("--speed-drop-ratio", type=float, default=0.5, help="Required drop ratio, e.g. 0.5 => 50%% drop.")
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
        ),
    )
    demo.run()


if __name__ == "__main__":
    main()
