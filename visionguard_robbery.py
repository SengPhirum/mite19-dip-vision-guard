"""
VisionGuard - Robbery demo (FAST, Python 3.14 compatible)

- No PyTorch / Ultralytics required.
- Runs in real-time on many laptops by:
  * processing at a target width (default 640),
  * optional frame skipping,
  * motion-gated person detection (background subtraction),
  * optional low-frequency HOG person detector (more stable than motion-only).

IMPORTANT:
This is still a DIP demo, not a production-grade "robbery detector".
It estimates "Robbery Behavior Potential" (RBP) using simple cues:
- loitering near high-value area (HV ROI)
- then moving quickly toward exit (EXIT ROI)
- plus elevated motion energy (agitation) and/or group near exit
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------- Small utilities ---------------------------------

def parse_source(src: str) -> Union[int, str]:
    # "0" -> webcam 0
    if src.isdigit() and len(src) < 4:
        return int(src)
    return src


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def norm_box_to_px(box_norm: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_norm
    x1 = int(clamp01(x1) * w)
    y1 = int(clamp01(y1) * h)
    x2 = int(clamp01(x2) * w)
    y2 = int(clamp01(y2) * h)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_xyxy(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_th: float = 0.4) -> List[int]:
    """Return indices kept after NMS."""
    if not boxes:
        return []
    idxs = np.argsort(np.array(scores, dtype=np.float32))[::-1]
    keep: List[int] = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        ious = np.array([box_iou(boxes[i], boxes[int(j)]) for j in rest], dtype=np.float32)
        idxs = rest[ious < iou_th]
    return keep


def draw_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------ Tracking --------------------------------------

@dataclass
class Track:
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    last_seen: int
    miss: int
    history: Deque[Tuple[float, float]]
    hv_frames: int = 0
    seen_hv: bool = False
    entered_exit_after_hv: bool = False
    rbp: float = 0.0
    last_alert_frame: int = -10_000
    sustain: int = 0  # consecutive frames over risk threshold


class CentroidTracker:
    """Simple centroid tracker (fast)."""

    def __init__(self, max_distance: float = 60.0, max_missing: int = 25):
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Tuple[int, int, int, int]], frame_idx: int) -> Dict[int, Track]:
        # Prepare det centroids
        det_centroids = []
        for (x1, y1, x2, y2) in detections:
            det_centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        if not self.tracks:
            for bbox, c in zip(detections, det_centroids):
                self._new_track(bbox, c, frame_idx)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids], dtype=np.float32)
        det_centroids_arr = np.array(det_centroids, dtype=np.float32) if det_centroids else np.zeros((0, 2), dtype=np.float32)

        assigned_tracks = set()
        assigned_dets = set()

        if det_centroids:
            # Compute distance matrix
            dists = np.linalg.norm(track_centroids[:, None, :] - det_centroids_arr[None, :, :], axis=2)
            # Greedy match by nearest pairs
            while True:
                i = int(np.argmin(dists)) if dists.size else -1
                if i < 0:
                    break
                ti, di = np.unravel_index(i, dists.shape)
                if np.isinf(dists[ti, di]):
                    break
                if dists[ti, di] <= self.max_distance and (track_ids[ti] not in assigned_tracks) and (di not in assigned_dets):
                    tid = track_ids[ti]
                    self._update_track(tid, detections[di], det_centroids[di], frame_idx)
                    assigned_tracks.add(tid)
                    assigned_dets.add(di)
                # block this row/col
                dists[ti, :] = np.inf
                dists[:, di] = np.inf

        # Mark unassigned tracks as missing
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                tr = self.tracks[tid]
                tr.miss += 1
                if tr.miss > self.max_missing:
                    del self.tracks[tid]

        # Create new tracks for unassigned detections
        for di, bbox in enumerate(detections):
            if di not in assigned_dets:
                self._new_track(bbox, det_centroids[di], frame_idx)

        return self.tracks

    def _new_track(self, bbox, centroid, frame_idx: int):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = Track(
            bbox=bbox,
            centroid=centroid,
            last_seen=frame_idx,
            miss=0,
            history=deque([centroid], maxlen=25),
        )

    def _update_track(self, tid: int, bbox, centroid, frame_idx: int):
        tr = self.tracks[tid]
        tr.bbox = bbox
        tr.centroid = centroid
        tr.last_seen = frame_idx
        tr.miss = 0
        tr.history.append(centroid)


# --------------------------- Person detection ---------------------------------

class MotionPersonDetector:
    """
    Fast motion-based person candidate detector (background subtraction).
    Outputs person-like moving blobs.
    """

    def __init__(self, history: int = 200, var_threshold: float = 16.0, detect_shadows: bool = False):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame_bgr: np.ndarray, min_area: int = 600) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fg = self.bg.apply(gray)

        # binary
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        fg = cv2.dilate(fg, self.kernel_dilate, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            ar = h / float(w + 1e-6)

            # person-like moving blob: tall-ish, not too wide
            # these are loose to avoid missing, later filtered by tracking+risk rules
            if ar > 1.15 and h > 35 and w > 12:
                dets.append((x, y, x + w, y + h))

        return dets, fg


class HogPersonDetector:
    """
    Optional: HOG people detector (slow if run every frame).
    We'll run it every N frames only.
    """

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Work in smaller image for speed
        # detectMultiScale wants BGR/gray; it internally converts
        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        for (x, y, w, h), s in zip(rects, weights):
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
            scores.append(float(s))
        keep = nms_xyxy(boxes, scores, iou_th=0.45)
        return [boxes[i] for i in keep]


def merge_boxes(primary: List[Tuple[int, int, int, int]],
                secondary: List[Tuple[int, int, int, int]],
                iou_match: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """
    Merge detections: keep primary, add secondary that don't overlap enough.
    """
    out = list(primary)
    for b in secondary:
        if all(box_iou(b, p) < iou_match for p in out):
            out.append(b)
    return out


# --------------------------- Robbery risk logic -------------------------------

class RobberyRiskScorer:
    """
    Computes per-track RBP score using:
      - loitering in HV ROI
      - then entering EXIT ROI after HV
      - quick movement toward exit after HV
      - high motion energy inside bbox ("agitation")
      - group near exit
    """

    def __init__(
        self,
        fps: float,
        hv_roi_norm=(0.10, 0.20, 0.70, 0.85),
        exit_roi_norm=(0.75, 0.55, 0.98, 0.98),
        loiter_time_sec: float = 3.0,
        speed_to_exit_px_per_sec: float = 240.0,
    ):
        self.fps = max(1.0, float(fps))
        self.hv_roi_norm = hv_roi_norm
        self.exit_roi_norm = exit_roi_norm
        self.loiter_frames = int(loiter_time_sec * self.fps)
        self.speed_to_exit = float(speed_to_exit_px_per_sec)

    @staticmethod
    def _in_box(pt: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
        x, y = pt
        x1, y1, x2, y2 = box
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def update(
        self,
        tracks: Dict[int, Track],
        frame_shape: Tuple[int, int, int],
        fg_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], List[Tuple[int, float]]]:
        h, w = frame_shape[:2]
        hv_box = norm_box_to_px(self.hv_roi_norm, w, h)
        exit_box = norm_box_to_px(self.exit_roi_norm, w, h)

        # Count group near exit (persons in exit ROI)
        exit_ids = [tid for tid, tr in tracks.items() if self._in_box(tr.centroid, exit_box)]
        group_score = 1.0 if len(exit_ids) >= 2 else 0.0

        suspicious: List[Tuple[int, float]] = []

        for tid, tr in tracks.items():
            # update HV/EXIT states
            in_hv = self._in_box(tr.centroid, hv_box)
            in_exit = self._in_box(tr.centroid, exit_box)

            if in_hv:
                tr.hv_frames += 1
                if tr.hv_frames >= 1:
                    tr.seen_hv = True

            if tr.seen_hv and in_exit:
                tr.entered_exit_after_hv = True

            # speed (px/sec) from last 2 history points
            speed = 0.0
            if len(tr.history) >= 2:
                (x0, y0), (x1, y1) = tr.history[-2], tr.history[-1]
                dist = float(np.hypot(x1 - x0, y1 - y0))
                speed = dist * self.fps

            loiter_score = 1.0 if tr.hv_frames >= self.loiter_frames else 0.0
            exit_score = 1.0 if tr.entered_exit_after_hv else 0.0
            speed_score = 1.0 if (tr.seen_hv and speed >= self.speed_to_exit) else 0.0

            motion_energy = 0.0
            if fg_mask is not None:
                x1, y1, x2, y2 = tr.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    roi = fg_mask[y1:y2, x1:x2]
                    # fraction of FG pixels
                    motion_energy = float((roi > 0).mean())
            agitation_score = 1.0 if motion_energy >= 0.12 else 0.0

            # Weighted RBP (0..1-ish)
            rbp = (
                0.35 * loiter_score +
                0.25 * exit_score +
                0.20 * speed_score +
                0.10 * agitation_score +
                0.10 * group_score
            )

            tr.rbp = float(rbp)

            if rbp >= 0.55:
                suspicious.append((tid, tr.rbp))

        return hv_box, exit_box, suspicious


# ------------------------------- Main demo ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionGuard Robbery Demo (fast, OpenCV-only)")
    p.add_argument("--source", default="0", help="Video path or camera index (e.g., 0)")
    p.add_argument("--width", type=int, default=640, help="Processing width (smaller = faster)")
    p.add_argument("--skip", type=int, default=1, help="Process every Nth frame (>=1). Display still updates.")
    p.add_argument("--enable-hog", action="store_true", help="Enable HOG person detection (more stable, slower)")
    p.add_argument("--hog-interval", type=int, default=8, help="Run HOG every N processed frames (>=1)")
    p.add_argument("--min-area", type=int, default=600, help="Min moving blob area")
    p.add_argument("--max-miss", type=int, default=25, help="Tracker max missing frames")
    p.add_argument("--max-dist", type=float, default=60.0, help="Tracker max centroid distance")

    p.add_argument("--hv-roi", default="0.10,0.20,0.70,0.85", help="HV ROI norm x1,y1,x2,y2")
    p.add_argument("--exit-roi", default="0.75,0.55,0.98,0.98", help="Exit ROI norm x1,y1,x2,y2")

    p.add_argument("--loiter-sec", type=float, default=3.0, help="Loiter time in HV (seconds)")
    p.add_argument("--speed-exit", type=float, default=240.0, help="Speed threshold (px/sec) after HV")
    p.add_argument("--alert-thres", type=float, default=0.65, help="Alert threshold (RBP)")
    p.add_argument("--sustain", type=int, default=8, help="Frames above threshold before alert")
    p.add_argument("--cooldown", type=int, default=60, help="Cooldown frames between alerts for same track")

    p.add_argument("--save", action="store_true", help="Save annotated output video")
    p.add_argument("--output", default="robbery_out.mp4", help="Output video path when --save is set")
    p.add_argument("--no-display", action="store_true", help="Disable imshow for max speed")
    p.add_argument("--snapshot-dir", default="robbery_alerts", help="Where to save alert snapshots/logs")
    return p.parse_args()


def parse_roi(s: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 4 comma-separated floats: x1,y1,x2,y2")
    return tuple(parts)  # type: ignore


def main() -> int:
    args = parse_args()
    src = parse_source(args.source)

    cv2.setUseOptimized(True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0  # fallback

    # Output writer (lazy init on first frame)
    writer = None

    # Prepare detectors
    motion_det = MotionPersonDetector(history=200, var_threshold=16.0, detect_shadows=False)
    hog_det = HogPersonDetector() if args.enable_hog else None

    tracker = CentroidTracker(max_distance=args.max_dist, max_missing=args.max_miss)
    scorer = RobberyRiskScorer(
        fps=fps,
        hv_roi_norm=parse_roi(args.hv_roi),
        exit_roi_norm=parse_roi(args.exit_roi),
        loiter_time_sec=args.loiter_sec,
        speed_to_exit_px_per_sec=args.speed_exit,
    )

    # Logs
    snap_dir = Path(args.snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    csv_path = snap_dir / "alerts.csv"
    new_csv = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(["timestamp", "frame", "track_id", "rbp", "note", "snapshot"])

    frame_idx = 0
    proc_idx = 0
    last_time = time.time()
    fps_smooth = 0.0

    # Cached detections for skip frames
    last_dets: List[Tuple[int, int, int, int]] = []
    last_fg = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Resize for speed
            h0, w0 = frame.shape[:2]
            if args.width > 0 and w0 > args.width:
                scale = args.width / float(w0)
                frame_small = cv2.resize(frame, (args.width, int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                frame_small = frame
                scale = 1.0

            # Process every Nth frame
            do_process = (frame_idx % max(1, args.skip) == 0)

            if do_process:
                proc_idx += 1
                dets_motion, fg = motion_det.detect(frame_small, min_area=args.min_area)
                last_fg = fg

                dets = dets_motion

                # Optional: HOG at low frequency to add stable person boxes
                if hog_det is not None and (proc_idx % max(1, args.hog_interval) == 0):
                    dets_hog = hog_det.detect(frame_small)
                    dets = merge_boxes(dets, dets_hog, iou_match=0.20)

                last_dets = dets

            # Track update on every frame using latest detections
            tracks = tracker.update(last_dets, frame_idx)

            # Score risk (use latest fg_mask from processed frame if available)
            hv_box, exit_box, suspicious = scorer.update(tracks, frame_small.shape, fg_mask=last_fg)

            # Compose annotated frame (small)
            annotated = frame_small.copy()

            # Draw ROIs
            cv2.rectangle(annotated, (hv_box[0], hv_box[1]), (hv_box[2], hv_box[3]), (255, 255, 0), 2)
            cv2.rectangle(annotated, (exit_box[0], exit_box[1]), (exit_box[2], exit_box[3]), (0, 255, 255), 2)
            draw_label(annotated, "HV", hv_box[0] + 4, max(12, hv_box[1] - 6))
            draw_label(annotated, "EXIT", exit_box[0] + 4, max(12, exit_box[1] - 6))

            # Draw tracks and alerts
            alert_any = False
            for tid, tr in tracks.items():
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                draw_label(annotated, f"ID {tid} RBP {tr.rbp:.2f}", x1, max(18, y1 - 8))

                # Alert logic: sustained risk and cooldown
                if tr.rbp >= args.alert_thres:
                    tr.sustain += 1
                else:
                    tr.sustain = 0

                if tr.sustain >= args.sustain and (frame_idx - tr.last_alert_frame) >= args.cooldown:
                    tr.last_alert_frame = frame_idx
                    tr.sustain = 0
                    alert_any = True

                    # snapshot
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"alert_{ts}_f{frame_idx}_id{tid}.jpg"
                    snap_path = snap_dir / snap_name
                    cv2.imwrite(str(snap_path), annotated)

                    csv_w.writerow([ts, frame_idx, tid, f"{tr.rbp:.3f}", "RBP alert", snap_name])
                    csv_f.flush()

            if alert_any:
                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0, 0, 255), -1)
                draw_label(annotated, "ALERT: SUSPICIOUS ROBBERY BEHAVIOR", 10, 26)

            # FPS calc
            now = time.time()
            dt = now - last_time
            last_time = now
            inst_fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
            draw_label(annotated, f"FPS {fps_smooth:.1f}  dets {len(last_dets)}  tracks {len(tracks)}", 10, annotated.shape[0] - 10)

            # Save
            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, float(fps), (annotated.shape[1], annotated.shape[0]))
                writer.write(annotated)

            # Display
            if not args.no_display:
                cv2.imshow("VisionGuard - Robbery (FAST)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1

    finally:
        csv_f.close()
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"[DONE] Alerts CSV saved to: {csv_path}")
    print(f"[DONE] Snapshots saved in: {snap_dir.resolve()}")
    if args.save:
        print(f"[DONE] Output video: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
