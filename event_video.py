"""
Event Camera Simulator — Video File Input | Python + OpenCV
============================================================
Feed any video file via a file-picker dialog (or CLI arg) and
visualise it as if captured by a Dynamic Vision Sensor (DVS).

Keypoint / 3-D reconstruction pipeline
---------------------------------------
• Harris corners detected on each event frame.
• Lucas–Kanade optical flow tracks keypoints across frames.
• Essential Matrix + pose recovery + triangulation builds a 3-D cloud.
• MR-NDSEF spatiotemporal filter strips raw event noise across multiple radii.
• DBSCAN isolates candidate target clusters from background noise.
• Tracking and 3-D pose estimation are fed strictly from cleaned target clusters.
• Press 'e' to export a PLY point cloud snapshot at any time.

Usage:
    python3 event_video.py               <- opens file-picker dialog
    python3 event_video.py video.mp4     <- direct path

Controls (press key while the window is focused):
    q / ESC  -- quit
    SPACE    -- pause / resume
    r        -- reset reference frame at current position
    [ / ]    -- decrease / increase threshold C
    f        -- toggle fullscreen event-only view
    s        -- save current event frame as PNG
    n        -- toggle MR-NDSEF noise filter
    d        -- toggle DBSCAN candidate target clustering
    b / o    -- toggle object bounding boxes & post-event tracking
    k        -- toggle keypoint overlay on event frame
    e        -- export 3-D point cloud as PLY
    c        -- clear accumulated 3-D cloud
"""

import cv2
import numpy as np
import sys
import os
import time

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# -- try to import tkinter for the file-picker -------------------------------- #
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ----------------------------- configuration --------------------------------- #
THRESHOLD      = 0.2
THRESHOLD_STEP = 0.01
NOISE_FLOOR    = 1e-6
BLUR_KSIZE     = (3, 3)
DECAY          = 0.92
FONT           = cv2.FONT_HERSHEY_SIMPLEX

VIDEO_EXTS = (
    ".mp4", ".avi", ".mov", ".mkv", ".flv",
    ".wmv", ".webm", ".m4v", ".ts", ".mts",
)

# BGR colour palette
COL_ON      = (220, 240,  10)
COL_OFF     = ( 30,  40, 240)
COL_BG      = (  5,   5,  10)
COL_TEXT    = (200, 200, 200)
COL_TITLE   = ( 80, 220, 255)
COL_THRESH  = ( 50, 255, 150)
COL_PAUSE   = ( 40, 200, 255)
COL_BAR_ON  = (200, 220,  10)
COL_BAR_OFF = ( 20,  30, 220)
COL_KP      = (  0, 255, 200)   # keypoint ring colour
COL_3D_BG   = (  8,   8,  18)

# -- MR-NDSEF & DBSCAN Filter Parameters -------------------------------------- #
MR_NDSEF_RADII      = (3, 5, 8)       # multiple neighborhood radii (pixels)
MR_NDSEF_THRESHOLDS = (0.16, 0.10, 0.06) # density thresholds per radius
DBSCAN_EPS          = 14.0            # max neighbor distance (px) for DBSCAN
DBSCAN_MIN_PTS      = 16              # min core points to form a target cluster
DBSCAN_MAX_PTS      = 2500            # max points fed to DBSCAN (subsampled if exceeded)

# -- Object Segregation & Post-Event Tracking --------------------------------- #
MIN_OBJ_AREA      = 100      # minimum bounding area (px) to consider an object
MAX_OBJ_AREA_FRAC = 0.80     # maximum fraction of frame (filter whole-screen flash)
OBJ_POST_FRAMES   = 10       # frames an object box persists post-event change
OBJ_MATCH_DIST    = 80.0     # max centroid distance (px) between frames to match

TRACK_COLORS = [
    (  0, 255, 255),  # Yellow
    (255, 160,  30),  # Bright Sky Blue
    (  0, 255, 130),  # Mint Green
    (255,  80, 255),  # Magenta
    (  0, 180, 255),  # Orange
    (255, 215,   0),  # Gold
    (190, 110, 255),  # Violet
    ( 60, 240, 240),  # Cyan
]

# -- SfM parameters ----------------------------------------------------------- #
HARRIS_BLOCK   = 5        # neighbourhood size
HARRIS_KSIZE   = 3        # Sobel kernel size
HARRIS_K       = 0.04     # Harris free parameter
HARRIS_THRESH  = 0.01     # fraction of max response
HARRIS_MAX_KP  = 300      # max keypoints to keep

LK_WIN        = (15, 15)
LK_MAX_LEVEL  = 2
LK_CRITERIA   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

SFM_INTERVAL   = 5        # triangulate every N frames
SFM_MIN_PAIRS  = 8        # min matched pairs for essential matrix
MAX_CLOUD_PTS  = 8000     # cap to keep rendering fast


# ------------------------------ helpers -------------------------------------- #

def pick_video_file() -> str:
    # On macOS, use native AppleScript dialog to avoid Cocoa/Tcl_FindHashEntry crash with OpenCV
    if sys.platform == "darwin":
        try:
            import subprocess
            cmd = """osascript -e 'POSIX path of (choose file with prompt "Select a video file")'"""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            chosen = res.stdout.strip()
            if chosen:
                return chosen
        except Exception:
            pass

    if not HAS_TK:
        print("[ERROR] tkinter not available. Pass the video path as an argument:")
        print("        python3 event_video.py <path/to/video>")
        sys.exit(1)
    root = tk.Tk()
    root.withdraw()
    root.call("wm", "attributes", ".", "-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", " ".join(f"*{e}" for e in VIDEO_EXTS)),
            ("All files",   "*.*"),
        ],
    )
    # Avoid root.destroy() on macOS as it tears down Tcl hash tables while OpenCV GUI runs
    return path


def log_lum(gray: np.ndarray) -> np.ndarray:
    return np.log(gray.astype(np.float32) + NOISE_FLOOR)


def draw_label(img, text, pos, color=COL_TEXT, scale=0.55, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     thickness,     cv2.LINE_AA)


def panel_header(panel, title):
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w, 28), (20, 20, 30), -1)
    draw_label(panel, title, (8, 20), COL_TITLE, scale=0.60)


def make_heatmap(activity: np.ndarray) -> np.ndarray:
    normed = cv2.normalize(activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normed, cv2.COLORMAP_INFERNO)


def event_frame_bgr(on_mask, off_mask, h, w) -> np.ndarray:
    frame = np.full((h, w, 3), COL_BG, dtype=np.uint8)
    frame[on_mask]  = COL_ON
    frame[off_mask] = COL_OFF
    return frame


def overlay_stats(panel, on_cnt, off_cnt, clean_cnt, threshold,
                  frame_no, total_frames, fps_vid, fps_proc, paused, h, w,
                  filter_active=True, n_targets=0):
    progress = frame_no / max(total_frames, 1)
    bar_x = int((w - 16) * progress)
    cv2.rectangle(panel, (8, 36), (w - 8, 44), (40, 40, 50), -1)
    cv2.rectangle(panel, (8, 36), (8 + bar_x, 44), (80, 180, 255), -1)
    draw_label(panel, f"Frame {frame_no}/{total_frames}   "
                      f"src:{fps_vid:.0f}fps  proc:{fps_proc:.1f}fps",
               (8, 58), COL_TEXT, scale=0.42)

    bar_max  = max(on_cnt + off_cnt, 1)
    bar_w    = w - 16
    on_frac  = int(bar_w * on_cnt  / bar_max)
    off_frac = int(bar_w * off_cnt / bar_max)
    y = h - 68
    cv2.rectangle(panel, (8, y),     (8 + on_frac,  y + 8),  COL_BAR_ON,  -1)
    cv2.rectangle(panel, (8, y + 12),(8 + off_frac, y + 20), COL_BAR_OFF, -1)
    draw_label(panel, f"ON  events : {on_cnt:6d}",  (8, y - 4),  COL_ON,  scale=0.43)
    draw_label(panel, f"OFF events : {off_cnt:6d}", (8, y + 34), COL_OFF, scale=0.43)

    raw_total = max(on_cnt + off_cnt, 1)
    rejected_pct = max(0.0, 100.0 * (1.0 - clean_cnt / raw_total)) if filter_active else 0.0
    status_filter = f"MR-NDSEF+DBSCAN: ON ({rejected_pct:.0f}% noise cut, {n_targets} tgts)" if filter_active else "Filter: OFF (raw events)"
    draw_label(panel, status_filter, (8, h - 26), (80, 220, 255) if filter_active else (150, 150, 150), scale=0.40)
    draw_label(panel, f"Threshold C = {threshold:.3f}  ([ ] to change)",
               (8, h - 10), COL_THRESH, scale=0.42)

    if paused:
        draw_label(panel, "PAUSED", (w // 2 - 38, h // 2),
                   COL_PAUSE, scale=1.0, thickness=2)


def save_frame(ev_frame, frame_no, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"event_frame_{frame_no:06d}.png")
    cv2.imwrite(fname, ev_frame)
    print(f"[INFO] Saved -> {fname}")


# ═════════════════════ 1. MR-NDSEF Filter ════════════════════════════════════

class MR_NDSEF_Filter:
    """
    Multi-Neighborhood-Radius Spatiotemporal Event Filter (MR-NDSEF).
    Evaluates neighborhood event density across multiple radii using circular
    sliding windows. Suppresses isolated background noise while preserving
    valid high-dynamic target trajectories.
    """
    def __init__(self,
                 radii: tuple = MR_NDSEF_RADII,
                 thresholds: tuple = MR_NDSEF_THRESHOLDS,
                 use_spatiotemporal: bool = True):
        self.radii = radii
        self.thresholds = thresholds
        self.use_spatiotemporal = use_spatiotemporal
        self.kernels = []
        for r in self.radii:
            ksize = 2 * r + 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)).astype(np.float32)
            self.kernels.append(k / float(np.count_nonzero(k)))

    def filter(self, on_mask: np.ndarray, off_mask: np.ndarray, activity: np.ndarray = None):
        raw_events = on_mask | off_mask
        if not np.any(raw_events):
            return on_mask, off_mask, raw_events

        raw_f = raw_events.astype(np.float32)
        valid = np.zeros_like(raw_events, dtype=bool)

        for k, th in zip(self.kernels, self.thresholds):
            density = cv2.filter2D(raw_f, -1, k)
            valid |= (raw_events & (density >= th))

        if self.use_spatiotemporal and activity is not None:
            # Persistent activity trail protection
            valid |= (raw_events & (activity > 1.8))

        clean_on = on_mask & valid
        clean_off = off_mask & valid
        clean_events = clean_on | clean_off
        return clean_on, clean_off, clean_events


# ═════════════════════ 2. DBSCAN Clustering ══════════════════════════════════

def dbscan_cluster_events(clean_events: np.ndarray,
                          eps: float = DBSCAN_EPS,
                          min_pts: int = DBSCAN_MIN_PTS,
                          max_pts: int = DBSCAN_MAX_PTS) -> tuple[list, np.ndarray]:
    """
    Run DBSCAN on cleaned event coordinates to extract candidate target clusters.
    Returns:
        clusters: list of cluster dicts:
            {'id': int, 'box': (x,y,w,h), 'centroid': (cx,cy), 'size': int, 'pts': np.ndarray}
        target_mask: binary uint8 mask containing only the candidate target regions.
    """
    h, w = clean_events.shape[:2]
    target_mask = np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(clean_events)
    n_pts = len(xs)
    if n_pts < min_pts:
        return [], target_mask

    all_pts = np.column_stack([xs, ys]).astype(np.float32)

    # Subsample if event count is extremely high to maintain 30+ FPS
    if n_pts > max_pts:
        step = int(np.ceil(n_pts / max_pts))
        pts_db = all_pts[::step]
    else:
        pts_db = all_pts

    labels = None
    if HAS_SKLEARN:
        try:
            db = DBSCAN(eps=eps, min_samples=min_pts)
            labels = db.fit_predict(pts_db)
        except Exception:
            labels = None

    if labels is None:
        # High-performance morphological connected-component fallback
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(eps), int(eps)))
        dil = cv2.dilate(clean_events.astype(np.uint8) * 255, kernel, iterations=1)
        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(dil)
        clusters = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_pts * 4:
                continue
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            c_mask = (cc_labels == i).astype(np.uint8) * 255
            target_mask |= c_mask
            clusters.append({
                'id': i - 1,
                'box': (bx, by, bw, bh),
                'centroid': (cx, cy),
                'size': int(area),
                'pts': all_pts,
            })
        return clusters, target_mask

    clusters = []
    unique_labels = [lbl for lbl in np.unique(labels) if lbl >= 0]

    for cid in unique_labels:
        c_pts = pts_db[labels == cid]
        if len(c_pts) < min_pts:
            continue

        x_min, y_min = np.min(c_pts, axis=0)
        x_max, y_max = np.max(c_pts, axis=0)
        bx = int(x_min)
        by = int(y_min)
        bw = max(4, int(x_max - x_min + 1))
        bh = max(4, int(y_max - y_min + 1))
        cx = float(np.mean(c_pts[:, 0]))
        cy = float(np.mean(c_pts[:, 1]))

        # Paint target mask for this cluster with margin
        margin = int(eps // 2)
        rx1 = max(0, bx - margin)
        ry1 = max(0, by - margin)
        rx2 = min(w, bx + bw + margin)
        ry2 = min(h, by + bh + margin)
        target_mask[ry1:ry2, rx1:rx2] = 255

        clusters.append({
            'id': int(cid),
            'box': (bx, by, bw, bh),
            'centroid': (cx, cy),
            'size': len(c_pts),
            'pts': c_pts,
        })

    # Sort largest clusters first
    clusters.sort(key=lambda c: c['size'], reverse=True)
    return clusters, target_mask


# ═════════════════════ 3. Tracking Module ════════════════════════════════════

class EventObjectTracker:
    """
    Maintains persistent object tracks directly from DBSCAN-isolated candidate
    target clusters with post-event persistence.
    """
    def __init__(self,
                 max_age: int = OBJ_POST_FRAMES,
                 match_dist: float = OBJ_MATCH_DIST):
        self.max_age = max_age
        self.match_dist = match_dist
        self.next_id = 1
        self.tracks = []  # list of track dicts

    def reset(self):
        self.tracks.clear()
        self.next_id = 1

    def update_from_clusters(self, clusters: list) -> list:
        """
        Takes candidate target clusters produced by DBSCAN and associates
        them with temporal tracks.
        """
        detections = []
        for c in clusters:
            bw, bh = c['box'][2], c['box'][3]
            detections.append({
                'box': c['box'],
                'centroid': c['centroid'],
                'area': float(bw * bh),
                'events': c['size'],
            })

        matched_det = set()
        for trk in self.tracks:
            best_dist = float('inf')
            best_d_idx = -1
            tcx, tcy = trk['centroid']

            for d_idx, det in enumerate(detections):
                if d_idx in matched_det:
                    continue
                dcx, dcy = det['centroid']
                dist = np.hypot(tcx - dcx, tcy - dcy)
                if dist < best_dist and dist < self.match_dist:
                    best_dist = dist
                    best_d_idx = d_idx

            if best_d_idx != -1:
                det = detections[best_d_idx]
                ox, oy, obw, obh = trk['box']
                nx, ny, nbw, nbh = det['box']
                smooth_box = (
                    int(0.25 * ox + 0.75 * nx),
                    int(0.25 * oy + 0.75 * ny),
                    int(0.25 * obw + 0.75 * nbw),
                    int(0.25 * obh + 0.75 * nbh),
                )
                trk['box'] = smooth_box
                trk['centroid'] = (smooth_box[0] + smooth_box[2] / 2.0,
                                  smooth_box[1] + smooth_box[3] / 2.0)
                trk['area'] = det['area']
                trk['events'] = det['events']
                trk['age'] = 0          # 0 = active cluster
                trk['active'] = True
                matched_det.add(best_d_idx)
            else:
                # Post-event persistence
                trk['age'] += 1
                trk['active'] = False

        # Spawn new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det:
                color = TRACK_COLORS[(self.next_id - 1) % len(TRACK_COLORS)]
                self.tracks.append({
                    'id': self.next_id,
                    'box': det['box'],
                    'centroid': det['centroid'],
                    'area': det['area'],
                    'events': det['events'],
                    'age': 0,
                    'active': True,
                    'color': color,
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]
        return self.tracks


def draw_object_box(img: np.ndarray, trk: dict):
    """Render a bounding box with sleek corner accents and status label."""
    x, y, w, h = trk['box']
    color = trk['color']
    is_active = trk['active']
    age = trk['age']
    tid = trk['id']
    events = trk.get('events', 0)

    # Box styling: solid bright if active, faded if post-event
    if is_active:
        box_col = color
        line_thick = 2
        corner_len = max(8, min(16, w // 4, h // 4))
        status_tag = f"Target #{tid} ({events} ev)"
    else:
        # Faded color for post-event persistence
        box_col = tuple(int(c * max(0.4, 1.0 - (age * 0.07))) for c in color)
        line_thick = 1
        corner_len = max(6, min(12, w // 4, h // 4))
        status_tag = f"Target #{tid} [POST]"

    # Main rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), box_col, 1)

    # Corner brackets for tech aesthetic
    cl = corner_len
    cv2.line(img, (x, y), (x + cl, y), box_col, line_thick)
    cv2.line(img, (x, y), (x, y + cl), box_col, line_thick)
    cv2.line(img, (x + w, y), (x + w - cl, y), box_col, line_thick)
    cv2.line(img, (x + w, y), (x + w, y + cl), box_col, line_thick)
    cv2.line(img, (x, y + h), (x + cl, y + h), box_col, line_thick)
    cv2.line(img, (x, y + h), (x, y + h - cl), box_col, line_thick)
    cv2.line(img, (x + w, y + h), (x + w - cl, y + h), box_col, line_thick)
    cv2.line(img, (x + w, y + h), (x + w, y + h - cl), box_col, line_thick)

    # Centroid marker
    cx, cy = int(trk['centroid'][0]), int(trk['centroid'][1])
    cv2.drawMarker(img, (cx, cy), box_col, cv2.MARKER_CROSS, 6, 1)

    # Label badge above or inside box
    label_size, _ = cv2.getTextSize(status_tag, FONT, 0.40, 1)
    lbl_w, lbl_h = label_size
    lbl_y1 = max(0, y - lbl_h - 4)
    lbl_y2 = lbl_y1 + lbl_h + 4
    lbl_x2 = min(img.shape[1], x + lbl_w + 8)

    cv2.rectangle(img, (x, lbl_y1), (lbl_x2, lbl_y2), (15, 15, 20), -1)
    cv2.rectangle(img, (x, lbl_y1), (lbl_x2, lbl_y2), box_col, 1)
    cv2.putText(img, status_tag, (x + 4, lbl_y2 - 3), FONT, 0.40, box_col, 1, cv2.LINE_AA)


# -------------------------- keypoint helpers --------------------------------- #

def detect_harris_keypoints(gray_event: np.ndarray, target_mask: np.ndarray = None):
    """
    Harris corner detection restricted strictly to cleaned target cluster regions.
    Returns (N,1,2) float32 array suitable for cv2.calcOpticalFlowPyrLK,
    or None if fewer than SFM_MIN_PAIRS corners found.
    """
    if target_mask is not None and np.any(target_mask):
        gray_event = cv2.bitwise_and(gray_event, gray_event, mask=target_mask)

    resp = cv2.cornerHarris(gray_event, HARRIS_BLOCK, HARRIS_KSIZE, HARRIS_K)
    resp = cv2.dilate(resp, None)

    thresh = HARRIS_THRESH * resp.max()
    ys, xs = np.where(resp > thresh)
    if len(xs) == 0:
        return None

    strengths = resp[ys, xs]
    order = np.argsort(-strengths)[:HARRIS_MAX_KP]
    pts = np.column_stack([xs[order], ys[order]]).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001)
    pts_refined = cv2.cornerSubPix(
        gray_event, pts.reshape(-1, 1, 2), (5, 5), (-1, -1), criteria
    )
    return pts_refined  # (N,1,2)


def track_keypoints(prev_gray, curr_gray, prev_pts):
    """LK optical flow. Returns (good_prev, good_curr) or (None, None)."""
    if prev_pts is None or len(prev_pts) < SFM_MIN_PAIRS:
        return None, None

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts,
        None, winSize=LK_WIN, maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )
    if curr_pts is None:
        return None, None

    mask = status.ravel() == 1
    return prev_pts[mask], curr_pts[mask]


# ----------------------------- SfM pipeline ---------------------------------- #

class SfMPipeline:
    """
    Incremental Structure from Motion on event-camera Harris keypoints.
    Accumulates a global 3-D point cloud rendered live in a 4th panel.
    """

    def __init__(self, h: int, w: int):
        fx = fy = max(h, w) * 1.2
        self.K = np.array([
            [fx,  0, w / 2],
            [ 0, fy, h / 2],
            [ 0,  0,     1],
        ], dtype=np.float64)

        self.cloud_pts  : np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.cloud_cols : np.ndarray = np.empty((0, 3), dtype=np.uint8)

        self.R_global = np.eye(3, dtype=np.float64)
        self.t_global = np.zeros((3, 1), dtype=np.float64)

        self.n_triangulations = 0
        self.last_added       = 0

    def try_triangulate(self, pts0, pts1, color_hint=(80, 220, 255)) -> int:
        """
        Estimate Essential Matrix, recover pose, triangulate new 3-D points.
        Returns number of new points added.
        """
        p0 = pts0.reshape(-1, 2).astype(np.float64)
        p1 = pts1.reshape(-1, 2).astype(np.float64)

        if len(p0) < SFM_MIN_PAIRS:
            return 0

        try:
            E, mask_e = cv2.findEssentialMat(
                p0, p1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if E is None or mask_e is None:
                return 0

            inlier_mask = mask_e.ravel().astype(bool)
            if inlier_mask.sum() < SFM_MIN_PAIRS:
                return 0

            _, R, t, _ = cv2.recoverPose(
                E, p0[inlier_mask], p1[inlier_mask], self.K
            )

            P0 = self.K @ np.hstack([self.R_global, self.t_global])
            R_new = R @ self.R_global
            t_new = R @ self.t_global + t
            P1 = self.K @ np.hstack([R_new, t_new])

            pts4d = cv2.triangulatePoints(
                P0, P1, p0[inlier_mask].T, p1[inlier_mask].T
            )

            w_coords = pts4d[3]
            valid = np.abs(w_coords) > 1e-8
            pts3d = (pts4d[:3, valid] / w_coords[valid]).T

            depths = pts3d[:, 2]
            depth_mask = (depths > 0.01) & (depths < 200.0)
            pts3d = pts3d[depth_mask]

            if len(pts3d) == 0:
                return 0

            cols = np.tile(
                np.array(color_hint, dtype=np.uint8), (len(pts3d), 1)
            )

            self.cloud_pts  = np.vstack([self.cloud_pts,  pts3d])
            self.cloud_cols = np.vstack([self.cloud_cols, cols])
            if len(self.cloud_pts) > MAX_CLOUD_PTS:
                keep = len(self.cloud_pts) - MAX_CLOUD_PTS
                self.cloud_pts  = self.cloud_pts[keep:]
                self.cloud_cols = self.cloud_cols[keep:]

            self.R_global = R_new
            self.t_global = t_new
            self.n_triangulations += 1
            self.last_added = len(pts3d)
            return len(pts3d)

        except cv2.error:
            return 0

    def clear(self):
        self.cloud_pts  = np.empty((0, 3), dtype=np.float64)
        self.cloud_cols = np.empty((0, 3), dtype=np.uint8)
        self.R_global   = np.eye(3, dtype=np.float64)
        self.t_global   = np.zeros((3, 1), dtype=np.float64)
        self.n_triangulations = 0
        print("[INFO] 3-D cloud cleared.")

    def export_ply(self, path: str):
        n = len(self.cloud_pts)
        if n == 0:
            print("[WARN] Cloud is empty -- nothing to export.")
            return
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar blue\nproperty uchar green\nproperty uchar red\n")
            f.write("end_header\n")
            for pt, col in zip(self.cloud_pts, self.cloud_cols):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                        f"{col[0]} {col[1]} {col[2]}\n")
        print(f"[INFO] Exported {n} points -> {path}")


# --------------------------------- main -------------------------------------- #

def main():
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        print("[INFO] No video path given -- opening file picker ...")
        video_path = pick_video_file()
        if not video_path:
            print("[INFO] No file selected. Exiting.")
            sys.exit(0)

    if not os.path.isfile(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay  = max(1, int(1000 / fps_vid))
    out_dir      = os.path.splitext(video_path)[0] + "_events"

    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Cannot read video frames.")
        cap.release()
        sys.exit(1)

    h, w  = frame0.shape[:2]
    gray0 = cv2.GaussianBlur(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
    ref_L = log_lum(gray0)
    activity = np.zeros((h, w), dtype=np.float32)

    ev_frame   = event_frame_bgr(np.zeros((h, w), bool), np.zeros((h, w), bool), h, w)
    orig_panel = frame0.copy()
    heat       = make_heatmap(activity)
    on_cnt = off_cnt = 0
    gray = gray0.copy()

    threshold  = THRESHOLD
    paused     = False
    fullscreen = False
    show_kp    = True
    show_boxes = True
    use_filter = True
    use_dbscan = True
    frame_no   = 1
    fps_proc   = fps_vid
    t_prev     = time.perf_counter()

    # Neuromorphic Pipeline: MR-NDSEF -> DBSCAN -> Tracking & SfM Pose
    ndsef        = MR_NDSEF_Filter()
    sfm          = SfMPipeline(h, w)
    tracker      = EventObjectTracker()
    tracked_objs = []
    prev_ev_gray = None
    prev_kp_pts  = None
    sfm_counter  = 0
    current_kps  = None
    clean_cnt    = 0
    candidate_clusters = []
    target_mask  = np.zeros((h, w), dtype=np.uint8)

    video_name = os.path.basename(video_path)
    cv2.namedWindow("Event Camera -- Video", cv2.WINDOW_NORMAL)

    print("=" * 70)
    print(f"  Event Camera Simulator + MR-NDSEF + DBSCAN  |  {video_name}")
    print(f"  Frames: {total_frames}   Source FPS: {fps_vid:.1f}")
    print("=" * 70)
    print("  SPACE  -- pause / resume")
    print("  r      -- reset reference frame & tracker")
    print("  [ / ]  -- decrease / increase threshold C")
    print("  f      -- toggle fullscreen event panel")
    print("  s      -- save current event frame as PNG")
    print("  n      -- toggle MR-NDSEF noise filter")
    print("  d      -- toggle DBSCAN clustering")
    print("  b / o  -- toggle object bounding boxes & tracking")
    print("  k      -- toggle keypoint overlay")
    print("  e      -- export 3-D cloud as PLY")
    print("  c      -- clear 3-D cloud")
    print("  q/ESC  -- quit")
    print("=" * 70)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                paused = True
                continue
            frame_no += 1

            # Event detection
            gray  = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
            cur_L = log_lum(gray)
            delta = cur_L - ref_L

            on_mask  = delta >  threshold
            off_mask = delta < -threshold

            ref_L[on_mask | off_mask] = cur_L[on_mask | off_mask]

            on_cnt  = int(np.count_nonzero(on_mask))
            off_cnt = int(np.count_nonzero(off_mask))

            activity *= DECAY
            activity[on_mask | off_mask] += 1.0

            # ------------------------------------------------------------------ #
            # 1. Run MR-NDSEF on raw events first to remove noise
            # ------------------------------------------------------------------ #
            if use_filter:
                clean_on, clean_off, clean_events = ndsef.filter(on_mask, off_mask, activity)
            else:
                clean_on, clean_off, clean_events = on_mask, off_mask, (on_mask | off_mask)

            clean_cnt = int(np.count_nonzero(clean_events))

            # ------------------------------------------------------------------ #
            # 2. DBSCAN to find candidate target clusters from cleaned events
            # ------------------------------------------------------------------ #
            if use_dbscan and clean_cnt > 0:
                candidate_clusters, target_mask = dbscan_cluster_events(clean_events)
            else:
                candidate_clusters, target_mask = [], np.zeros((h, w), dtype=np.uint8)

            # ------------------------------------------------------------------ #
            # 3. Feed only these cleaned clusters to tracking/pose module
            # ------------------------------------------------------------------ #
            # 3a. Tracking module
            tracked_objs = tracker.update_from_clusters(candidate_clusters)

            orig_panel = frame.copy()
            ev_frame   = event_frame_bgr(clean_on, clean_off, h, w)
            heat       = make_heatmap(activity)

            # 3b. Pose module: construct intensity map strictly from cleaned cluster events
            ev_intensity = np.zeros((h, w), dtype=np.uint8)
            ev_intensity[clean_on]  = 255
            ev_intensity[clean_off] = 128
            if np.any(target_mask):
                ev_intensity = cv2.bitwise_and(ev_intensity, target_mask)
            ev_gray = cv2.GaussianBlur(ev_intensity, (5, 5), 1.5)

            sfm_counter += 1

            if prev_ev_gray is not None and prev_kp_pts is not None:
                good_prev, good_curr = track_keypoints(
                    prev_ev_gray, ev_gray, prev_kp_pts
                )
                current_kps = good_curr

                if sfm_counter >= SFM_INTERVAL and good_prev is not None:
                    if good_curr is not None and len(good_curr) > 0:
                        mean_x = float(good_curr[:, 0, 0].mean()) / w
                        mean_y = float(good_curr[:, 0, 1].mean()) / h
                        hint = (
                            int(30  + 220 * (1 - mean_x)),
                            int(180 + 75  * mean_y),
                            int(30  + 220 * mean_x),
                        )
                        sfm.try_triangulate(good_prev, good_curr, color_hint=hint)
                    sfm_counter = 0

            # Refresh keypoints strictly constrained to validated target clusters
            n_tracked = len(current_kps) if current_kps is not None else 0
            if n_tracked < SFM_MIN_PAIRS or sfm_counter == 0:
                new_kps = detect_harris_keypoints(ev_gray, target_mask=target_mask)
                if new_kps is not None:
                    prev_kp_pts = new_kps
                    current_kps = new_kps
                else:
                    prev_kp_pts = None
            else:
                prev_kp_pts = current_kps

            prev_ev_gray = ev_gray

            t_now    = time.perf_counter()
            fps_proc = 0.9 * fps_proc + 0.1 / max(t_now - t_prev, 1e-9)
            t_prev   = t_now

        # Annotate panels
        ev_disp   = ev_frame.copy()
        orig_disp = orig_panel.copy()
        heat_disp = heat.copy()

        # Draw DBSCAN candidate cluster boundaries faintly on event panel
        if use_dbscan and candidate_clusters:
            for c in candidate_clusters:
                bx, by, bw, bh = c['box']
                cv2.rectangle(ev_disp, (bx, by), (bx + bw, by + bh), (60, 60, 120), 1)

        # Object bounding boxes overlay (on Event Frame and Original Video)
        if show_boxes and tracked_objs:
            for trk in tracked_objs:
                draw_object_box(ev_disp, trk)
                draw_object_box(orig_disp, trk)

        # Keypoint overlay
        if show_kp and current_kps is not None:
            for pt in current_kps.reshape(-1, 2):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(ev_disp, (x, y), 4, COL_KP, 1, cv2.LINE_AA)
                cv2.circle(ev_disp, (x, y), 1, COL_KP, -1, cv2.LINE_AA)

        n_kp   = len(current_kps) if current_kps is not None else 0
        n_3dpt = len(sfm.cloud_pts)
        n_objs = len(tracked_objs)
        n_tgts = len(candidate_clusters)
        panel_header(orig_disp, f"Original: {video_name}")
        panel_header(ev_disp,   f"Event Frame [MR-NDSEF+DBSCAN]  Tgts:{n_tgts}  Objs:{n_objs}  KP:{n_kp}")
        panel_header(heat_disp, f"Activity Heat-map  |  3D pts:{n_3dpt}")

        overlay_stats(ev_disp, on_cnt, off_cnt, clean_cnt, threshold,
                      frame_no, total_frames, fps_vid, fps_proc,
                      paused, h, w, filter_active=use_filter, n_targets=n_tgts)

        draw_label(orig_disp,
                   "SPACE=pause  r=reset  n=filter  d=dbscan  b=boxes  k=kp  e=ply  q=quit",
                   (8, h - 10), COL_TEXT, scale=0.36)

        if fullscreen:
            display = ev_disp
        else:
            display = np.hstack([orig_disp, ev_disp, heat_disp])

        cv2.imshow("Event Camera -- Video", display)

        key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_no}.")
        elif key == ord('r'):
            ref_L    = log_lum(gray)
            activity = np.zeros((h, w), dtype=np.float32)
            tracker.reset()
            print(f"[INFO] Reference reset at frame {frame_no}.")
        elif key == ord('n'):
            use_filter = not use_filter
            print(f"[INFO] MR-NDSEF filter = {use_filter}")
        elif key == ord('d'):
            use_dbscan = not use_dbscan
            print(f"[INFO] DBSCAN clustering = {use_dbscan}")
        elif key == ord(']'):
            threshold = min(threshold + THRESHOLD_STEP, 1.0)
            print(f"[INFO] Threshold = {threshold:.3f}")
        elif key == ord('['):
            threshold = max(threshold - THRESHOLD_STEP, 0.005)
            print(f"[INFO] Threshold = {threshold:.3f}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            print(f"[INFO] Fullscreen = {fullscreen}")
        elif key == ord('s'):
            save_frame(ev_disp, frame_no, out_dir)
        elif key in (ord('b'), ord('o')):
            show_boxes = not show_boxes
            print(f"[INFO] Object bounding boxes = {show_boxes}")
        elif key == ord('k'):
            show_kp = not show_kp
            print(f"[INFO] Keypoint overlay = {show_kp}")
        elif key == ord('e'):
            ply_path = os.path.join(out_dir, f"cloud_{frame_no:06d}.ply")
            os.makedirs(out_dir, exist_ok=True)
            sfm.export_ply(ply_path)
        elif key == ord('c'):
            sfm.clear()
            tracker.reset()
            print("[INFO] 3-D cloud and object tracks cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
