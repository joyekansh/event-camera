"""
Event Camera Simulator — Video File Input | Python + OpenCV
============================================================
Feed any video file via a file-picker dialog (or CLI arg) and
visualise it as if captured by a Dynamic Vision Sensor (DVS).

Usage:
    python3 event_video.py               ← opens file-picker dialog
    python3 event_video.py video.mp4     ← direct path

Controls (press key while the window is focused):
    q / ESC  — quit
    SPACE    — pause / resume
    r        — reset reference frame at current position
    [ / ]    — decrease / increase threshold C
    f        — toggle fullscreen event-only view
    s        — save current event frame as PNG
"""

import cv2
import numpy as np
import sys
import os
import time

# ── try to import tkinter for the file-picker ──────────────────────────────── #
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ─────────────────────────── configuration ────────────────────────────────── #
THRESHOLD      = 0.08
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

# ─────────────────────────── helpers ──────────────────────────────────────── #

def pick_video_file() -> str:
    """Open a native file-picker dialog and return the chosen path."""
    if not HAS_TK:
        print("[ERROR] tkinter not available. Pass the video path as an argument:")
        print("        python3 event_video.py <path/to/video>")
        sys.exit(1)
    root = tk.Tk()
    root.withdraw()          # hide the blank Tk window
    root.call("wm", "attributes", ".", "-topmost", True)  # bring dialog to front
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", " ".join(f"*{e}" for e in VIDEO_EXTS)),
            ("All files",   "*.*"),
        ],
    )
    root.destroy()
    return path

def log_lum(gray: np.ndarray) -> np.ndarray:
    return np.log(gray.astype(np.float32) + NOISE_FLOOR)

def draw_label(img, text, pos, color=COL_TEXT, scale=0.55, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,   thickness,   cv2.LINE_AA)

def panel_header(panel, title):
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0,0), (w, 28), (20,20,30), -1)
    draw_label(panel, title, (8, 20), COL_TITLE, scale=0.60)

def make_heatmap(activity: np.ndarray) -> np.ndarray:
    normed = cv2.normalize(activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normed, cv2.COLORMAP_INFERNO)

def event_frame_bgr(on_mask, off_mask, h, w) -> np.ndarray:
    frame = np.full((h, w, 3), COL_BG, dtype=np.uint8)
    frame[on_mask]  = COL_ON
    frame[off_mask] = COL_OFF
    return frame

def overlay_stats(panel, on_cnt, off_cnt, threshold,
                  frame_no, total_frames, fps_vid, fps_proc, paused, h, w):
    # progress bar
    progress = frame_no / max(total_frames, 1)
    bar_x = int((w - 16) * progress)
    cv2.rectangle(panel, (8, 36), (w-8, 44), (40,40,50), -1)
    cv2.rectangle(panel, (8, 36), (8+bar_x, 44), (80,180,255), -1)
    draw_label(panel, f"Frame {frame_no}/{total_frames}   "
                      f"src:{fps_vid:.0f}fps  proc:{fps_proc:.1f}fps",
               (8, 58), COL_TEXT, scale=0.42)

    bar_max  = max(on_cnt + off_cnt, 1)
    bar_w    = w - 16
    on_frac  = int(bar_w * on_cnt  / bar_max)
    off_frac = int(bar_w * off_cnt / bar_max)
    y = h - 60
    cv2.rectangle(panel, (8, y),    (8+on_frac,  y+8),  COL_BAR_ON,  -1)
    cv2.rectangle(panel, (8, y+12), (8+off_frac, y+20), COL_BAR_OFF, -1)
    draw_label(panel, f"ON  events : {on_cnt:6d}",  (8, y-4),  COL_ON,  scale=0.45)
    draw_label(panel, f"OFF events : {off_cnt:6d}", (8, y+34), COL_OFF, scale=0.45)
    draw_label(panel, f"Threshold C = {threshold:.3f}  ([ ] to change)",
               (8, h-10), COL_THRESH, scale=0.42)

    if paused:
        draw_label(panel, "PAUSED", (w//2-38, h//2),
                   COL_PAUSE, scale=1.0, thickness=2)

def save_frame(ev_frame, frame_no, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"event_frame_{frame_no:06d}.png")
    cv2.imwrite(fname, ev_frame)
    print(f"[INFO] Saved → {fname}")

# ─────────────────────────── main ─────────────────────────────────────────── #

def main():
    # ── resolve video path ────────────────────────────────────────────────── #
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        print("[INFO] No video path given — opening file picker …")
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
    frame_delay  = max(1, int(1000 / fps_vid))   # ms between frames
    out_dir      = os.path.splitext(video_path)[0] + "_events"

    # ── first frame → initialise reference ───────────────────────────────── #
    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Cannot read video frames.")
        cap.release()
        sys.exit(1)

    h, w  = frame0.shape[:2]
    gray0 = cv2.GaussianBlur(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
    ref_L = log_lum(gray0)
    activity = np.zeros((h, w), dtype=np.float32)

    # keep last computed results so we can still render while paused
    ev_frame  = event_frame_bgr(np.zeros((h,w), bool), np.zeros((h,w), bool), h, w)
    orig_panel = frame0.copy()
    heat       = make_heatmap(activity)
    on_cnt = off_cnt = 0
    gray = gray0.copy()

    threshold  = THRESHOLD
    paused     = False
    fullscreen = False
    frame_no   = 1
    fps_proc   = fps_vid
    t_prev     = time.perf_counter()

    video_name = os.path.basename(video_path)
    cv2.namedWindow("Event Camera — Video", cv2.WINDOW_NORMAL)

    print("=" * 62)
    print(f"  Event Camera Simulator  |  {video_name}")
    print(f"  Frames: {total_frames}   Source FPS: {fps_vid:.1f}")
    print("=" * 62)
    print("  SPACE  — pause / resume")
    print("  r      — reset reference frame")
    print("  [ / ]  — decrease / increase threshold C")
    print("  f      — toggle fullscreen event panel")
    print("  s      — save current event frame as PNG")
    print("  q/ESC  — quit")
    print("=" * 62)

    # ── main loop ─────────────────────────────────────────────────────────── #
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                paused = True   # freeze on last frame instead of closing
                continue
            frame_no += 1

            # ── detect events ─────────────────────────────────────────────── #
            gray  = cv2.GaussianBlur(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
            cur_L = log_lum(gray)
            delta = cur_L - ref_L

            on_mask  = delta >  threshold    # log-lum increased → ON  event
            off_mask = delta < -threshold    # log-lum decreased → OFF event

            # async reference reset — only fired pixels are updated
            ref_L[on_mask | off_mask] = cur_L[on_mask | off_mask]

            on_cnt  = int(np.count_nonzero(on_mask))
            off_cnt = int(np.count_nonzero(off_mask))

            # ── heat-map ──────────────────────────────────────────────────── #
            activity *= DECAY
            activity[on_mask | off_mask] += 1.0

            # ── build panels ─────────────────────────────────────────────── #
            orig_panel = frame.copy()
            ev_frame   = event_frame_bgr(on_mask, off_mask, h, w)
            heat       = make_heatmap(activity)

            # FPS tracking
            t_now    = time.perf_counter()
            fps_proc = 0.9 * fps_proc + 0.1 / max(t_now - t_prev, 1e-9)
            t_prev   = t_now

        # ── annotate ──────────────────────────────────────────────────────── #
        ev_disp   = ev_frame.copy()
        orig_disp = orig_panel.copy()
        heat_disp = heat.copy()

        panel_header(orig_disp, f"Original: {video_name}")
        panel_header(ev_disp,   "Event Frame  [ON=cyan  OFF=red]")
        panel_header(heat_disp, "Activity Heat-map")

        overlay_stats(ev_disp, on_cnt, off_cnt, threshold,
                      frame_no, total_frames, fps_vid, fps_proc,
                      paused, h, w)

        draw_label(orig_disp, "SPACE=pause  r=reset  s=save  f=full  q=quit",
                   (8, h-10), COL_TEXT, scale=0.38)

        # ── compose & show ────────────────────────────────────────────────── #
        display = ev_disp if fullscreen else np.hstack([orig_disp, ev_disp, heat_disp])
        cv2.imshow("Event Camera — Video", display)

        # ── key handling ──────────────────────────────────────────────────── #
        key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_no}.")
        elif key == ord('r'):
            ref_L    = log_lum(gray)
            activity = np.zeros((h, w), dtype=np.float32)
            print(f"[INFO] Reference reset at frame {frame_no}.")
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

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
