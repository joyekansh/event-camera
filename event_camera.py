"""
Event Camera Simulator — Python + OpenCV
=========================================
Simulates a Dynamic Vision Sensor (DVS / Event Camera) using a standard webcam.

How it works:
  • Each frame is converted to grayscale and log-luminance is computed.
  • Per-pixel difference from the last reference frame is accumulated.
  • When the accumulated change exceeds a threshold (+C or −C),
    an event is fired:
      - ON  event  → brightness INCREASED  (shown in CYAN)
      - OFF event  → brightness DECREASED  (shown in RED)
  • The reference log-luminance is updated only at fired pixels,
    mimicking how a real DVS resets its reference asynchronously.

Controls (press key while any window is focused):
  q / ESC  — quit
  r        — reset reference frame
  [ / ]    — decrease / increase threshold C
  f        — toggle fullscreen event view
"""

import cv2
import numpy as np
import time

# ─────────────────────────── configuration ────────────────────────────────── #
THRESHOLD      = 0.1  # log-luminance change threshold  (0.01 – 0.5 typical)
THRESHOLD_STEP = 0.05
NOISE_FLOOR    = 1e-6   # added before log to avoid log(0)
BLUR_KSIZE     = (3, 3) # gaussian blur kernel to reduce noise
DECAY          = 0.92   # heat-map exponential decay per frame
FPS_SMOOTH     = 0.1    # EMA coefficient for FPS display
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# colours (BGR)
COL_ON       = (220, 240,  10)   # cyan-ish  — ON  event
COL_OFF      = ( 30,  40, 240)   # red-ish   — OFF event
COL_BG       = (  5,   5,  10)   # near-black background
COL_TEXT     = (200, 200, 200)
COL_TITLE    = ( 80, 220, 255)
COL_THRESH   = ( 50, 255, 150)
COL_BAR_ON   = (200, 220,  10)
COL_BAR_OFF  = ( 20,  30, 220)

# ─────────────────────────── helpers ──────────────────────────────────────── #

def log_luminance(gray):
    """Return log-luminance as float32."""
    return np.log(gray.astype(np.float32) + NOISE_FLOOR)


def draw_label(img, text, pos, color=COL_TEXT, scale=0.55, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def panel_header(panel, title):
    """Draw a stylised header bar at the top of a panel."""
    h, w = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (w, 28), (20, 20, 30), -1)
    draw_label(panel, title, (8, 20), COL_TITLE, scale=0.60, thickness=1)


def make_heatmap(activity):
    """Convert normalised activity map to a coloured heat-map BGR image."""
    normed = cv2.normalize(activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat   = cv2.applyColorMap(normed, cv2.COLORMAP_INFERNO)
    return heat


def event_frame_to_bgr(on_mask, off_mask, h, w):
    """Render ON/OFF event masks onto a dark background."""
    frame = np.full((h, w, 3), COL_BG, dtype=np.uint8)
    frame[on_mask]  = COL_ON
    frame[off_mask] = COL_OFF
    return frame


def overlay_stats(panel, on_count, off_count, threshold, fps, h, w):
    """Overlay counters and threshold info on the event panel."""
    bar_max = max(on_count + off_count, 1)
    bar_w   = w - 16
    on_frac  = int(bar_w * on_count  / bar_max)
    off_frac = int(bar_w * off_count / bar_max)

    y = h - 60
    cv2.rectangle(panel, (8, y),     (8 + on_frac,  y + 8),  COL_BAR_ON,  -1)
    cv2.rectangle(panel, (8, y + 12),(8 + off_frac, y + 20), COL_BAR_OFF, -1)

    draw_label(panel, f"ON  events : {on_count:6d}",  (8, y - 4),  COL_ON,  scale=0.45)
    draw_label(panel, f"OFF events : {off_count:6d}", (8, y + 34), COL_OFF, scale=0.45)

    draw_label(panel, f"Threshold C = {threshold:.3f}  ([ ] to change)",
               (8, h - 10), COL_THRESH, scale=0.42)
    draw_label(panel, f"{fps:.1f} FPS", (w - 80, 20), COL_TEXT, scale=0.50)


# ─────────────────────────── main loop ────────────────────────────────────── #

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # try to set a comfortable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame0 = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return

    h, w = frame0.shape[:2]

    # initialise reference log-luminance
    gray0  = cv2.GaussianBlur(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
    ref_L  = log_luminance(gray0)

    # activity accumulator for heat-map
    activity = np.zeros((h, w), dtype=np.float32)

    threshold  = THRESHOLD
    fps        = 30.0
    t_prev     = time.perf_counter()
    fullscreen = False

    cv2.namedWindow("Event Camera Simulator", cv2.WINDOW_NORMAL)

    print("=" * 60)
    print("  Event Camera Simulator  |  Python + OpenCV")
    print("=" * 60)
    print("  q / ESC  — quit")
    print("  r        — reset reference frame")
    print("  [ / ]    — decrease / increase threshold C")
    print("  f        — toggle fullscreen event panel")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── compute events ──────────────────────────────────────────────── #
        gray  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BLUR_KSIZE, 0)
        cur_L = log_luminance(gray)
        delta = cur_L - ref_L

        on_mask  = delta >  threshold   # brightness INCREASED
        off_mask = delta < -threshold   # brightness DECREASED

        # update reference only at fired pixels (async reset simulation)
        ref_L[on_mask | off_mask] = cur_L[on_mask | off_mask]

        on_count  = int(np.count_nonzero(on_mask))
        off_count = int(np.count_nonzero(off_mask))

        # ── update activity heat-map ─────────────────────────────────────── #
        activity *= DECAY
        activity[on_mask | off_mask] += 1.0

        # ── build display panels ─────────────────────────────────────────── #
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ev_frame = event_frame_to_bgr(on_mask, off_mask, h, w)
        heat     = make_heatmap(activity)

        # ── annotate panels ──────────────────────────────────────────────── #
        panel_header(gray_bgr, "Webcam Input (gray)")
        panel_header(ev_frame, "Event Frame  [ON=cyan  OFF=red]")
        panel_header(heat,     "Activity Heat-map")

        # FPS (EMA)
        t_now  = time.perf_counter()
        fps    = (1 - FPS_SMOOTH) * fps + FPS_SMOOTH / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        overlay_stats(ev_frame, on_count, off_count, threshold, fps, h, w)

        draw_label(gray_bgr, "r=reset  [/]=thresh  f=full  q=quit",
                   (8, h - 10), COL_TEXT, scale=0.40)

        # ── compose side-by-side display ─────────────────────────────────── #
        if fullscreen:
            display = ev_frame
        else:
            display = np.hstack([gray_bgr, ev_frame, heat])

        cv2.imshow("Event Camera Simulator", display)

        # ── key handling ─────────────────────────────────────────────────── #
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            ref_L    = log_luminance(gray)
            activity = np.zeros((h, w), dtype=np.float32)
            print(f"[INFO] Reference frame reset.")
        elif key == ord(']'):
            threshold = min(threshold + THRESHOLD_STEP, 1.0)
            print(f"[INFO] Threshold = {threshold:.3f}")
        elif key == ord('['):
            threshold = max(threshold - THRESHOLD_STEP, 0.005)
            print(f"[INFO] Threshold = {threshold:.3f}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            print(f"[INFO] Fullscreen = {fullscreen}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
