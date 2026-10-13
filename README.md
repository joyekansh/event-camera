# 🎥 Event Camera Simulator — Python + OpenCV

Simulates a **Dynamic Vision Sensor (DVS / Event Camera)** using a standard webcam or any video file. Instead of capturing full frames, it detects **per-pixel brightness changes** in real time — just like a real neuromorphic event camera.

---

## 📁 Project Structure

```
event-camera/
├── event_camera.py   ← Live webcam → event detection
├── event_video.py    ← Video file → event detection (with file picker)
└── README.md
```

---

## ⚙️ Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| opencv-python | ≥ 4.x |
| numpy | ≥ 1.21 |
| tkinter | (stdlib, required by event_video.py for file picker) |

### Install dependencies

```bash
pip3 install opencv-python numpy
```

---

## 🧠 How It Works (DVS Physics)

Real event cameras (like the iniVation DAVIS or Prophesee Metavision) output asynchronous events of the form:

```
(x, y, timestamp, polarity)
```

A pixel fires an event when its **log-luminance** changes by a threshold `C`:

```
L(x,y,t) = log( I(x,y,t) + ε )

ΔL = L_current − L_reference

 ΔL > +C  →  ON  event  (brightness INCREASED)  ← cyan pixel
 ΔL < −C  →  OFF event  (brightness DECREASED)  ← red pixel
```

After firing, **only that pixel's reference is updated** — mimicking the asynchronous reset behaviour of a real DVS chip.

---

## 🖥️ Display Panels (side-by-side)

```
┌────────────────────┬──────────────────────────┬──────────────────────┐
│  Webcam / Original │      Event Frame         │   Activity Heat-map  │
│  (grayscale/colour)│  ON = CYAN  OFF = RED    │  (inferno colormap)  │
│                    │  on near-black background│  decays over time    │
└────────────────────┴──────────────────────────┴──────────────────────┘
```

Press **`f`** to toggle fullscreen event-only view.

---

## 🚀 Usage

### 1. Live Webcam (`event_camera.py`)

```bash
python3 event_camera.py
```

- Opens your default webcam (index `0`)
- Runs at live FPS
- Three panels: grayscale input | event frame | heat-map

### 2. Video File (`event_video.py`)

```bash
# Option A: file-picker dialog (recommended)
python3 event_video.py

# Option B: pass path directly
python3 event_video.py path/to/video.mp4
```

Supported formats: `.mp4 .avi .mov .mkv .flv .wmv .webm .m4v .ts .mts`

---

## ⌨️ Keyboard Controls

### `event_camera.py` (webcam)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset reference frame |
| `[` | Decrease threshold → more sensitive |
| `]` | Increase threshold → less sensitive |
| `f` | Toggle fullscreen event panel |

### `event_video.py` (video file)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `SPACE` | Pause / Resume |
| `r` | Reset reference frame at current position |
| `[` | Decrease threshold → more sensitive |
| `]` | Increase threshold → less sensitive |
| `f` | Toggle fullscreen event panel |
| `s` | Save current event frame as PNG |

> Saved frames are written to `<video_name>_events/event_frame_XXXXXX.png`

---

## 🔧 Configuration Reference

All constants are defined at the top of each script.

| Constant | Default | Description |
|---|---|---|
| `THRESHOLD` | `0.1` | Log-luminance change threshold `C`. Lower = more events (more sensitive). Typical range: `0.01` – `0.5`. |
| `THRESHOLD_STEP` | `0.05` | Step size when pressing `[` or `]` keys at runtime. |
| `NOISE_FLOOR` | `1e-6` | Added to pixel values before `log()` to prevent `log(0)`. |
| `BLUR_KSIZE` | `(3, 3)` | Gaussian blur kernel applied before computing log-luminance. Increase to reduce noise at the cost of spatial resolution. |
| `DECAY` | `0.92` | Exponential decay factor for the activity heat-map per frame. `1.0` = no decay, `0.5` = fast decay. |
| `FPS_SMOOTH` | `0.1` | EMA smoothing coefficient for the FPS display counter. |

### Colour Palette (BGR format)

| Constant | Colour | Represents |
|---|---|---|
| `COL_ON` | `(220, 240, 10)` — cyan-yellow | ON events (brightness increase) |
| `COL_OFF` | `(30, 40, 240)` — red-orange | OFF events (brightness decrease) |
| `COL_BG` | `(5, 5, 10)` — near-black | Event frame background |
| `COL_TITLE` | `(80, 220, 255)` — amber | Panel header titles |
| `COL_THRESH` | `(50, 255, 150)` — green | Threshold label |

---

## 🔬 Tuning Guide

| Scenario | Recommendation |
|---|---|
| Too few events / nothing firing | Lower `THRESHOLD` (press `[`) |
| Too noisy / events everywhere | Raise `THRESHOLD` (press `]`) |
| Flickering / false events | Increase `BLUR_KSIZE` to `(5,5)` |
| Heat-map clears too fast | Increase `DECAY` closer to `1.0` |
| Heat-map too persistent | Decrease `DECAY` closer to `0.5` |
| Slow camera / lag | Reduce webcam resolution in `cap.set(...)` calls |

---

## 📐 Architecture / Workflow

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Input Frame │────▶│ Grayscale + Blur │────▶│  Log-Luminance L_t │
└──────────────┘     └──────────────────┘     └────────┬───────────┘
                                                        │
                                              ΔL = L_t − L_ref
                                                        │
                             ┌──────────────────────────┤
                             │                          │
                        ΔL > +C                    ΔL < −C
                             │                          │
                        ON event                  OFF event
                        (CYAN px)                 (RED px)
                             │                          │
                             └──────────┬───────────────┘
                                        │
                              Update L_ref at fired pixels
                                        │
                              ┌─────────▼──────────┐
                              │  Render 3 panels   │
                              │  + Activity map    │
                              └────────────────────┘
```

---

## 📌 macOS Camera Permission

If you see `not authorized to capture video`, grant camera access to Terminal:

```
System Settings → Privacy & Security → Camera → enable Terminal
```

Then fully quit and reopen Terminal, and run the script again.

---

## 📄 License

MIT — free to use, modify, and distribute.
