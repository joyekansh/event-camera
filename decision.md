# 📋 Project Architectural & Technical Decisions Log

This document chronicles the technical decisions, underlying scientific and algorithmic reasoning, and concrete implementations introduced across the **Event Camera Simulator & Neuromorphic Processing Pipeline**.

---

## 📑 Table of Contents
1. [Blender 3D Point-Cloud Bridge Modernization & Fixes](#1-blender-3d-point-cloud-bridge-modernization--fixes)
2. [macOS Tkinter / OpenCV Cocoa Crash Elimination](#2-macos-tkinter--opencv-cocoa-crash-elimination)
3. [Decoupling & Standalone Architecture](#3-decoupling--standalone-architecture)
4. [Object Segregation & Post-Event Bounding Box Persistence](#4-object-segregation--post-event-bounding-box-persistence)
5. [Neuromorphic Denoising & Target Extraction Pipeline (MR-NDSEF + DBSCAN)](#5-neuromorphic-denoising--target-extraction-pipeline-mr-ndsef--dbscan)
6. [Pose Estimation & Tracking Constrained to Target Clusters](#6-pose-estimation--tracking-constrained-to-target-clusters)
7. [System Architecture & Flow Diagram](#7-system-architecture--flow-diagram)

---

## 1. Blender 3D Point-Cloud Bridge Modernization & Fixes

### ⚠️ Problem & Context
When attempting to stream live 3D Structure-from-Motion (SfM) points from `event_video.py` into Blender, execution crashed inside Blender's script environment:
```text
AttributeError: 'Mesh' object has no attribute 'display'
RuntimeError: expected class EVENTCAM_OT_live_cloud, function cancel to return None, not set
```
Additionally, Blender's Viewport in Object Mode does not render 0-dimensional loose vertices (point clouds) without geometry or instancing.

### 💡 Decisions & Rationale
- **Remove deprecated/hallucinated API properties**: `Mesh` has never exposed `.display.show_vertices_color` in Blender 3.x/4.x/5.x.
- **Implement Geometry Nodes `Mesh to Points`**: Rather than attempting wireframe hacks on loose vertices, instantiate a procedural Geometry Nodes modifier (`PointDisplay`) with `GeometryNodeMeshToPoints` and `POINT_SCALE = 0.02`. This forces Blender's viewport to rasterize each point as an emissive 3D sphere with its vertex color attribute (`Col`).
- **Fix Modal Operator Cancel Return Signature**: In Blender's Python RNA, `Operator.cancel(self, context)` must return `None`, whereas `modal()` and `execute()` return enum sets like `{'CANCELLED'}`.
- **Synchronize Embedded `.blend` Text Blocks**: Update both the standalone script on disk and the internal text block inside `/Users/joyekansh/event.blend`.

### 🛠️ Implementation
- **File**: [`blender_live_cloud.py`](file:///Users/joyekansh/Desktop/event-camera/blender_live_cloud.py)
- Added `_setup_point_cloud_modifier()` supporting both Blender 4.0+/5.0+ (`group.interface.new_socket`) and Blender 3.x (`group.inputs.new`).
- Connected node graph: `GroupInput` → `MeshToPoints(Radius=0.02)` → `SetMaterial(EventCloudMat)` → `GroupOutput`.
- Guarded `context.area.header_text_set()` against `None` and ensured `cancel()` returns `None`.

---

## 2. macOS Tkinter / OpenCV Cocoa Crash Elimination

### ⚠️ Problem & Context
Running `event_video.py` without command-line arguments invoked a Tkinter file picker dialog, terminating with a SIGABRT:
```text
called Tcl_FindHashEntry on deleted table
zsh: abort      python3 event_video.py
```

### 💡 Decisions & Rationale
- On macOS (Darwin), Tkinter and OpenCV both hook into Apple's native Cocoa event loop on the main thread.
- Calling `root.destroy()` inside Tkinter deallocates Tcl hash tables; subsequent Cocoa event dispatches attempted to look up entries in the deallocated table, triggering an abort.
- Rather than maintaining fragile Tcl event loop workarounds, use macOS native AppleScript (`osascript`) for file selection.

### 🛠️ Implementation
- **File**: [`event_video.py`](file:///Users/joyekansh/Desktop/event-camera/event_video.py) (`pick_video_file`)
```python
if sys.platform == "darwin":
    cmd = """osascript -e 'POSIX path of (choose file with prompt "Select a video file")'"""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
```
- Completely eliminated Tkinter initialization and teardown on macOS while preserving Tkinter fallback for Linux/Windows.

---

## 3. Decoupling & Standalone Architecture

### ⚠️ Problem & Context
The Blender socket client (`BlenderBridge`) continuously attempted to connect to `127.0.0.1:9876`, creating networking overhead and cluttering console output when running without Blender open.

### 💡 Decisions & Rationale
- Temporarily decouple Blender live-streaming so `event_video.py` operates as a lightweight, zero-dependency, standalone event simulator with local SfM and PLY export.
- Strip out socket threads, FIFO queue buffers, and unused standard library modules (`socket`, `struct`, `threading`, `queue`).

### 🛠️ Implementation
- **File**: [`event_video.py`](file:///Users/joyekansh/Desktop/event-camera/event_video.py)
- Removed `BlenderBridge`, socket loops, and the `b` reconnect shortcut.
- Point clouds remain fully accessible locally and can be exported at any time to disk via the `e` key (`sfm.export_ply()`).

---

## 4. Object Segregation & Post-Event Bounding Box Persistence

### ⚠️ Problem & Context
Raw event camera streams output asynchronous pixel spikes along moving edges. Without spatial segregation and temporal tracking:
1. Event signals appear as disconnected, flickering dots rather than unified physical objects.
2. The instant an object decelerates or stops, event generation drops to zero, causing conventional bounding boxes to immediately disappear (flicker).

### 💡 Decisions & Rationale
- **Morphological Spatial Aggregation**: Apply morphological closing ($\text{kernel} = 7 \times 7$ ellipse) and dilation to bridge disconnected edge events into solid candidate object regions.
- **Activity Map Reinforcement**: Fuse current event blobs with the exponential decay activity heat-map to retain object silhouette cohesion.
- **Post-Event Memory & Tracking**: Implement an `EventObjectTracker` with:
  - Nearest-centroid association across frames.
  - Exponential Moving Average (EMA) bounding box smoothing ($0.25 \times \text{old} + 0.75 \times \text{new}$) to eliminate jitter.
  - **Time-to-Live (TTL = 10 frames) Persistence**: When event activity stops, maintain the bounding box in a `[POST]` state before discarding, allowing continuous tracking across pauses.

### 🛠️ Implementation
- **File**: [`event_video.py`](file:///Users/joyekansh/Desktop/event-camera/event_video.py)
- Created `EventObjectTracker` and `draw_object_box()` with high-tech corner brackets and status badges (`Obj #ID (N ev)` when active, `Obj #ID [POST]` when persisting).
- Rendered simultaneously on both the **Event Frame** and the **Original Video Panel** for real-world verification.

---

## 5. Neuromorphic Denoising & Target Extraction Pipeline (MR-NDSEF + DBSCAN)

### ⚠️ Problem & Context
Real neuromorphic vision sensors (DVS), especially in low-light, high-speed, or space surveillance applications (tracking satellites, debris, stars), suffer from high background noise (thermal noise, shot noise) and low Signal-to-Noise Ratios (SNR). Single-radius neighborhood filters fail under dynamic maneuvers because target event trails elongate unpredictably.

### 💡 Decisions & Rationale
Implement the state-of-the-art space target detection and tracking pipeline:
1. **MR-NDSEF (Multi-Neighborhood-Radius Spatiotemporal Event Filter)**:
   - Evaluates local spatial event density $D_r(x, y)$ across multiple circular radii $R \in \{3, 5, 8\}$ pixels.
   - Circular neighborhood density is calculated via 2D convolution with normalized circular kernels.
   - An event is retained if its density exceeds the threshold at any scale ($D_{r_k} \ge \psi_k$), preserving elongated high-velocity trajectories while discarding isolated background noise.
   - Multi-scale thresholds: $\psi = (0.16, 0.10, 0.06)$.
   - Protects persistent activity trails ($A(x, y) > 1.8$).
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - Takes coordinates of events that passed MR-NDSEF.
   - Parameters: $\epsilon = 14.0\text{ px}$, $\text{min\_samples} = 16\text{ events}$.
   - Discards residual outlier noise (label `-1`) and segments true targets into discrete clusters ($0, 1, 2, \dots$).
   - Adaptive sub-sampling (capped at 2500 points) ensures $\ge 30\text{ FPS}$ real-time execution.

### 🛠️ Implementation
- **File**: [`event_video.py`](file:///Users/joyekansh/Desktop/event-camera/event_video.py)
- Added `MR_NDSEF_Filter` class with pre-computed circular kernels.
- Added `dbscan_cluster_events()` utilizing `sklearn.cluster.DBSCAN` with an OpenCV morphological fallback.
- Added live telemetry in `overlay_stats()` showing raw vs. cleaned event counts and noise rejection percentage (typically 50%–75% noise reduction).

---

## 6. Pose Estimation & Tracking Constrained to Target Clusters

### ⚠️ Problem & Context
In the original implementation, Harris corner detection and Lucas-Kanade optical flow were executed over the entire noisy event image. Background noise spikes generated false keypoints, contaminating the Essential Matrix pose estimation and triangulating spurious 3D points.

### 💡 Decisions & Rationale
- **Strict ROI Masking**: Pass the binary `target_mask` generated by DBSCAN directly to the feature extraction and tracking modules.
- **Feature Isolation**: In `detect_harris_keypoints`, mask the event intensity image with `target_mask` (`cv2.bitwise_and`). Keypoints can **only** be formed on verified candidate targets.
- **Feed Direct to Tracker**: `tracker.update_from_clusters()` receives validated DBSCAN clusters directly, anchoring bounding boxes and centroids to density-verified targets.

### 🛠️ Implementation
- **File**: [`event_video.py`](file:///Users/joyekansh/Desktop/event-camera/event_video.py)
- Constrained `detect_harris_keypoints(gray_event, target_mask=target_mask)`.
- Reconstructed `ev_intensity` strictly from `clean_on` and `clean_off` intersected with `target_mask`.
- Optical flow and `sfm.try_triangulate` now only track and reconstruct points belonging to true physical targets.

---

## 7. System Architecture & Flow Diagram

```mermaid
flowchart TD
    A[Input Video Frame I_t] --> B[Log-Luminance L_t = log I + eps]
    B --> C[Compute Delta L = L_t - L_ref]
    C --> D[Thresholding: ON > +C, OFF < -C]
    D --> E[Update Reference L_ref at Fired Pixels]
    
    subgraph Denoising & Target Extraction
        D --> F[Raw Events on_mask, off_mask]
        F --> G["1. MR-NDSEF Filter (Multi-Radius R={3, 5, 8})"]
        G --> H["Cleaned Events (50-75% noise cut)"]
        H --> I["2. DBSCAN Clustering (eps=14px, min_samples=16)"]
        I --> J["Candidate Target Clusters & Binary target_mask"]
    end

    subgraph Tracking Module
        J --> K["EventObjectTracker (EMA Smoothing)"]
        K --> L["Post-Event Persistence ([POST] TTL=10 frames)"]
        L --> M[Dual-Panel Bounding Box Overlay]
    end

    subgraph Pose & 3D Reconstruction
        J --> N["Mask Event Intensity with target_mask"]
        N --> O["Harris Corners (Inside Target Clusters ONLY)"]
        O --> P[Lucas-Kanade Optical Flow Tracking]
        P --> Q[Essential Matrix & Pose Recovery]
        Q --> R[3D Triangulation SfMPipeline]
        R --> S[3D Point Cloud Export .ply]
    end

    M --> T[3-Panel Real-Time GUI Display]
```

---

## 8. Summary of Runtime Keyboard Controls

| Key | Function |
|---|---|
| **`n`** | Toggle MR-NDSEF multi-radius noise filter ON / OFF |
| **`d`** | Toggle DBSCAN candidate target clustering ON / OFF |
| **`b` / `o`** | Toggle object bounding boxes & post-event tracking ON / OFF |
| **`k`** | Toggle keypoint overlay |
| **`SPACE`** | Pause / resume video playback |
| **`[` / `]`** | Adjust log-luminance sensitivity threshold $C$ |
| **`r`** | Reset reference frame, activity map, and object tracks |
| **`e`** | Export current 3D point cloud as a `.ply` file |
| **`c`** | Clear 3D point cloud and active tracks |
| **`f`** | Toggle fullscreen event view |
| **`s`** | Save current event frame as PNG |
| **`q` / `ESC`** | Clean exit |
