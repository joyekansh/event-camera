"""
Blender Live Point-Cloud Receiver
===================================
Run this script inside Blender (Scripting tab → Run Script).

It starts a TCP server on localhost:9876 and listens for point batches
streamed from event_video.py.  Every batch is appended to a Blender mesh
object ("EventCameraCloud") which you can view, orbit, and render in
Blender's 3D Viewport.

Wire protocol (matches BlenderBridge in event_video.py):
    4 bytes  uint32  N   — number of points in this batch
                          (0xFFFFFFFF = clear-cloud signal)
    N * 16 bytes         — per point:
        float32 x, float32 y, float32 z
        uint8   r, uint8 g, uint8 b, uint8 pad

How to use:
    1. Open Blender.
    2. Go to the Scripting workspace.
    3. Open this file (or paste it) and click "Run Script".
    4. A status panel appears in the 3D Viewport header.
    5. Run:  python3 event_video.py  (in a terminal)
    6. Points will stream in live.

Keyboard shortcuts (in Blender 3D Viewport):
    The object updates automatically — just orbit with middle-mouse.
    To stop the server, click "Stop Live Cloud" in the N-panel (press N).
"""

import bpy
import bmesh
import socket
import struct
import threading
import queue
import time
from mathutils import Vector, Color

# ─────────────── configuration ──────────────────────────────────────────────
HOST          = "127.0.0.1"
PORT          = 9876
OBJ_NAME      = "EventCameraCloud"
MAT_NAME      = "EventCloudMat"
MAX_POINTS    = 15_000          # cap total points in Blender
UPDATE_HZ     = 15              # Blender timer fires this many times/sec
POINT_SCALE   = 0.02            # radius of each instanced sphere (display only)

# ─────────────── globals shared between server thread & Blender timer ────────
_recv_queue   : queue.Queue = queue.Queue()
_server_thread: threading.Thread | None = None
_server_sock  : socket.socket | None    = None
_client_sock  : socket.socket | None    = None
_running      = False
_status       = "Idle"

# accumulated cloud
_all_pts   : list[tuple[float,float,float]] = []
_all_cols  : list[tuple[float,float,float]] = []  # linear RGB 0..1


# ═══════════════════ TCP server (background thread) ═════════════════════════

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes or return None on disconnect."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _server_loop():
    global _server_sock, _client_sock, _running, _status

    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        srv.settimeout(1.0)
        _server_sock = srv
        _status = f"Listening on {HOST}:{PORT} …"
        print(f"[Blender Bridge] {_status}")

        while _running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            _client_sock = conn
            _status = f"Connected: {addr[0]}:{addr[1]}"
            print(f"[Blender Bridge] {_status}")
            conn.settimeout(1.0)

            while _running:
                # Read 4-byte header
                hdr = _recv_exact(conn, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack("<I", hdr)

                if n == 0xFFFFFFFF:
                    # Clear signal
                    _recv_queue.put(("clear", None, None))
                    continue

                if n == 0:
                    continue

                # Read N * 16 bytes of point data
                payload = _recv_exact(conn, n * 16)
                if payload is None:
                    break

                pts  = []
                cols = []
                offset = 0
                for _ in range(n):
                    x, y, z = struct.unpack_from("<fff", payload, offset)
                    r, g, b, _ = struct.unpack_from("BBBB", payload, offset + 12)
                    pts.append((x, y, z))
                    cols.append((r / 255.0, g / 255.0, b / 255.0))
                    offset += 16

                _recv_queue.put(("add", pts, cols))

            _client_sock = None
            try:
                conn.close()
            except OSError:
                pass
            _status = f"Listening on {HOST}:{PORT} …"

    except OSError as e:
        _status = f"Server error: {e}"
        print(f"[Blender Bridge] {_status}")
    finally:
        try:
            if _server_sock:
                _server_sock.close()
        except OSError:
            pass
        _status = "Stopped"
        print("[Blender Bridge] Server stopped.")


# ═══════════════════ Blender mesh helpers ════════════════════════════════════

def _setup_point_cloud_modifier(obj: bpy.types.Object, mat: bpy.types.Material):
    """Add a Geometry Nodes modifier so vertices are rendered as visible 3D points."""
    MOD_NAME = "PointDisplay"
    if MOD_NAME in obj.modifiers:
        return

    mod = obj.modifiers.new(name=MOD_NAME, type="NODES")
    tree_name = "EventCameraPointTree"
    tree = bpy.data.node_groups.get(tree_name)
    if not tree:
        tree = bpy.data.node_groups.new(tree_name, "GeometryNodeTree")
        # Support both Blender 4.0+/5.0+ (interface) and Blender 3.x (inputs/outputs)
        if hasattr(tree, "interface"):
            tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        elif hasattr(tree, "inputs"):
            tree.inputs.new("NodeSocketGeometry", "Geometry")
            tree.outputs.new("NodeSocketGeometry", "Geometry")

        in_node = tree.nodes.new("NodeGroupInput")
        in_node.location = (-300, 0)
        out_node = tree.nodes.new("NodeGroupOutput")
        out_node.location = (400, 0)

        m2p = tree.nodes.new("GeometryNodeMeshToPoints")
        m2p.location = (-50, 0)
        m2p.inputs["Radius"].default_value = POINT_SCALE

        set_mat = tree.nodes.new("GeometryNodeSetMaterial")
        set_mat.location = (180, 0)
        set_mat.inputs["Material"].default_value = mat

        tree.links.new(in_node.outputs["Geometry"], m2p.inputs["Mesh"])
        tree.links.new(m2p.outputs["Points"], set_mat.inputs["Geometry"])
        tree.links.new(set_mat.outputs["Geometry"], out_node.inputs["Geometry"])

    mod.node_group = tree


def _get_or_create_mesh_obj() -> bpy.types.Object:
    """Return (creating if needed) the point-cloud mesh object."""
    mat = bpy.data.materials.get(MAT_NAME)
    if not mat:
        mat = bpy.data.materials.new(MAT_NAME)
        if hasattr(mat, "use_nodes"):
            mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        attr  = nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "Col"
        attr.location = (-300, 0)

        emit  = nodes.new("ShaderNodeEmission")
        emit.location = (0, 0)

        out   = nodes.new("ShaderNodeOutputMaterial")
        out.location = (200, 0)

        links.new(attr.outputs["Color"], emit.inputs["Color"])
        links.new(emit.outputs["Emission"], out.inputs["Surface"])

        if hasattr(mat, "blend_method"):
            mat.blend_method = "OPAQUE"

    if OBJ_NAME in bpy.data.objects:
        obj = bpy.data.objects[OBJ_NAME]
        _setup_point_cloud_modifier(obj, mat)
        return obj

    mesh = bpy.data.meshes.new(OBJ_NAME)
    obj  = bpy.data.objects.new(OBJ_NAME, mesh)
    bpy.context.collection.objects.link(obj)

    if not obj.data.materials:
        obj.data.materials.append(mat)

    _setup_point_cloud_modifier(obj, mat)
    return obj


def _rebuild_mesh(obj: bpy.types.Object,
                  pts: list, cols: list):
    """Replace the mesh vertices with the current point list."""
    mesh = obj.data
    mesh.clear_geometry()

    if not pts:
        return

    bm = bmesh.new()
    col_layer = bm.verts.layers.float_color.new("Col")

    for (x, y, z), (r, g, b) in zip(pts, cols):
        v = bm.verts.new((x, y, z))
        v[col_layer] = (r, g, b, 1.0)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ═══════════════════ Blender modal timer operator ════════════════════════════

class EVENTCAM_OT_live_cloud(bpy.types.Operator):
    """Start / stop the live point-cloud receiver."""
    bl_idname  = "eventcam.live_cloud"
    bl_label   = "Start Live Cloud"
    bl_options = {"REGISTER"}

    _timer = None

    def modal(self, context, event):
        global _all_pts, _all_cols, _running

        if event.type == "TIMER":
            dirty = False

            # Drain the receive queue
            while not _recv_queue.empty():
                try:
                    cmd, pts, cols = _recv_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd == "clear":
                    _all_pts.clear()
                    _all_cols.clear()
                    dirty = True
                elif cmd == "add":
                    _all_pts.extend(pts)
                    _all_cols.extend(cols)
                    # Trim to cap
                    if len(_all_pts) > MAX_POINTS:
                        trim = len(_all_pts) - MAX_POINTS
                        _all_pts  = _all_pts[trim:]
                        _all_cols = _all_cols[trim:]
                    dirty = True

            if dirty:
                obj = _get_or_create_mesh_obj()
                _rebuild_mesh(obj, _all_pts, _all_cols)
                if context.screen:
                    for area in context.screen.areas:
                        if area.type == "VIEW_3D":
                            area.tag_redraw()

            # Update status in header
            if context.area:
                context.area.header_text_set(
                    f"EventCam Bridge | {_status} | "
                    f"Points: {len(_all_pts)} | "
                    f"Press ESC to stop"
                )

            if not _running:
                self.cancel(context)
                return {"CANCELLED"}

        if event.type == "ESC":
            self.cancel(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def execute(self, context):
        global _running, _server_thread

        if _running:
            self.report({"WARNING"}, "Already running.")
            return {"CANCELLED"}

        _running = True
        _server_thread = threading.Thread(target=_server_loop, daemon=True)
        _server_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / UPDATE_HZ, window=context.window)
        wm.modal_handler_add(self)

        self.report({"INFO"}, f"Live cloud server started on port {PORT}.")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        global _running
        _running = False
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        if context.area:
            context.area.header_text_set(None)
        self.report({"INFO"}, "Live cloud server stopped.")
        return None


class EVENTCAM_OT_clear_cloud(bpy.types.Operator):
    """Clear the accumulated point cloud in Blender."""
    bl_idname = "eventcam.clear_cloud"
    bl_label  = "Clear Cloud"

    def execute(self, context):
        global _all_pts, _all_cols
        _all_pts.clear()
        _all_cols.clear()
        if OBJ_NAME in bpy.data.objects:
            obj = bpy.data.objects[OBJ_NAME]
            obj.data.clear_geometry()
        return {"FINISHED"}


# ═══════════════════ N-panel UI ══════════════════════════════════════════════

class EVENTCAM_PT_panel(bpy.types.Panel):
    bl_label       = "Event Camera SfM"
    bl_idname      = "EVENTCAM_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "EventCam"

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Status: {_status}", icon="SNAP_FACE")
        layout.label(text=f"Points: {len(_all_pts)}")
        layout.separator()
        if not _running:
            layout.operator("eventcam.live_cloud", text="Start Live Cloud",
                            icon="PLAY")
        else:
            layout.label(text="Running — press ESC in viewport to stop",
                         icon="INFO")
        layout.operator("eventcam.clear_cloud", text="Clear Cloud",
                        icon="TRASH")


# ═══════════════════ registration ════════════════════════════════════════════

CLASSES = [
    EVENTCAM_OT_live_cloud,
    EVENTCAM_OT_clear_cloud,
    EVENTCAM_PT_panel,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    global _running
    _running = False
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


# ─── auto-start when run as a script (not as an addon) ───────────────────────
if __name__ == "__main__":
    register()
    # Immediately invoke the operator so the server starts right away
    bpy.ops.eventcam.live_cloud("INVOKE_DEFAULT")
    print(f"[Blender Bridge] Ready.  Waiting for event_video.py on port {PORT} …")
