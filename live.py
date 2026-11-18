"""Live side: index cameras continuously, record them for playback, and keep a live snapshot of each.

Cameras are listed in cameras.json (see cameras.example.json):
    [{"name": "lobby",  "source": "rtsp://user:pass@192.168.1.20:554/stream1"},
     {"name": "webcam", "source": 0},
     {"name": "demo",   "source": "videos/4750061-hd_1920_1080_30fps.mp4", "loop": true}]
A relative path is relative to this folder.
A video file is played at its real speed, as if it were a camera (handy for testing). A looped file is indexed on
its first pass only (replays would add the same items again), and its items from earlier runs are replaced. That first
pass plays as fast as the indexer can process SAMPLE_FPS frames per video second (slower than real time on a busy
GPU), so the whole video is indexed like index_videos.py does, instead of the few frames a real-time pass gets.
Optional per camera: "static": false turns off the static-object detector (left-item finder). It is meant for
fixed cameras; on moving or very crowded footage it mostly finds background.

Each camera gets a reader thread that keeps only its newest frame, so a busy GPU skips frames instead of
falling behind. The main loop runs the same pipeline as index_videos.py on each camera's newest frame (up to
SAMPLE_FPS per camera) and writes items to the index every few seconds while they are still in view, so the
app can find them almost immediately.

Run:  python live.py      (Ctrl+C to stop; items in view are saved first)
"""

import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # RTSP over TCP: no smeared frames

import cv2
from ultralytics import YOLO

from common import (GROUP_CLIP_MODEL, resolve_source, HERE, LIVE_DIR, RECORD_FPS, RECORD_WIDTH, RECORDINGS_DIR, RETENTION_HOURS,
                    SEGMENT_SECONDS, YOLO_SEG_WEIGHTS, load_clip, load_dino, load_world)
from index_videos import SAMPLE_FPS, CameraIndexer
from store import Store

try:
    import imageio_ffmpeg  # pip install imageio-ffmpeg: a bundled ffmpeg, for small H.264 recordings
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except (ImportError, RuntimeError):
    FFMPEG = None

CAMERAS_FILE = os.path.join(HERE, "cameras.json")
FLUSH_SECONDS = 3      # how often items in view are written to the index (= how soon the app can find them)
SNAPSHOT_WIDTH = 640


class Camera(threading.Thread):
    """Reads one camera continuously, keeping only the newest frame. Reconnects when the stream drops."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.camera = cfg["name"]
        self.source = resolve_source(cfg["source"])  # "0" / 0 -> webcam 0; relative paths -> this folder
        self.is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        self.loop = cfg.get("loop", True)
        self.replaying = False  # a looped file has played through once: show it, but don't index it again
        self.first_pass_boxes = {}  # looped file: frame position -> boxes drawn on the first pass, reused on replays
        self.frame, self.frame_time, self.frame_no, self.pos = None, 0.0, 0, 0
        self.connected, self.error = False, "connecting"
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.indexed = threading.Event()  # first pass of a file: set by the main loop when the frame is processed
        self.indexed.set()

    def latest(self):
        with self.lock:
            return self.frame, self.frame_time, self.frame_no

    @property
    def first_pass(self):
        return self.is_file and not self.replaying

    def position(self):
        """Frame number within a video file (where the newest frame is in the video)."""
        with self.lock:
            return self.pos

    def replay_boxes(self, pos, fps=25):
        """On a replay: the first pass's boxes from the nearest processed frame (within ~half a second)."""
        if not self.first_pass_boxes:
            return []
        near = min(self.first_pass_boxes, key=lambda p: abs(p - pos))
        return self.first_pass_boxes[near] if abs(near - pos) <= fps / 2 else []

    def run(self):
        backoff = 1
        while not self.stopped.is_set():
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.connected, self.error = False, f"cannot open source, retrying in {backoff}s"
                self.stopped.wait(backoff)
                backoff = min(30, backoff * 2)
                continue
            backoff, self.connected, self.error = 1, True, ""
            fps, start, n = cap.get(cv2.CAP_PROP_FPS) or 25, time.time(), 0
            stride = max(1, round(fps / SAMPLE_FPS))
            while not self.stopped.is_set():
                ok, frame = cap.read()
                if not ok and self.is_file and self.loop:  # rewind without "disconnecting" (keeps the recording going)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.replaying = True
                    start, n = time.time(), 0
                    ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame, self.frame_time, self.frame_no, self.pos = frame, time.time(), self.frame_no + 1, n
                if self.first_pass:  # wait until this sampled frame is indexed, then skip to the next one
                    self.indexed.clear()
                    while not self.indexed.wait(0.1) and not self.stopped.is_set():
                        pass
                    for _ in range(stride - 1):
                        cap.grab()
                    n += stride
                    start = time.time() - n / fps  # real-time playback resumes from here on the replays
                elif self.is_file:  # play files at their real speed, like a camera
                    n += 1
                    self.stopped.wait(max(0.0, start + n / fps - time.time()))
            cap.release()
            self.connected = False
            if self.is_file:
                self.error = "video file ended"
                return
            self.error = "stream lost, reconnecting"
            self.stopped.wait(1)


class SegmentWriter:
    """One recording segment. H.264 through ffmpeg as fragmented MP4: ~4x smaller than OpenCV's mp4v, and
    playable while it is still being written (all but the last ~2 s). Falls back to OpenCV if ffmpeg is missing."""

    def __init__(self, path, size):
        self.proc = self.cv = None
        if FFMPEG:
            w, h = size
            self.proc = subprocess.Popen(
                [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", str(RECORD_FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                 "-crf", "30", "-g", str(2 * RECORD_FPS), "-pix_fmt", "yuv420p", "-flush_packets", "1",
                 "-movflags", "+frag_keyframe+empty_moov+default_base_moof", path],
                stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, size)

    def write(self, frame):
        if self.cv is not None:
            self.cv.write(frame)
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except OSError:
            pass  # ffmpeg died; this segment just ends early

    def release(self):
        if self.cv is not None:
            self.cv.release()
        else:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


class Recorder(threading.Thread):
    """Records a camera into SEGMENT_SECONDS-long MP4 files named by their start time (Unix ms), at exactly
    RECORD_FPS, so the app can find the frame for any moment. Deletes segments older than RETENTION_HOURS."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.folder = os.path.join(RECORDINGS_DIR, cam.camera)
        os.makedirs(self.folder, exist_ok=True)
        self.writer, self.seg_start, self.written = None, 0.0, 0

    def run(self):
        while not self.cam.stopped.is_set():
            frame, _, _ = self.cam.latest()
            now = time.time()
            if frame is None or not self.cam.connected:
                self.close()  # a gap in the footage: start a new segment when the camera is back
            else:
                if self.writer is None or now - self.seg_start >= SEGMENT_SECONDS:
                    self.open(frame, now)
                small = cv2.resize(frame, self.size)
                # Write as many frames as the clock says are due, so frame k is always at seg_start + k / RECORD_FPS
                due = int((now - self.seg_start) * RECORD_FPS) + 1
                while self.written < due:
                    self.writer.write(small)
                    self.written += 1
            self.cam.stopped.wait(1 / RECORD_FPS)
        self.close()

    def open(self, frame, now):
        self.close()
        self.cleanup(now)
        h, w = frame.shape[:2]
        scale = min(1.0, RECORD_WIDTH / w)
        self.size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
        path = os.path.join(self.folder, f"{int(now * 1000)}.mp4")
        self.writer = SegmentWriter(path, self.size)
        self.seg_start, self.written = now, 0

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def cleanup(self, now):
        for name in os.listdir(self.folder):
            stem = os.path.splitext(name)[0]
            if stem.isdigit() and int(stem) / 1000 < now - RETENTION_HOURS * 3600:
                try:
                    os.remove(os.path.join(self.folder, name))
                except OSError:
                    pass  # being read by the app right now; try again next segment


def write_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: the app is reading the old file right now; skip this update
        pass


def boxes_in_view(indexer, t):
    """[(x1, y1, x2, y2, label)] of the items seen in the frame processed at time t."""
    return [(*it["last_box"], max(it["votes"], key=it["votes"].get)) for it in indexer.items if it["last_seen"] == t]


def write_snapshot(cam, frame, t, boxes):
    """The camera's latest frame with the items in view boxed, for the app's live view."""
    scale = SNAPSHOT_WIDTH / frame.shape[1]
    img = cv2.resize(frame, (SNAPSHOT_WIDTH, int(frame.shape[0] * scale)))
    for *box, label in sorted(boxes, key=lambda b: b[4] != "person"):  # people first, items drawn on top
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        if label == "person":  # thin green, unlabelled: there are many and they'd hide the item labels
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 80), 1)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, datetime.fromtimestamp(t).strftime("%H:%M:%S"), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    write_atomic(os.path.join(LIVE_DIR, cam.camera + ".jpg"), cv2.imencode(".jpg", img)[1].tobytes())


def main():
    if not os.path.exists(CAMERAS_FILE):
        raise SystemExit(f"No {CAMERAS_FILE}. Copy cameras.example.json to cameras.json and list your cameras.")
    with open(CAMERAS_FILE) as f:
        cfgs = json.load(f)
    names = [c["name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise SystemExit("Camera names in cameras.json must be unique.")
    os.makedirs(LIVE_DIR, exist_ok=True)
    try:  # a run that was killed (not stopped with Ctrl+C) leaves its status behind; don't show it as live
        os.remove(os.path.join(LIVE_DIR, "status.json"))
    except OSError:
        pass

    def stop(*_):  # Ctrl+Break / termination: stop cleanly like Ctrl+C, saving items in view
        raise KeyboardInterrupt
    for sig in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), stop)
    if FFMPEG is None:
        print("imageio-ffmpeg not installed: recording with OpenCV mp4v (about 4x larger files)", flush=True)

    print("Loading models...", flush=True)
    store = Store()
    seg, world = YOLO(YOLO_SEG_WEIGHTS), load_world()
    load_clip(), load_clip(GROUP_CLIP_MODEL), load_dino()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # item ids stay unique across restarts
    cams = [Camera(c) for c in cfgs]
    for cam in cams:
        if cam.is_file:  # a video file shows the same items every run: replace the last run's
            store.delete_camera(cam.camera)
    indexers = {c["name"]: CameraIndexer(c["name"], store, seg, world, id_prefix=f"{c['name']}_{stamp}",
                                         use_static=c.get("static", True)) for c in cfgs}
    recorders = [Recorder(cam) for cam in cams]
    for th in cams + recorders:
        th.start()
    print(f"Watching {len(cams)} camera(s): {', '.join(names)}. Ctrl+C to stop.", flush=True)

    last_no = {cam.camera: 0 for cam in cams}
    last_t = {cam.camera: 0.0 for cam in cams}
    done = {cam.camera: [] for cam in cams}  # times of processed frames, for the status page
    last_flush = time.time()
    try:
        while True:
            busy = False
            for cam in cams:  # round-robin over cameras: each one's newest frame, at most SAMPLE_FPS per camera
                frame, t, no = cam.latest()
                # (a file's first pass only delivers sampled frames, and waits for each one: no rate limit)
                if frame is None or no == last_no[cam.camera] or (
                        not cam.first_pass and t - last_t[cam.camera] < 1 / SAMPLE_FPS):
                    continue
                last_no[cam.camera], last_t[cam.camera] = no, t
                pos = cam.position()
                if cam.replaying:  # looped file, already indexed: keep its live view moving, with the first pass's boxes
                    write_snapshot(cam, frame, t, cam.replay_boxes(pos))
                    continue
                indexers[cam.camera].process(frame, t)
                boxes = boxes_in_view(indexers[cam.camera], t)
                if cam.is_file:
                    cam.first_pass_boxes[pos] = boxes
                write_snapshot(cam, frame, t, boxes)
                cam.indexed.set()
                done[cam.camera] = [x for x in done[cam.camera] if x > t - 10] + [t]
                busy = True

            now = time.time()
            if now - last_flush >= FLUSH_SECONDS:
                for ix in indexers.values():
                    ix.flush(now)
                status = {cam.camera: {"connected": cam.connected,
                                       "error": cam.error or ("replaying (already indexed)" if cam.replaying else ""), "last_frame": last_t[cam.camera],
                                       "processed_fps": round(len(done[cam.camera]) / 10, 1),
                                       "items_in_view": len(indexers[cam.camera].items)} for cam in cams}
                write_atomic(os.path.join(LIVE_DIR, "status.json"), json.dumps(status).encode())
                last_flush = now
            if not busy:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping: saving items in view...", flush=True)
    finally:
        for cam in cams:
            cam.stopped.set()
        for ix in indexers.values():
            ix.flush()
        for rec in recorders:
            rec.join(timeout=5)
        if os.path.exists(os.path.join(LIVE_DIR, "status.json")):
            os.remove(os.path.join(LIVE_DIR, "status.json"))


if __name__ == "__main__":
    main()
