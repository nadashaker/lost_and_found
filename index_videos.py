"""Camera side: video -> sample frames -> detect -> crop (+mask) -> embed (CLIP + DINOv2) -> index.

People are indexed too (YOLO-seg's person class) and searched separately (the app's People search).

Three detectors feed the index:
  - YOLO-seg: COCO lost-item classes, with masks for clean crops
  - YOLO-World: open vocabulary (wallet, keys, shopping bag, ...) for what COCO doesn't cover
  - Static-object detector: background subtraction finds things that appeared and stayed still (left items),
    which works on the small, top-down objects in CCTV that both YOLOs miss. CLIP checks each region is an item.

Each item is stored ONCE (not once per frame) with its camera, first/last seen time, every sighting's
box, a thumbnail and a context frame. Sightings are grouped into items when:
  - they are in the same place as an item seen recently and look similar (stationary/left items), or
  - they look the same to CLIP and were seen within a few seconds (moving items in crowds).

Run:  python index_videos.py                (all videos in VIDEO_DIRS)
      python index_videos.py path/to/a.mp4  (specific videos)
"""

import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from common import (BAGS, COLORS, DETECT_IMGSZ, GROUP_CLIP_MODEL, INDEX_DIR, LOST_ITEM_CLASSES, PERSON, VIDEO_DIRS,
                    VIDEO_EXTS, WORLD_CLASSES, YOLO_SEG_WEIGHTS, color_probs, embed_images,
                    embed_images_dino, embed_texts, load_world)
from store import MEAN_KEYS, VIEW_KEYS, Store

SAMPLE_FPS = 2        # frames per second to process (the fps.py idea)
SEG_CONF = 0.35       # YOLO-seg confidence threshold
WORLD_CONF = 0.12     # YOLO-World scores are lower for the same quality of box
MIN_SIZE = 16         # ignore boxes smaller than this many pixels per side
PERSON_CONF = 0.35    # people: same threshold as items (0.5 kept only ~half the people YOLO finds in a crowd)
PERSON_MIN_HEIGHT = 48  # people shorter than this (pixels) are too small to describe
MERGE_GAP = 3.0       # moving items: join an item last seen at most this long ago...
MERGE_SIM = 0.85      # ...if (small) CLIP similarity is at least this...
# Grouping trade-off, measured 2026-11-12 on eval/ground_truth.json (50 targets): to the small CLIP many bags look
# alike (it scores 35% of different-object pairs >= 0.85), so similarity alone merged different bags into one item.
# The motion check below stops the worst of that. Stricter matching (DINOv2 0.6/0.7) or 4 fps sampling didn't help:
# items split into fragments, and single-sighting fragments are dropped (MIN_HITS), losing targets.
MOVE_BASE = 0.5       # ...and its centre moved at most (MOVE_BASE + MOVE_PER_SECOND * gap) x its size...
MOVE_PER_SECOND = 2.0
MAX_AREA_RATIO = 4.0  # ...and its box area changed less than this factor
STATIC_GAP = 30.0     # stationary items: join an item in the same place last seen this long ago...
STATIC_IOU = 0.4      # ...if the boxes overlap at least this much...
STATIC_SIM = 0.7      # ...and CLIP similarity is at least this
MIN_HITS = 2          # keep items seen at least this many times...
KEEP_CONF = 0.6       # ...or seen once with at least this confidence
STILL_SECONDS = 4     # static detector: a region must stay still this long to count as a left object
STATIC_ITEM_PROB = 0.4  # ...CLIP must give the item labels at least this much probability (vs. background)...
STATIC_MIN_TEXTURE = 15  # ...and the crop can't be a plain surface (grey-level std; lighting changes on walls)
STATIC_WIDTH = 640    # static detector works on frames downscaled to this width (speed)
PLAIN_PAD = 0.3       # context around the plain crop (the masked crop keeps a tight 0.1)
TOP_VIEWS = 5         # best-quality sightings stored per item, so search can match its sharpest view
PERSON_VIEWS = 3      # people: 3 views that look different (front / side / back), not the 5 sharpest of one pose.
PERSON_VIEW_SIM = 0.9  # A sighting this similar (small CLIP) to a stored view is the same pose: it only replaces
                       # that view if sharper. Saves most of the ViT-L/14 + DINOv2 work on people, who are in view long.
THUMB_SIZE = 224
CONTEXT_WIDTH = 640


def pad_box(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))


def masked_crop(frame, box, polygon, pad=0.1):
    """Crop the item twice: plain with some surrounding context (CLIP recognises small items better
    in context), and tight with the background greyed out when there's a segmentation mask."""
    cx1, cy1, cx2, cy2 = pad_box(frame, box, PLAIN_PAD)
    context = frame[cy1:cy2, cx1:cx2]
    x1, y1, x2, y2 = pad_box(frame, box, pad)
    plain = frame[y1:y2, x1:x2]
    if polygon is None or len(polygon) < 3:
        return context, plain
    mask = np.zeros(plain.shape[:2], np.uint8)
    cv2.fillPoly(mask, [(polygon - [x1, y1]).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    masked = np.full_like(plain, 127)
    masked[mask > 0] = plain[mask > 0]
    return context, masked


def context_frame(frame, box, label):
    scale = CONTEXT_WIDTH / frame.shape[1]
    img = cv2.resize(frame, (CONTEXT_WIDTH, int(frame.shape[0] * scale)))
    x1, y1, x2, y2 = (int(v * scale) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def overlap(a, b):
    """Fraction of box a that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-9)


class StaticDetector:
    """Finds regions that differ from the long-term background but have stopped moving (dual background model)."""

    def __init__(self):
        self.long_bg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=25, detectShadows=False)
        self.short_bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=False)
        self.still_count = None
        self.frames = 0

    def update(self, frame, person_boxes):
        """Returns boxes (full-frame pixels) of regions that appeared and have stayed still for STILL_SECONDS.
        Works on a frame downscaled to STATIC_WIDTH: on full HD the background models took ~190 ms a frame."""
        s = min(1.0, STATIC_WIDTH / frame.shape[1])
        small = cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return [box / s for box in self._update(small, [b * s for b in person_boxes], s)]

    def _update(self, frame, person_boxes, s):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        warming_up = self.frames < 5 * SAMPLE_FPS  # learn the empty scene quickly at first
        fg_long = self.long_bg.apply(blurred, learningRate=0.2 if warming_up else 0.0008)
        fg_short = self.short_bg.apply(blurred, learningRate=0.05)
        self.frames += 1

        static = (fg_long > 0) & (fg_short == 0)  # new compared to the scene, but not moving now
        for x1, y1, x2, y2 in person_boxes:       # people standing still are not left objects
            static[int(y1):int(y2), int(x1):int(x2)] = False
        if self.still_count is None:
            self.still_count = np.zeros(static.shape, np.int32)
        self.still_count = np.where(static, self.still_count + 1, 0)

        mask = (self.still_count >= STILL_SECONDS * SAMPLE_FPS).astype(np.uint8) * 255
        close = max(3, round(9 * s) // 2 * 2 + 1)  # kernel sizes and the minimum area were set for full-size frames
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        h, w = mask.shape
        return [np.array([x, y, x + bw, y + bh], dtype=np.float32) for x, y, bw, bh, area in stats[1:]
                if 150 * s * s <= area <= 0.05 * h * w and bw < w / 3 and bh < h / 3]


# CLIP zero-shot check for static regions: lost item vs. part of the scene
BACKGROUND_PROMPTS = ["floor", "tiled floor", "wall", "shadow", "reflection of light", "chair", "table", "door",
                      "stairs", "person", "pillar", "window", "curtain", "plant", "trash can", "car",
                      "blank white surface", "table edge", "metal frame", "crowd of people", "blurry background",
                      "white wall", "bright light", "glare", "corner of a table", "people walking", "sign",
                      "row of chairs"]
_prompt_vecs = None


def classify_static(clip_vecs):
    """CLIP vectors of static-region crops -> [(category, item_probability)]."""
    global _prompt_vecs
    if _prompt_vecs is None:
        _prompt_vecs = embed_texts(WORLD_CLASSES + BACKGROUND_PROMPTS, model=GROUP_CLIP_MODEL)
    logits = 100 * clip_vecs @ _prompt_vecs.T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    n = len(WORLD_CLASSES)
    return [(WORLD_CLASSES[int(row[:n].argmax())], float(row[:n].sum())) for row in p]


def group_of(category):
    return "bag" if category in BAGS else category


def detect(seg, world, static_detector, frame):
    """Run all detectors. Returns [(box, category, conf, polygon, source)]; people have category PERSON.

    YOLO-seg boxes come with masks; YOLO-World and the static detector (optional: None) add what seg missed
    (boxes overlapping an earlier box are dropped). Static regions get their category from CLIP later.
    """
    # Plain detection, not seg.track(): with tracking on, Ultralytics only returns boxes the tracker has
    # confirmed, which silently drops most bags in crowded footage. find_item() does the grouping instead.
    r = seg.predict(frame, classes=[0] + list(LOST_ITEM_CLASSES), conf=SEG_CONF, agnostic_nms=True, imgsz=DETECT_IMGSZ,
                    verbose=False)[0]
    polygons = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    dets, persons, people = [], [], []
    for box, cls, conf, poly in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().tolist(),
                                    r.boxes.conf.tolist(), polygons):
        if cls == 0:
            persons.append(box)
            if conf >= PERSON_CONF and box[3] - box[1] >= PERSON_MIN_HEIGHT:
                people.append((box, PERSON, conf, poly, "yolo-seg"))
        else:
            dets.append((box, LOST_ITEM_CLASSES[cls], conf, poly, "yolo-seg"))

    w = world.predict(frame, conf=WORLD_CONF, agnostic_nms=True, imgsz=DETECT_IMGSZ, verbose=False)[0]
    for box, cls, conf in zip(w.boxes.xyxy.cpu().numpy(), w.boxes.cls.int().tolist(), w.boxes.conf.tolist()):
        if all(iou(box, d[0]) < 0.5 for d in dets):
            dets.append((box, WORLD_CLASSES[cls], conf, None, "yolo-world"))

    for box in static_detector.update(frame, persons) if static_detector else []:
        if all(overlap(box, d[0]) < 0.5 for d in dets):  # skip regions mostly covered by a YOLO box
            dets.append((box, None, 0.0, None, "static"))
    dets = [d for d in dets if min(d[0][2] - d[0][0], d[0][3] - d[0][1]) >= MIN_SIZE]
    return dets + people  # people added last: the overlap checks above compare items with items only


def find_item(items, category, t, box, vec, taken):
    """Return the existing item this sighting belongs to, or None for a new item."""
    best, best_sim = None, 0
    for it in items:
        gap = t - it["last_seen"]
        if id(it) in taken or gap > STATIC_GAP:
            continue
        sim = float(it["clip_b32_sum"] / np.linalg.norm(it["clip_b32_sum"]) @ vec)
        if (it["group"] == PERSON) != (category == PERSON):  # people only group with people
            continue
        # Same place + similar look: same item even if the label flips (CLIP/YOLO labels are noisy on small crops)
        same_place = iou(box, it["last_box"]) >= STATIC_IOU and sim >= STATIC_SIM
        # Similar look alone isn't enough (to CLIP many bags look alike): it must also be a plausible move
        looks_same = (gap <= MERGE_GAP and sim >= MERGE_SIM and it["group"] == group_of(category)
                      and plausible_move(it["last_box"], box, gap))
        if (same_place or looks_same) and sim > best_sim:
            best, best_sim = it, sim
    return best


def plausible_move(last, box, gap):
    """Could an item seen at `last` be at `box` `gap` seconds later? Its centre may move a few of its own sizes
    per second (a carried bag), and its size may change, but not jump across the frame or grow several-fold."""
    lw, lh, w, h = last[2] - last[0], last[3] - last[1], box[2] - box[0], box[3] - box[1]
    dist = np.hypot((box[0] + box[2] - last[0] - last[2]) / 2, (box[1] + box[3] - last[1] - last[3]) / 2)
    area_ratio = (w * h) / max(lw * lh, 1e-6)
    return (dist <= max(lw, lh, w, h) * (MOVE_BASE + MOVE_PER_SECOND * gap)
            and 1 / MAX_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO)


def view_slot(it, quality, b32):
    """Should this sighting be embedded as one of the item's stored views? Returns (yes, index of the stored view
    it would replace, or None to add it). Items keep their TOP_VIEWS sharpest views; people keep PERSON_VIEWS
    views that look different from each other."""
    views = it["views"]
    if it["group"] != PERSON:
        return len(views) < TOP_VIEWS or quality > views[-1][0], None
    if views:
        sims = [float(view["b32"] @ b32) for _, view in views]
        j = int(np.argmax(sims))
        if sims[j] >= PERSON_VIEW_SIM:  # a pose we already have: only a sharper copy of it is worth embedding
            return quality > views[j][0], j
    return len(views) < PERSON_VIEWS or quality > views[-1][0], None


class CameraIndexer:
    """Turns one camera's frames into items: detect, embed, group sightings, and write items to the store.

    Used for video files (index_video) and live cameras (live.py). Items are written while they are still being
    seen, so live items are searchable within seconds; items not seen for STATIC_GAP are closed and forgotten.
    """

    def __init__(self, camera, store, seg, world, video=None, id_prefix=None, use_static=True):
        self.camera, self.store, self.seg, self.world, self.video = camera, store, seg, world, video
        self.id_prefix = id_prefix or camera
        self.static_detector = StaticDetector() if use_static else None
        self.items = []    # open items (can still get new sightings)
        self.dirty = []    # open items changed since the last flush
        self.n_created = 0
        self.n_embedded = 0  # crops run through the big models (the slow step), for the timing printout
        self.n_written = set()

    def process(self, frame, t):
        """Run the pipeline on one sampled frame taken at time t (seconds)."""
        dets = detect(self.seg, self.world, self.static_detector, frame)
        if not dets:
            return
        crops = [masked_crop(frame, d[0], d[3]) for d in dets]
        b32 = embed_images([to_pil(m) for _, m in crops], model=GROUP_CLIP_MODEL)

        # Static regions: let the small CLIP name them, and drop the ones that look like background.
        # Done before the expensive embeddings: crowded scenes give ~80 static regions a frame, most rejected.
        keep = []
        for k, d in enumerate(dets):
            if d[4] == "static":
                category, item_prob = classify_static(b32[k:k + 1])[0]
                texture = cv2.cvtColor(crops[k][1], cv2.COLOR_BGR2GRAY).std()  # tight crop (no mask here)
                if item_prob < STATIC_ITEM_PROB or texture < STATIC_MIN_TEXTURE:
                    continue
                dets[k] = (d[0], category, item_prob * 0.5, None, "static")
            keep.append(k)
        if not keep:
            return
        dets = [dets[k] for k in keep]
        crops = [crops[k] for k in keep]
        b32 = b32[keep]

        # 1. Group sightings into items with the small CLIP, and see which sightings are worth embedding:
        #    search only uses each item's TOP_VIEWS best views, so a sighting that wouldn't make that list
        #    (most frames of an item that is in view for a while) skips the expensive ViT-L/14 + DINOv2 step.
        taken = set()  # one sighting per item per frame (ids of items)
        to_embed = []  # (k, item, quality, index of the stored view it replaces or None)
        for k, (box, category, conf, _, source) in enumerate(dets):
            it = find_item(self.items, category, t, box, b32[k], taken)
            if it is None:
                self.n_created += 1
                it = {"id": f"{self.id_prefix}_{self.n_created}", "group": group_of(category), "votes": {},
                      "sources": set(), "first_seen": t, "sightings": [], "views": [], "best_quality": -1,
                      **{f"{key}_sum": 0 for key in MEAN_KEYS}}
                self.items.append(it)
            taken.add(id(it))
            if not any(d is it for d in self.dirty):
                self.dirty.append(it)
            it["last_seen"] = t
            it["last_box"] = box
            it["sightings"].append([round(t, 2)] + [round(float(v)) for v in box] + [source])
            it["votes"][category] = it["votes"].get(category, 0) + conf
            it["sources"].add(source)
            it["clip_b32_sum"] = it["clip_b32_sum"] + b32[k]

            quality = conf * np.sqrt((box[2] - box[0]) * (box[3] - box[1]))  # confident, large views
            wanted, replace = view_slot(it, quality, b32[k])
            if wanted:
                to_embed.append((k, it, quality, replace))
            if quality > it["best_quality"]:
                it.update(best_quality=quality, conf=conf, box=box, best_time=t, thumb=crops[k][0].copy(),
                          frame=frame, frame_size=[frame.shape[1], frame.shape[0]], images_stale=True)

        # 2. Embed only those sightings (an item has at most one sighting per frame, so each one is decided once)
        if not to_embed:
            return
        idx = [k for k, _, _, _ in to_embed]
        self.n_embedded += len(idx)
        vecs = {"clip": embed_images([to_pil(crops[k][1]) for k in idx]),
                "clip_plain": embed_images([to_pil(crops[k][0]) for k in idx]),
                "dino": embed_images_dino([to_pil(crops[k][1]) for k in idx])}
        for j, (k, it, quality, replace) in enumerate(to_embed):
            for key in VIEW_KEYS:  # averaged vectors: over the embedded sightings (a superset of the top views)
                it[f"{key}_sum"] = it[f"{key}_sum"] + vecs[key][j]
            view = (quality, {key: vecs[key][j] for key in VIEW_KEYS} | {"b32": b32[k]})
            if replace is None:
                it["views"].append(view)
            else:
                it["views"][replace] = view
            limit = PERSON_VIEWS if it["group"] == PERSON else TOP_VIEWS
            it["views"] = sorted(it["views"], key=lambda v: -v[0])[:limit]

    def flush(self, now=None):
        """Write changed items to the store. With `now`, also close items that can't get new sightings anymore
        (not seen for STATIC_GAP); without it (end of a video), close everything."""
        rows = [self._row(it) for it in self.dirty if len(it["sightings"]) >= MIN_HITS or it["conf"] >= KEEP_CONF]
        self.store.put_many(rows)
        self.n_written.update(m["id"] for m, _ in rows)
        self.dirty = []
        self.items = [] if now is None else [it for it in self.items if now - it["last_seen"] <= STATIC_GAP]

    def _row(self, it):
        category = max(it["votes"], key=it["votes"].get)  # most confident label wins
        item_id = it["id"]
        if it.pop("images_stale", False):
            thumb = to_pil(it["thumb"])
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
            thumb.save(os.path.join(INDEX_DIR, "thumbs", item_id + ".jpg"), quality=85)
            cv2.imwrite(os.path.join(INDEX_DIR, "context", item_id + ".jpg"),
                        context_frame(it["frame"], it["box"], category), [cv2.IMWRITE_JPEG_QUALITY, 80])
        vectors = {key: it[f"{key}_sum"] / np.linalg.norm(it[f"{key}_sum"]) for key in MEAN_KEYS}
        for key in VIEW_KEYS:
            vectors[f"{key}_views"] = np.stack([view[key] for _, view in it["views"]])
        meta = {
            "id": item_id, "camera": self.camera, "video": self.video, "category": category,
            "color": COLORS[int(color_probs(vectors["clip"][None])[0].argmax())],
            "confidence": round(it["conf"], 3), "bbox": [round(float(v)) for v in it["box"]],
            "frame_size": it["frame_size"], "first_seen": round(it["first_seen"], 2),
            "last_seen": round(it["last_seen"], 2), "best_time": round(it["best_time"], 2),
            "n_sightings": len(it["sightings"]), "sources": sorted(it["sources"]), "sightings": it["sightings"],
            "thumb": f"thumbs/{item_id}.jpg", "context": f"context/{item_id}.jpg",
        }
        return meta, vectors


def camera_name(video_path):
    return os.path.basename(os.path.dirname(video_path)) + "_" + os.path.splitext(os.path.basename(video_path))[0]


def index_video(video_path, store, seg, world):
    """Index a whole video file (replacing that camera's earlier items). Returns (items written, crops embedded)."""
    camera = camera_name(video_path)
    store.delete_camera(camera)
    indexer = CameraIndexer(camera, store, seg, world, video=os.path.abspath(video_path))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps / SAMPLE_FPS))
    frame_no = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_no % stride == 0:
            indexer.process(frame, frame_no / fps)
        frame_no += 1
    cap.release()
    indexer.flush()
    return len(indexer.n_written), indexer.n_embedded


def main():
    videos = sys.argv[1:] or sorted(p for d in VIDEO_DIRS for p in glob.glob(os.path.join(d, "*"))
                                    if p.lower().endswith(VIDEO_EXTS))
    store = Store()
    seg = YOLO(YOLO_SEG_WEIGHTS)
    world = load_world()
    total = 0
    for v in videos:
        start = time.time()
        n, n_embedded = index_video(v, store, seg, world)
        total += n
        print(f"{os.path.basename(v):40s} {n:3d} items  {n_embedded:5d} crops embedded  ({time.time() - start:.0f}s)",
              flush=True)
    print(f"Done: {total} items from {len(videos)} video(s) -> {INDEX_DIR} ({store.count()} items in the index)")


if __name__ == "__main__":
    main()
