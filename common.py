"""Shared settings, models (CLIP, DINOv2) and search for the lost-and-found demo."""

import json
import os
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))


def _local_settings():
    """Optional local_settings.json next to the code (not in git): this machine's folders, e.g.
    {"data_dir": "D:/lost_and_found_data", "clip_dir": "...", "video_dirs": ["...", "..."]}"""
    try:
        with open(os.path.join(HERE, "local_settings.json"), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


LOCAL = _local_settings()
# Everything that takes disk space (models, index, live recordings, downloaded videos) lives in DATA_DIR.
# Default: ./data next to the code; LF_DATA_DIR or local_settings.json can put it on another drive.
DATA_DIR = os.environ.get("LF_DATA_DIR") or LOCAL.get("data_dir") or os.path.join(HERE, "data")
torch.hub.set_dir(os.path.join(DATA_DIR, "torch_hub"))  # DINOv2 code + weights

# Inputs: every video in these folders is treated as one camera (camera id = folder_file name).
# videos/ in the repo holds a few small samples; download_samples.py fetches the rest into DATA_DIR.
VIDEO_DIRS = LOCAL.get("video_dirs") or [os.path.join(HERE, "videos"), os.path.join(DATA_DIR, "videos"),
                                         os.path.join(DATA_DIR, "ObjectTracking")]
VIDEO_EXTS = (".mp4", ".avi")

# Models
# (LF_* environment variables override these, so detector variants can be indexed side by side)
MODELS_DIR = os.path.join(DATA_DIR, "models")  # YOLO weights download here on first use
os.makedirs(MODELS_DIR, exist_ok=True)
YOLO_SEG_WEIGHTS = os.path.join(MODELS_DIR, os.environ.get("LF_SEG", "yolov8n-seg.pt"))
YOLO_WORLD_WEIGHTS = os.path.join(MODELS_DIR, os.environ.get("LF_WORLD", "yolov8s-worldv2.pt"))
DETECT_IMGSZ = int(os.environ.get("LF_IMGSZ", 640))  # YOLO input size; larger finds smaller items, slower
# CLIP weights download here on first use (~1.2 GB). Must end in "clip": YOLO-World looks in <parent>/clip.
CLIP_DIR = LOCAL.get("clip_dir") or os.path.join(MODELS_DIR, "clip")
# Search model: bigger CLIP with 14px patches at 336px sees much more detail on small CCTV crops than B/32
CLIP_MODEL = "ViT-L/14@336px"
# Small CLIP used only while indexing, to group sightings into items and to filter static regions.
# The merge thresholds in index_videos.py were tuned on this model's similarity scale.
GROUP_CLIP_MODEL = "ViT-B/32"
# Which CLIP model produced each stored vector set (search must embed the query with the same model)
CLIP_KEY_MODEL = {"clip": CLIP_MODEL, "clip_plain": CLIP_MODEL, "clip_fused": CLIP_MODEL, "clip_b32": GROUP_CLIP_MODEL}
DINO_REPO = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")

# Outputs
INDEX_DIR = os.environ.get("LF_INDEX_DIR", os.path.join(DATA_DIR, "index"))  # index.db, thumbs/, context/, live/
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")  # live cameras' rolling video, for "Play clip"
LIVE_DIR = os.path.join(INDEX_DIR, "live")             # live.py's latest snapshot + status of each camera
RECORD_FPS = 8            # recordings: frames per second (enough to see who picked an item up)...
RECORD_WIDTH = 960        # ...downscaled to this width...
SEGMENT_SECONDS = 30      # ...in files this long...
RETENTION_HOURS = 12      # ...kept this long. H.264: ~1.5 GB per camera per 12 h for a quiet scene, up to ~3.5 GB
                          # for a busy one (measured: 0.8 MB per 10 s of a crowded street at 960 px)

# COCO classes that are typical lost items (YOLO-seg: gives masks)
LOST_ITEM_CLASSES = {
    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
    29: "frisbee", 32: "sports ball", 39: "bottle", 41: "cup", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 73: "book",
    74: "clock", 76: "scissors", 77: "teddy bear",
}
# Open-vocabulary classes (YOLO-World: finds items by name, incl. ones COCO doesn't have)
WORLD_CLASSES = [
    "backpack", "handbag", "suitcase", "bag", "shopping bag", "plastic bag", "umbrella", "wallet", "keys",
    "sunglasses", "water bottle", "cell phone", "laptop", "tablet", "book", "headphones", "teddy bear",
]
# YOLO mixes these up for the same object, so sightings may merge across them
BAGS = {"backpack", "handbag", "suitcase", "bag", "shopping bag", "plastic bag"}
# People are indexed too (YOLO-seg class 0), as their own kind: searched separately from lost items
PERSON = "person"

COLORS = ["black", "white", "gray", "silver", "red", "pink", "orange", "yellow", "green", "blue", "purple", "brown"]
COLOR_WORDS = {c: c for c in COLORS} | {"grey": "gray", "violet": "purple", "navy": "blue", "beige": "brown",
                                        "tan": "brown", "golden": "yellow", "gold": "yellow", "dark": "black"}

# Search weights (see eval.py for how each one changes Recall@K)
CATEGORY_WEIGHT = 0.05  # text query matches the detected category (e.g. "phone" -> cell phone)
COLOR_WEIGHT = 0.05     # text query names a colour the item has

device = "cuda" if torch.cuda.is_available() else "cpu"
_models = {}


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------- CLIP: text <-> image ----------

def load_clip(name=CLIP_MODEL):
    if name not in _models:
        model, preprocess = clip.load(name, device=device, download_root=CLIP_DIR)
        _models[name] = (model.eval(), preprocess)
    return _models[name]


@torch.no_grad()
def embed_images(pil_images, model=CLIP_MODEL, batch_size=64):
    """List of PIL images -> (N, D) L2-normalized CLIP vectors."""
    net, preprocess = load_clip(model)
    out = []
    for i in range(0, len(pil_images), batch_size):
        batch = torch.stack([preprocess(im) for im in pil_images[i:i + batch_size]]).to(device)
        out.append(net.encode_image(batch).float().cpu().numpy())
    return _normalize(np.concatenate(out))


@torch.no_grad()
def embed_texts(texts, template="a photo of a {}", model=CLIP_MODEL):
    """List of descriptions -> (N, D) L2-normalized CLIP vectors."""
    net, _ = load_clip(model)
    tokens = clip.tokenize([template.format(t) for t in texts], truncate=True).to(device)
    return _normalize(net.encode_text(tokens).float().cpu().numpy())


# Prompt ensemble for search queries: CCTV crops are small, low-res and cut out of a wider scene
TEXT_TEMPLATES = [
    "a photo of a {}",
    "a CCTV photo of a {}",
    "a low resolution photo of a {}",
    "a blurry photo of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a photo of a {} in a crowd",
    "a photo of a person carrying a {}",
]


# Prompt ensemble for people searches ("person carrying a {}" would be wrong here)
PERSON_TEMPLATES = [
    "a photo of a {}",
    "a CCTV photo of a {}",
    "a low resolution photo of a {}",
    "a blurry photo of a {}",
    "a photo of a {} in a crowd",
    "a full body photo of a {}",
]
PERSON_WORDS = {"person", "people", "man", "men", "woman", "women", "child", "children", "kid", "kids", "boy",
                "girl", "baby", "toddler", "lady", "guy", "adult", "teen", "teenager", "someone", "he", "she"}


def person_query(text):
    """'red jacket' -> 'person wearing red jacket'; queries that already describe a person are kept."""
    words = set(text.lower().replace(",", " ").split())
    return text if words & PERSON_WORDS else f"person wearing {text}"


def embed_text(text, ensemble=False, model=CLIP_MODEL, templates=TEXT_TEMPLATES):
    if not ensemble:
        return embed_texts([text], model=model)[0]
    vecs = np.stack([embed_texts([text], tpl, model)[0] for tpl in templates])
    return _normalize(vecs.mean(axis=0))


# ---------- DINOv2: image <-> image (fine visual detail) ----------

def load_dino():
    if "dino" not in _models:
        if os.path.isdir(DINO_REPO):  # downloaded before: no network needed
            model = torch.hub.load(DINO_REPO, "dinov2_vits14", source="local")
        else:  # first run: downloads the code and weights into DATA_DIR/torch_hub
            model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14", trust_repo=True)
        model = model.to(device).eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        _models["dino"] = (model, preprocess)
    return _models["dino"]


@torch.no_grad()
def embed_images_dino(pil_images):
    """List of PIL images -> (N, 384) L2-normalized DINOv2 vectors."""
    model, preprocess = load_dino()
    batch = torch.stack([preprocess(im) for im in pil_images]).to(device)
    return _normalize(model(batch).float().cpu().numpy())


# ---------- YOLO-World ----------

def load_world():
    """YOLO-World with our vocabulary. It encodes class names with CLIP; point it at the existing weights."""
    import ultralytics.nn.text_model as text_model
    from ultralytics import YOLO
    text_model.WEIGHTS_DIR = Path(CLIP_DIR).parent
    model = YOLO(YOLO_WORLD_WEIGHTS)
    model.set_classes(WORLD_CLASSES)
    return model


# ---------- Index + search ----------

def color_probs(clip_vectors):
    """(N, D) CLIP_MODEL image vectors -> (N, len(COLORS)) probability of each colour."""
    if "color_text" not in _models:
        _models["color_text"] = embed_texts(COLORS, "a photo of a {} object")
    logits = 100 * clip_vectors @ _models["color_text"].T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def item_sims(vecs, key, query_vec, agg="mean"):
    """Similarity of each item to a query vector.

    mean: against the item's averaged vector (all sightings)
    max:  against its best-matching stored view (the top-quality sightings)
    mix:  average of the two
    """
    mean_sim = vecs[key] @ query_vec
    if agg == "mean" or f"{key}_views" not in vecs:
        return mean_sim
    view_sim = vecs[f"{key}_views"] @ query_vec
    max_sim = np.full(len(mean_sim), -np.inf)
    np.maximum.at(max_sim, vecs["view_item"], view_sim)
    return max_sim if agg == "max" else (mean_sim + max_sim) / 2


_label_cache = {}


def category_probs(text_vec, categories, model=CLIP_MODEL):
    """For each item's category, how likely the text query is asking for that category (0..1)."""
    labels = sorted(set(categories))
    missing = [label for label in labels if (model, label) not in _label_cache]
    if missing:
        _label_cache.update(zip([(model, label) for label in missing], embed_texts(missing, model=model)))
    logits = 100 * np.array([_label_cache[(model, label)] @ text_vec for label in labels])  # CLIP's logit scale
    probs = np.exp(logits - logits.max())
    probs = dict(zip(labels, probs / probs.sum()))
    return np.array([probs[c] for c in categories])


def colors_in(text):
    words = text.lower().replace(",", " ").split()
    return sorted({COLOR_WORDS[w] for w in words if w in COLOR_WORDS})


def _zscore(x, keep=None):
    """Standardize x, using the statistics of x[keep] (the items being ranked) when given."""
    ref = x if keep is None else x[keep]
    return (x - ref.mean()) / (ref.std() + 1e-9) if len(ref) else x


# Default search settings; eval.py turns these on and off to measure each one
# (ViT-L/14 + context crop + best view: MRR 0.28 -> 0.47 on eval/ground_truth.json; the category and
# colour boosts no longer help with the bigger model, so they are off)
DEFAULT_CONFIG = {"clip_key": "clip_plain", "use_category": False, "use_color": False, "image_models": ("clip", "dino"),
                  "prompt_ensemble": True, "view_agg": "max"}
# People: the tight masked crop, averaged over sightings. The context crop of a person in a crowd is full of other
# people (2026-11-18, demo cameras: "red jacket" ranked the woman in the red jacket below people in brown and black)
PEOPLE_CONFIG = {"clip_key": "clip", "view_agg": "mean"}


def score_items(vecs, items, text=None, image=None, config=None, people=False):
    """Score every indexed item against a text description and/or a photo. Returns (N,) scores or None.
    Only one kind is ranked: lost items (people=False) or people (people=True, the text is phrased as a description
    of a person); the other kind scores -inf."""
    cfg = DEFAULT_CONFIG | (PEOPLE_CONFIG if people else {}) | (config or {})
    keep = np.array([(it["category"] == PERSON) == people for it in items], dtype=bool)
    key, model = cfg["clip_key"], CLIP_KEY_MODEL[cfg["clip_key"]]
    parts = []
    if text:
        if people:
            t = embed_text(person_query(text), ensemble=cfg["prompt_ensemble"], model=model,
                           templates=PERSON_TEMPLATES)
        else:
            t = embed_text(text, ensemble=cfg["prompt_ensemble"], model=model)
        s = item_sims(vecs, key, t, cfg["view_agg"])
        if cfg["use_category"]:
            s = s + CATEGORY_WEIGHT * category_probs(t, [it["category"] for it in items], model)
        if cfg["use_color"]:
            for c in colors_in(text):
                s = s + COLOR_WEIGHT * vecs["color"][:, COLORS.index(c)]
        parts.append(s)
    if image is not None:
        sims = []
        if "clip" in cfg["image_models"]:
            sims.append(item_sims(vecs, key, embed_images([image], model)[0], cfg["view_agg"]))
        if "dino" in cfg["image_models"]:
            sims.append(item_sims(vecs, "dino", embed_images_dino([image])[0], cfg["view_agg"]))
        parts.append(np.mean(sims, axis=0) if len(sims) == 1 else np.mean([_zscore(s, keep) for s in sims], axis=0))
    if not parts:
        return None
    # Text and photo scores live on different scales, so standardize before combining
    scores = parts[0] if len(parts) == 1 else np.mean([_zscore(p, keep) for p in parts], axis=0)
    return np.where(keep, scores, -np.inf)


def search(vecs, items, text=None, image=None, camera=None, category=None, color=None, top_k=12, config=None,
           skip_cameras=(), people=False):
    """Rank indexed items. Returns a list of (score, item), best first (score None when there's no query).
    people=False searches lost items, people=True searches people.
    Items from cameras in skip_cameras are left out."""
    scores = score_items(vecs, items, text, image, config, people=people)
    order = np.argsort(-scores) if scores is not None else np.argsort([-it["last_seen"] for it in items])
    # (score_items already gives the other kind -inf; the check below also covers the no-query listing)
    results = []
    for i in order:
        it = items[i]
        if it["camera"] in skip_cameras or (it["category"] == PERSON) != people:
            continue
        if camera and it["camera"] != camera:
            continue
        if category and it["category"] != category:
            continue
        if color and it["color"] != color:
            continue
        results.append((float(scores[i]) if scores is not None else None, it))
        if len(results) >= top_k:
            break
    return results


def similar_people(vecs, items, item_id, top_k=24, skip_cameras=()):
    """'Find this person on all cameras': rank every other person by how much they look like this one.

    Compares the person's stored views with every other person's views (best pair), with DINOv2 (fine visual detail:
    clothes, patterns) and CLIP (the overall look), standardized over the people and averaged, like a photo search
    with all of this person's views as the photo. Returns (query item, [(score, item)]) best first."""
    q = next((i for i, it in enumerate(items) if it["id"] == item_id), None)
    if q is None:
        return None, []
    people = np.array([it["category"] == PERSON and it["camera"] not in skip_cameras for it in items], dtype=bool)
    people[q] = False
    if not people.any():
        return items[q], []
    parts = []
    for key in ("dino", "clip"):
        views = vecs[f"{key}_views"]
        query_views = views[vecs["view_item"] == q]          # (V, D) this person's stored views
        best = (views @ query_views.T).max(axis=1)             # each view's best match with any query view
        sims = np.full(len(items), -np.inf)
        np.maximum.at(sims, vecs["view_item"], best)          # each item's best view pair
        parts.append(_zscore(sims, people))
    scores = np.where(people, np.mean(parts, axis=0), -np.inf)
    order = [i for i in np.argsort(-scores)[:top_k] if people[i]]
    return items[q], [(float(scores[i]), items[i]) for i in order]


def resolve_source(src):
    """A camera source from cameras.json: webcam number, stream URL, or a video path (relative to the project)."""
    if isinstance(src, int) or str(src).isdigit():
        return int(src)
    if "://" in str(src) or os.path.isabs(src):
        return src
    return os.path.join(HERE, src)


def fmt_time(seconds):
    """Video files use seconds from the start of the video; live cameras use wall-clock (Unix) time."""
    if seconds > 1e9:
        return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S")
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
