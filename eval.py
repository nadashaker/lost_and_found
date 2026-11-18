"""Measure search accuracy (Recall@K) against hand-labelled ground truth, with each upgrade on and off.

Run after index_videos.py:  python eval.py
Writes a report to eval/results.md
"""

import json
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from common import HERE, INDEX_DIR, score_items
from store import load_index

GT_FILE = os.path.join(HERE, "eval", "ground_truth.json")
# An index built elsewhere (LF_INDEX_DIR, for comparing variants) gets its report next to it
REPORT_FILE = (os.path.join(INDEX_DIR, "results.md") if "LF_INDEX_DIR" in os.environ
               else os.path.join(HERE, "eval", "results.md"))
KS = (1, 5, 10)

MODEL_CONFIGS = [  # text search, no boosts, prompt ensemble: which CLIP model / crop / view aggregation
    ("ViT-B/32, masked crop (old model)", {"clip_key": "clip_b32"}),
    ("ViT-L/14, masked crop", {"clip_key": "clip"}),
    ("ViT-L/14, context crop", {"clip_key": "clip_plain"}),
    ("ViT-L/14, masked + context fused", {"clip_key": "clip_fused"}),
    ("ViT-L/14, fused, best view (max)", {"clip_key": "clip_fused", "view_agg": "max"}),
    ("ViT-L/14, fused, mean + best view", {"clip_key": "clip_fused", "view_agg": "mix"}),
]
MODEL_CONFIGS = [(label, {"use_category": False, "use_color": False, "prompt_ensemble": True, "view_agg": "mean"} | cfg)
                 for label, cfg in MODEL_CONFIGS]

MODEL_CONFIGS += [("ViT-L/14, context crop, best view (app default)", {"clip_key": "clip_plain", "view_agg": "max",
                                                                        "use_category": False, "use_color": False,
                                                                        "prompt_ensemble": True})]

TEXT_CONFIGS = [  # tweaks on top of the app default (ViT-L/14, context crop, best view, prompt ensemble)
    ("single prompt template", {"prompt_ensemble": False}),
    ("+ colour boost", {"use_color": True}),
    ("+ category + colour boost", {"use_category": True, "use_color": True}),
    ("app default", {}),
]
PHOTO_CONFIGS = [
    ("CLIP", {"image_models": ("clip",)}),
    ("DINOv2", {"image_models": ("dino",)}),
    ("CLIP + DINOv2, averaged vectors", {"image_models": ("clip", "dino"), "view_agg": "mean"}),
    ("CLIP + DINOv2 (app default)", {"image_models": ("clip", "dino")}),
]
DETECTOR_SETS = [
    ("YOLO-seg only", {"yolo-seg"}),
    ("+ YOLO-World", {"yolo-seg", "yolo-world"}),
    ("+ static-object detector", {"yolo-seg", "yolo-world", "static"}),
]


def matches(item, target, sources=None, slack=0.6):
    """Does this indexed item contain the target point at a time the target is visible?

    Moving targets are labelled at one moment, which rarely lines up with a sampled frame, so their
    margin grows with the box size (the item moves a bit between samples).
    """
    if item["camera"] != target["camera"]:
        return False
    (px, py), (t0, t1) = target["point"], target["t"]
    moving = t1 - t0 < 1
    for t, x1, y1, x2, y2, source in item["sightings"]:
        if sources and source not in sources:
            continue
        mx, my = (max(15, (x2 - x1) / 4), max(15, (y2 - y1) / 4)) if moving else (15, 15)
        if t0 - slack <= t <= t1 + slack and x1 - mx <= px <= x2 + mx and y1 - my <= py <= y2 + my:
            return True
    return False


def rank_of(scores, correct):
    """1-based rank of the best-ranked correct item, or None if no correct item is in the index."""
    if not correct.any():
        return None
    order = np.argsort(-scores)
    return int(np.where(correct[order])[0][0]) + 1


def query_photo(target, video_of_camera, augment=True):
    """Cut the target out of its video, then alter it (flip, lower resolution, brighter) to act like a new photo."""
    cap = cv2.VideoCapture(video_of_camera[target["camera"]])
    cap.set(cv2.CAP_PROP_POS_MSEC, target["photo"]["t"] * 1000)
    ok, frame = cap.read()
    cap.release()
    x1, y1, x2, y2 = target["photo"]["box"]
    px, py = (x2 - x1) // 5, (y2 - y1) // 5
    crop = frame[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if augment:
        w, h = img.size
        img = img.transpose(Image.FLIP_LEFT_RIGHT).resize((max(8, int(w * 0.6)), max(8, int(h * 0.6)))).resize((w, h))
        img = ImageEnhance.Brightness(img).enhance(1.2)
    return img


def summarize(ranks):
    found = [r for r in ranks if r is not None]
    row = [f"{sum(r is not None and r <= k for r in ranks) / len(ranks):.0%}" for k in KS]
    mrr = sum(1 / r for r in found) / len(ranks)
    return row + [f"{mrr:.2f}"]


def table(title, header, rows):
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main():
    vecs, items = load_index()
    gt = json.load(open(GT_FILE))
    targets = {t["name"]: t for t in gt["targets"]}
    for group in gt["same_item"]:  # the same physical item filmed by several cameras: any of them is correct
        for name in group:
            targets[name]["also"] = [n for n in group if n != name]
    video_of_camera = {it["camera"]: it["video"] for it in items}

    def correct_mask(target, include_same=True):
        names = [target["name"]] + (target.get("also", []) if include_same else [])
        return np.array([any(matches(it, targets[n]) for n in names) for it in items])

    report = [f"# Lost & found search evaluation\n",
              f"Gallery: **{len(items)} indexed items** from {len(video_of_camera)} cameras. "
              f"Targets: {len(targets)} hand-labelled items.\n"]

    # 1. Detection coverage: is the item in the index at all? (an upper bound for any search)
    rows = []
    for label, sources in DETECTOR_SETS:
        hit = [any(matches(it, t, sources) for it in items) for t in targets.values()]
        rows.append([label, f"{sum(hit)}/{len(hit)}"])
    report.append(table("Detection coverage (target appears in the index)", ["Detectors", "Targets indexed"], rows))
    missing = [t["name"] for t in targets.values() if not correct_mask(t, include_same=False).any()]
    report.append(f"Not indexed: {', '.join(missing) or 'none'}\n")

    # 2. Text search
    text_ranks = {}
    rows = []
    for label, cfg in MODEL_CONFIGS:
        ranks = [rank_of(score_items(vecs, items, text=t["text"], config=cfg), correct_mask(t)) for t in targets.values()]
        rows.append([label] + summarize(ranks))
    report.append(table("Text search: embedding model, crop and view aggregation (no boosts)",
                        ["Setting", "R@1", "R@5", "R@10", "MRR"], rows))
    rows = []
    for label, cfg in TEXT_CONFIGS:
        ranks = [rank_of(score_items(vecs, items, text=t["text"], config=cfg), correct_mask(t)) for t in targets.values()]
        text_ranks[label] = ranks
        rows.append([label] + summarize(ranks))
    report.append(table(f"Text search ({len(targets)} descriptions)", ["Setting", "R@1", "R@5", "R@10", "MRR"], rows))

    # 3. Photo search (same-camera photos are altered crops; cross-camera is a real second view)
    photo_queries = [(t["name"], query_photo(t, video_of_camera), correct_mask(t))
                     for t in targets.values() if "photo" in t]
    for q in gt["cross_camera_photo"]:
        photo_queries.append((f"{q['photo_of']} -> {q['find']}", query_photo(targets[q["photo_of"]], video_of_camera, False),
                              correct_mask(targets[q["find"]], include_same=False)))
    rows, photo_ranks = [], {}
    for label, cfg in PHOTO_CONFIGS:
        ranks = [rank_of(score_items(vecs, items, image=img, config=cfg), c) for _, img, c in photo_queries]
        photo_ranks[label] = ranks
        rows.append([label] + summarize(ranks))
    report.append(table(f"Photo search ({len(photo_queries)} photos)", ["Setting", "R@1", "R@5", "R@10", "MRR"], rows))

    # 4. Per-query ranks with the app's default settings
    default_text, default_photo = TEXT_CONFIGS[-1][0], PHOTO_CONFIGS[-1][0]
    rows = [[t["name"], f'"{t["text"]}"', r if r else "not indexed"]
            for t, r in zip(targets.values(), text_ranks[default_text])]
    report.append(table("Text queries (app default)", ["Target", "Query", "Rank"], rows))
    rows = [[name, r if r else "not indexed"] for (name, _, _), r in zip(photo_queries, photo_ranks[default_photo])]
    report.append(table("Photo queries (app default)", ["Photo of", "Rank"], rows))

    text = "\n".join(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
