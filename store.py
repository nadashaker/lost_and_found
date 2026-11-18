"""The item index: a SQLite database shared by the indexers (video files, live cameras) and the search app.

Writers (index_videos.py, live.py) insert or update one row per item while it is being seen. Readers (app.py,
eval.py) keep an in-memory copy for fast search and call refresh() to pick up new and changed items, so the
app sees live items without restarting. SQLite in WAL mode lets one process write while others read.
"""

import io
import json
import os
import sqlite3

import numpy as np

from common import INDEX_DIR, _normalize, color_probs

DB_FILE = os.path.join(INDEX_DIR, "index.db")
MEAN_KEYS = ["clip", "clip_plain", "dino", "clip_b32"]  # one averaged vector per item
VIEW_KEYS = ["clip", "clip_plain", "dino"]              # plus up to TOP_VIEWS separate vectors per item


def _pack(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _unpack(blob):
    return dict(np.load(io.BytesIO(blob)))


class Store:
    def __init__(self, path=DB_FILE):
        for sub in ("thumbs", "context"):
            os.makedirs(os.path.join(INDEX_DIR, sub), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        # seq increases with every write (writes are serialized), so readers can ask "what changed since seq N?"
        self.db.execute("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, camera TEXT NOT NULL, "
                        "seq INTEGER NOT NULL, meta TEXT NOT NULL, vectors BLOB NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS items_seq ON items (seq)")
        self.db.commit()

    def put_many(self, rows):
        """rows: [(meta dict, {key: vector or (V, D) views})]. Inserts new items, replaces existing ones."""
        if not rows:
            return
        with self.db:
            seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM items").fetchone()[0]
            self.db.executemany("INSERT OR REPLACE INTO items (id, camera, seq, meta, vectors) VALUES (?, ?, ?, ?, ?)",
                                [(m["id"], m["camera"], seq + 1, json.dumps(m), _pack(v)) for m, v in rows])

    def delete_camera(self, camera):
        """Remove a camera's items (before re-indexing its video), including their images."""
        ids = [r[0] for r in self.db.execute("SELECT id FROM items WHERE camera = ?", (camera,))]
        with self.db:
            self.db.execute("DELETE FROM items WHERE camera = ?", (camera,))
        for item_id in ids:
            for sub in ("thumbs", "context"):
                path = os.path.join(INDEX_DIR, sub, item_id + ".jpg")
                if os.path.exists(path):
                    os.remove(path)

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def changed_since(self, seq):
        return self.db.execute("SELECT seq, meta, vectors FROM items WHERE seq > ? ORDER BY seq", (seq,)).fetchall()


class Index:
    """In-memory search copy of the store: items (list of dicts) and stacked vectors for score_items()."""

    def __init__(self, store=None):
        self.store = store or Store()
        self.rows = {}  # id -> (meta, vectors)
        self.seq = 0
        self.items, self.vecs = [], {}
        self.refresh()

    def refresh(self):
        """Load new and changed items. Cheap when nothing changed. Returns True if the index changed."""
        changed = self._apply(self.store.changed_since(self.seq))
        if self.store.count() != len(self.rows):  # items were deleted (a video was re-indexed): reload everything
            self.rows, self.seq = {}, 0
            changed = self._apply(self.store.changed_since(0))
        if changed:
            self._stack()
        return changed

    def _apply(self, rows):
        for seq, meta, blob in rows:
            meta = json.loads(meta)
            self.rows[meta["id"]] = (meta, _unpack(blob))
            self.seq = max(self.seq, seq)
        return bool(rows)

    def _stack(self):
        rows = list(self.rows.values())
        self.items = [m for m, _ in rows]
        vecs = {}
        if rows:
            for key in MEAN_KEYS:
                vecs[key] = np.stack([v[key] for _, v in rows]).astype(np.float32)
            for key in VIEW_KEYS:
                vecs[f"{key}_views"] = np.concatenate([v[f"{key}_views"] for _, v in rows]).astype(np.float32)
            vecs["view_item"] = np.concatenate([np.full(len(v["clip_views"]), i, np.int32)
                                                for i, (_, v) in enumerate(rows)])
            vecs["clip_fused"] = _normalize(vecs["clip"] + vecs["clip_plain"])
            vecs["clip_fused_views"] = _normalize(vecs["clip_views"] + vecs["clip_plain_views"])
            vecs["color"] = color_probs(vecs["clip"])
        self.vecs = vecs


def load_index():
    """(vecs, items) snapshot of the whole index, for one-off use such as eval.py."""
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError("No index yet. Run index_videos.py or live.py first.")
    index = Index()
    if not index.items:
        raise FileNotFoundError("The index is empty. Run index_videos.py or live.py first.")
    return index.vecs, index.items
