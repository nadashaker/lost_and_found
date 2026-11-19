"""Report side: a small web app to search the index by description and/or photo.

Works with items from video files (index_videos.py) and live cameras (live.py). While live.py runs, new items
show up on the next search, and the page shows each camera's latest frame.

Run:  python app.py   then open http://127.0.0.1:5000
"""

import json
import os
import threading
import time

import cv2
from flask import Flask, Response, abort, jsonify, render_template_string, request, send_from_directory
from PIL import Image

from common import (COLORS, INDEX_DIR, LIVE_DIR, PERSON, resolve_source, similar_people, RECORD_FPS, RECORDINGS_DIR, SEGMENT_SECONDS, fmt_time, load_clip,
                    load_dino, search)
from store import Index

app = Flask(__name__)
index_lock = threading.Lock()
live_index = Index()  # refreshed on each request, so items from live.py appear without a restart
load_clip()  # load models once at startup so the first search is fast
load_dino()


def current_index():
    with index_lock:
        live_index.refresh()
        return live_index.vecs, live_index.items


def live_status():
    try:
        with open(os.path.join(LIVE_DIR, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}  # live.py isn't running


def same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def hidden_cameras(items):
    """Videos that a demo camera in cameras.json plays on a loop were usually also indexed on their own
    (index_videos.py); hide those copies, so each item shows up once (from the demo camera)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.json")) as f:
            sources = [resolve_source(c["source"]) for c in json.load(f)]
    except (OSError, ValueError, KeyError, TypeError):
        return set()
    files = [src for src in sources if isinstance(src, str) and os.path.isfile(src)]
    return {it["camera"] for it in items if it.get("video") and any(same_file(it["video"], src) for src in files)}


PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lost &amp; Found Search</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#6b7280; --accent:#2563eb; --border:#e3e6eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161b; --card:#1d2027; --text:#e7e9ee; --muted:#9aa1ad; --accent:#6ea0ff; --border:#2c3039; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width: 1150px; margin: 0 auto; padding: 24px 16px; }
  h1 { margin: 0 0 4px; font-size: 1.5rem; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  form { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px;
         display:grid; gap:12px; grid-template-columns: 2fr 1.4fr 1fr 1fr 1fr auto; align-items:end; }
  .mode { grid-column: 1 / -1; display:flex; gap:6px; }
  .mode label { margin:0; font-size:.9rem; color:var(--text); padding:6px 14px; border:1px solid var(--border);
                border-radius:18px; cursor:pointer; }
  .mode input { display:none; }
  .mode input:checked + span { color:var(--accent); font-weight:600; }
  .mode label:has(input:checked) { border-color:var(--accent); }
  label { display:block; font-size:.8rem; color:var(--muted); margin-bottom:4px; }
  input, select, button { width:100%; padding:8px 10px; border-radius:6px; border:1px solid var(--border);
         background:var(--bg); color:var(--text); font:inherit; }
  button { background:var(--accent); color:#fff; border:none; cursor:pointer; font-weight:600; }
  @media (max-width: 900px) { form { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 520px) { form { grid-template-columns: 1fr; } }
  .query { margin: 18px 0 10px; color: var(--muted); }
  .query a { color:var(--accent); text-decoration:none; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:14px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .card .img { height:170px; display:flex; align-items:center; justify-content:center; background:#0001; }
  .card .img img { max-width:100%; max-height:170px; }
  .card .body { padding:10px 12px; font-size:.86rem; line-height:1.5; }
  .rank { font-weight:700; } .score { float:right; color:var(--accent); font-weight:600; }
  .muted { color:var(--muted); }
  .tag { display:inline-block; font-size:.72rem; padding:1px 6px; border-radius:9px; border:1px solid var(--border);
         color:var(--muted); margin-right:3px; }
  .card a { color:var(--accent); text-decoration:none; margin-right:10px; }
  .empty { color:var(--muted); padding: 30px 0; }
  .similar { display:flex; gap:14px; align-items:center; background:var(--card); border:1px solid var(--accent);
             border-radius:10px; padding:12px; margin-top:18px; font-size:.9rem; line-height:1.6; }
  .similar img { max-height:150px; max-width:120px; border-radius:6px; }
  .similar a { color:var(--accent); text-decoration:none; margin-right:10px; }
  .live { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; margin-bottom:18px; }
  .cam { background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .cam img { width:100%; display:block; background:#000; aspect-ratio:16/9; object-fit:contain; }
  .cam-body { padding:8px 12px; font-size:.86rem; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .dot.on { background:#16a34a; } .dot.off { background:#dc2626; }
</style></head>
<body><main>
<h1>Lost &amp; Found Search</h1>
<p class="sub">{{ n_items }} items and {{ n_people }} people indexed from {{ cameras|length }} camera(s). Describe what you're looking for, upload a photo, or both.</p>

{% if live %}
<div class="live">
{% for name, st in live.items() %}
  <div class="cam">
    <img src="/live/{{ name }}.jpg" data-cam="{{ name }}" alt="{{ name }} live">
    <div class="cam-body"><span class="dot {{ 'on' if st.connected else 'off' }}"></span><b>{{ name }}</b>
      <span class="muted" data-status="{{ name }}">{{ st.error or ('%s fps &middot; %d in view' % (st.processed_fps, st.items_in_view))|safe }}</span></div>
  </div>
{% endfor %}
</div>
<script>
  // Refresh the live frames and their status every second
  setInterval(async () => {
    document.querySelectorAll(".cam img").forEach(img => { img.src = "/live/" + img.dataset.cam + ".jpg?t=" + Date.now(); });
    try {
      const st = await (await fetch("/live/status")).json();
      for (const [name, s] of Object.entries(st)) {
        const el = document.querySelector('[data-status="' + name + '"]');
        if (el) el.textContent = s.error || (s.processed_fps + " fps · " + s.items_in_view + " in view");
      }
    } catch (e) {}
  }, 1000);
</script>
{% endif %}

<form method="post" action="/" enctype="multipart/form-data">
  <div class="mode">
    <label><input type="radio" name="mode" value="items" {{ 'checked' if not people }} onchange="this.form.submit()"><span>Items ({{ n_items }})</span></label>
    <label><input type="radio" name="mode" value="people" {{ 'checked' if people }} onchange="this.form.submit()"><span>People ({{ n_people }})</span></label>
  </div>
  <div><label>Description</label><input name="text" value="{{ text }}"
    placeholder="{{ 'e.g. little girl in pink, man in a red jacket' if people else 'e.g. purple shopping bag' }}"></div>
  <div><label>Photo (optional)</label><input type="file" name="photo" accept="image/*"></div>
  <div><label>Camera</label><select name="camera"><option value="">All</option>
    {% for c in cameras %}<option {{ 'selected' if c == camera }}>{{ c }}</option>{% endfor %}</select></div>
  {% if not people %}
  <div><label>Category</label><select name="category"><option value="">Any</option>
    {% for c in categories %}<option {{ 'selected' if c == category }}>{{ c }}</option>{% endfor %}</select></div>
  {% endif %}
  <div><label>Colour</label><select name="color"><option value="">Any</option>
    {% for c in colors %}<option {{ 'selected' if c == color }}>{{ c }}</option>{% endfor %}</select></div>
  <div><button>Search</button></div>
  <input type="hidden" name="sort" value="{{ sort }}">
</form>
<script>
  // "show most recent / most confident first": re-submit the form, keeping the current filters
  function setSort(order) { const f = document.querySelector("form"); f.sort.value = order; f.submit(); return false; }
</script>

{% if similar_to %}
<div class="similar">
  <img src="/files/{{ similar_to.thumb }}" alt="the person">
  <div><b>This person on all cameras</b><br>
    <span class="muted">Seen on {{ similar_to.camera }}, {{ fmt(similar_to.first_seen) }} &ndash; {{ fmt(similar_to.last_seen) }}.
    Below: everyone who looks like them, most similar first. Check each one: similar clothes are not proof it's the same person.</span><br>
    <a href="/files/{{ similar_to.context }}" target="_blank">Frame</a> <a href="/clip/{{ similar_to.id }}" target="_blank">Play clip</a></div>
</div>
{% endif %}
<div class="query">
  {% if similar_to %}{% elif searched %}Top matches{% if text %} for "<b>{{ text }}</b>"{% endif %}{% if photo %} + uploaded photo{% endif %}
  {% else %}All indexed {{ 'people' if people else 'items' }},
    {% if sort == 'recent' %}most recent first &middot; <a href="#" onclick="return setSort('confident')">show most confident first</a>
    {% else %}most confident first &middot; <a href="#" onclick="return setSort('recent')">show most recent first</a>{% endif %}
  {% endif %}
</div>

{% if results %}
<div class="grid">
{% for score, it in results %}
  <div class="card">
    <div class="img"><img src="/files/{{ it.thumb }}" alt="{{ it.category }}" loading="lazy"></div>
    <div class="body">
      <span class="rank">#{{ loop.index }} {{ it.category }}</span>
      {% if score is not none %}<span class="score">{{ '%.2f' % score }}</span>{% endif %}<br>
      <span class="muted">Camera:</span> {{ it.camera }}<br>
      <span class="muted">Seen:</span> {{ fmt(it.first_seen) }} &ndash; {{ fmt(it.last_seen) }}<br>
      <span class="tag">{{ it.color }}</span>{% for s in it.sources %}<span class="tag">{{ s }}</span>{% endfor %}<br>
      <a href="/files/{{ it.context }}" target="_blank">Frame</a>
      <a href="/clip/{{ it.id }}" target="_blank">Play clip</a>
      {% if it.category == 'person' %}<br><a href="/person/{{ it.id }}">Find on all cameras</a>{% endif %}
    </div>
  </div>
{% endfor %}
</div>
{% else %}<p class="empty">No matching {{ 'people' if people else 'items' }}.</p>{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text = request.form.get("text", "").strip()
    camera = request.form.get("camera", "")
    category = request.form.get("category", "")
    color = request.form.get("color", "")
    people = request.form.get("mode") == "people"
    sort = "recent" if request.form.get("sort") == "recent" else "confident"
    photo = request.files.get("photo")
    image = Image.open(photo.stream).convert("RGB") if photo and photo.filename else None

    vecs, items = current_index()
    hidden = hidden_cameras(items)
    results = search(vecs, items, text=text, image=image, camera=camera, category="" if people else category,
                     color=color, top_k=12 if (text or image) else 200, skip_cameras=hidden,
                     people=people, sort=sort) if items else []
    shown = [it for it in items if it["camera"] not in hidden]
    n_people = sum(it["category"] == PERSON for it in shown)
    return render_template_string(
        PAGE, results=results, text=text, camera=camera, category=category, color=color, photo=image is not None,
        searched=bool(text or image), n_items=len(shown) - n_people, n_people=n_people, people=people, sort=sort,
        cameras=sorted({it["camera"] for it in shown}),
        categories=sorted({it["category"] for it in shown if it["category"] != PERSON}), colors=COLORS,
        fmt=fmt_time, live=live_status(), similar_to=None)


@app.route("/person/<item_id>")
def same_person(item_id):
    """Find on all cameras: everyone who looks like this person, most similar first."""
    vecs, items = current_index()
    hidden = hidden_cameras(items)
    person, results = similar_people(vecs, items, item_id, skip_cameras=hidden)
    if person is None:
        abort(404)
    shown = [it for it in items if it["camera"] not in hidden]
    n_people = sum(it["category"] == PERSON for it in shown)
    return render_template_string(
        PAGE, results=results, text="", camera="", category="", color="", photo=False, searched=True,
        n_items=len(shown) - n_people, n_people=n_people, people=True, cameras=sorted({it["camera"] for it in shown}),
        categories=[], colors=COLORS, fmt=fmt_time, live=live_status(), similar_to=person)


@app.route("/files/<path:path>")
def files(path):
    return send_from_directory(INDEX_DIR, path)


@app.route("/live/<camera>.jpg")
def live_frame(camera):
    response = send_from_directory(LIVE_DIR, camera + ".jpg", max_age=0)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/live/status")
def live_status_json():
    return jsonify(live_status())


def video_frames(path, start, end):
    """(time, frame) from a video file, start..end in seconds from the start of the video."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    t = start
    while t <= end:
        ok, frame = cap.read()
        if not ok:
            break
        yield t, frame
        t += 1 / fps
    cap.release()


def recorded_frames(camera, start, end):
    """(time, frame) from a live camera's recorded segments, start..end in wall-clock seconds.
    Segment files are named by their start time in ms; frame k is at start + k / RECORD_FPS."""
    folder = os.path.join(RECORDINGS_DIR, camera)
    names = os.listdir(folder) if os.path.isdir(folder) else []
    segments = sorted((int(n[:-4]) / 1000, os.path.join(folder, n)) for n in names
                      if n.endswith(".mp4") and n[:-4].isdigit())
    for seg_start, path in segments:
        if seg_start > end or seg_start + SEGMENT_SECONDS + 1 < start:
            continue
        cap = cv2.VideoCapture(path)  # also works on the segment being written (all but its last ~2 s)
        k = max(0, int((start - seg_start) * RECORD_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        while True:
            ok, frame = cap.read()
            t = seg_start + k / RECORD_FPS
            if not ok or t > end:
                break
            yield t, frame
            k += 1
        cap.release()


CRLF = bytes([13, 10])


def mjpeg(img):
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
    return b"--frame" + CRLF + b"Content-Type: image/jpeg" + CRLF + CRLF + jpg + CRLF


def clip_frames(item, before=2.0, after=2.0, max_seconds=20, width=720):
    """Stream the footage around an item's sightings as JPEG frames, with the item boxed."""
    start = max(0.0, item["first_seen"] - before)
    end = min(item["last_seen"] + after, start + max_seconds)
    if item.get("video"):
        frames = video_frames(item["video"], start, end)
    else:
        frames = recorded_frames(item["camera"], start, end)
    sightings = item["sightings"]
    shown, prev_t = False, None
    for t, frame in frames:
        box_scale = frame.shape[1] / item["frame_size"][0]  # recordings are downscaled
        nearest = min(sightings, key=lambda s: abs(s[0] - t))
        if abs(nearest[0] - t) <= 1.0:
            x1, y1, x2, y2 = (int(v * box_scale) for v in nearest[1:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), max(2, frame.shape[1] // 400))
        scale = width / frame.shape[1]
        frame = cv2.resize(frame, (width, int(frame.shape[0] * scale)))
        cv2.putText(frame, f"{item['camera']}  {fmt_time(t)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if prev_t is not None:
            time.sleep(min(0.5, max(0.0, t - prev_t)))
        prev_t, shown = t, True
        yield mjpeg(frame)
    if not shown:  # no footage: expired recording, live.py wasn't recording, or the last ~2 s not written yet
        frame = cv2.imread(os.path.join(INDEX_DIR, item["context"]))
        if frame is not None:
            cv2.putText(frame, "No footage for this item (expired, or not recorded)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            yield mjpeg(frame)


@app.route("/clip/<item_id>")
def clip(item_id):
    _, items = current_index()
    item = next((it for it in items if it["id"] == item_id), None)
    if item is None:
        abort(404)
    return Response(clip_frames(item), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
