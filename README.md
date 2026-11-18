# Lost & Found Search

Search CCTV footage for lost items and people by **description** ("purple shopping bag", "little girl in pink")
or by **photo**. Each match comes with the camera, when it was seen, the frame, and a playable clip of the moment.
Works on recorded videos and on live cameras (RTSP, webcam), where new items become searchable within seconds.

## What it does

- **Items**: bags, backpacks, suitcases, phones, laptops, wallets, keys, umbrellas, bottles and more,
  found with three detectors: YOLO-seg (with masks), YOLO-World (open vocabulary, for things COCO doesn't have),
  and a static-object detector that finds things left behind (it appeared and stayed still).
- **People**: a separate People search ("man in a red jacket", "boy in a blue jacket"), plus
  **Find on all cameras**: click a person to see everyone who looks like them, on every camera.
- **Each item or person is stored once**, not once per frame: sightings are grouped, and each keeps its best views.
- **Search**: CLIP ViT-L/14@336px matches text to images; DINOv2 + CLIP match photos to images.
- **Live cameras**: each camera's newest frame is processed (never falling behind), items are written to the
  index every 3 seconds, and the footage is recorded (H.264, rolling 12 hours) so every result has a clip.

## Quick start

Needs Python 3.10+ and, for usable speed, an NVIDIA GPU (it runs on CPU, slowly).

```bash
git clone https://github.com/nadashaker/lost_and_found.git
cd lost_and_found
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match your GPU, see pytorch.org
pip install -r requirements.txt

python index_videos.py            # index the sample videos in videos/ (first run downloads ~1.4 GB of models)
python app.py                     # open http://127.0.0.1:5000
```

Models (YOLO, CLIP, DINOv2) download on first use into `data/models` and `data/torch_hub`.
The index, thumbnails and recordings also go into `data/`. Nothing is uploaded anywhere.

### More sample videos

The repo ships 4 short videos in `videos/`. Eight more (about 200 MB, including wide CCTV-style mall views and a
dense crowd) are attached to the [`sample-videos` release](https://github.com/nadashaker/lost_and_found/releases/tag/sample-videos):

```bash
python download_samples.py        # saves them into data/
python index_videos.py
```

### Live cameras

```bash
copy cameras.example.json cameras.json     # Linux/macOS: cp
python live.py                    # run next to app.py; Ctrl+C to stop
```

`cameras.json` lists the cameras (it is not committed, since real camera URLs contain passwords):

```json
[
  {"name": "lobby",  "source": "rtsp://user:password@192.168.1.20:554/stream1"},
  {"name": "webcam", "source": 0},
  {"name": "demo",   "source": "videos/4750061-hd_1920_1080_30fps.mp4", "loop": true}
]
```

A video file plays at real speed as if it were a camera (`"loop": true` repeats it; it is indexed on its first
pass only). `"static": false` turns off the left-object detector, for moving or very crowded views.

## Configuration

| Setting | How | Default |
|---|---|---|
| Where data goes | `LF_DATA_DIR` env var, or `"data_dir"` in `local_settings.json` | `./data` |
| Video folders to index | `"video_dirs"` in `local_settings.json` | `videos/`, `data/videos`, `data/ObjectTracking` |
| CLIP weights folder | `"clip_dir"` in `local_settings.json` (must end in `clip`) | `data/models/clip` |
| Detector input size | `LF_IMGSZ` env var (1280 found 1.6-4x more people in wide, high-mounted views; up to ~4x slower) | 640 |

`local_settings.json` (not committed) is for your machine's folders, for example to keep everything on another drive:

```json
{"data_dir": "D:/lost_and_found_data", "video_dirs": ["D:/footage/cam1", "D:/footage/cam2"]}
```

Every video in a folder is one camera, named `<folder>_<file>`.

## Files

| File | What it does |
|---|---|
| `index_videos.py` | Video files: detect, group sightings into items and people, embed, write the index |
| `live.py` | Live cameras: the same pipeline on each camera's newest frame, plus recording and live snapshots |
| `app.py` | The search web app (Flask) |
| `common.py` | Settings, models, and search |
| `store.py` | The index (SQLite in `data/index`, shared by `live.py` and the app) |
| `eval.py` | Measures search accuracy against 50 hand-labelled items (`eval/ground_truth.json`) |
| `download_samples.py` | Downloads the full set of sample videos |

## Accuracy and speed

`python eval.py` writes `eval/results.md`. Small, distant items
are the hardest.

Speed, on a laptop RTX 4050: indexing a recorded video takes about 1-5x its length (crowds are slowest).
Live, each camera gets about 1 analysed frame per second with two cameras (target: 2), with items searchable
3-5 seconds after they appear.

## Limitations

- People: in dense crowds partly hidden or distant people are missed.
- Find on all cameras ranks people by appearance (clothes, colours). Similar clothes are not proof it is the same
  person, and people walking together tend to match each other.

## Privacy

This searches people by how they look. If you use it beyond a demo, limit who can open the app, keep recordings
and the index only as long as needed (`RETENTION_HOURS` in `common.py`), and check your local data-protection law.

## Credits

Sample videos are from [Pexels](https://www.pexels.com) (free to use under the Pexels license).
Models: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and YOLO-World,
[OpenAI CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2).
