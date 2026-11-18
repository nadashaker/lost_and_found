"""Download 8 more sample videos (the repo itself ships 4 small ones in videos/).

They are attached to the GitHub release "sample-videos" (kept out of the repo so cloning stays small) and are saved
into DATA_DIR, in the folders the evaluation's ground truth expects (camera id = folder_file name). Then index them:

    python download_samples.py
    python index_videos.py
"""

import os
import shutil
import ssl
import sys
import urllib.request

from common import DATA_DIR

try:  # certifi's CA list (installed with ultralytics): Python's own can fail with "unable to get issuer certificate"
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = None

REPO = "https://github.com/nadashaker/lost_and_found"
RELEASE = REPO + "/releases/download/sample-videos/"
VIDEOS = {  # file -> folder under DATA_DIR
    # wide, high-mounted mall views, like real CCTV
    "14393758-hd_1920_1080_30fps.mp4": "videos",   # escalators from above
    "14959168_3840_2160_30fps.mp4": "videos",      # wide 4K view of a mall atrium
    "15007111-uhd_3840_2160_31fps.mp4": "videos",  # dense crowd
    # more mall and street scenes (most of eval/ground_truth.json's labelled items are in these)
    "144443-784570594_medium.mp4": "videos",
    "30291-380713825_medium.mp4": "videos",
    "31873-389248736_medium.mp4": "videos",
    "19172722-hd_1080_1920_30fps.mp4": "ObjectTracking",
    "4750056-hd_1920_1080_25fps.mp4": "ObjectTracking",
}


def main():
    failed = []
    for name, folder in VIDEOS.items():
        path = os.path.join(DATA_DIR, folder, name)
        if os.path.exists(path):
            print(f"have     {name}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"download {name} ...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(RELEASE + name, context=SSL_CONTEXT) as r, open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
        except OSError as e:
            print(f"failed ({e})")
            failed.append(name)
            continue
        os.replace(path + ".part", path)
        print(f"{os.path.getsize(path) / 1e6:.0f} MB")
    if failed:
        print(f"{len(failed)} download(s) failed. Run it again, or download them from {REPO}/releases")
        return 1
    print(f"Videos are in {DATA_DIR}. Next: python index_videos.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
