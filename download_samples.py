"""Download 3 more sample videos (the repo itself ships 4 small ones in videos/): wide CCTV-style mall views.

They are attached to the GitHub release "sample-videos" (kept out of the repo so cloning stays small) and are saved
into DATA_DIR/videos. Then index them:

    python download_samples.py
    python index_videos.py
"""

import os
import sys
import urllib.request

from common import DATA_DIR

RELEASE = "https://github.com/nadashaker/lost_and_found/releases/download/sample-videos/"
VIDEOS = {  # file -> folder under DATA_DIR (wide, high-mounted mall views, like real CCTV)
    "14393758-hd_1920_1080_30fps.mp4": "videos",   # escalators from above
    "14959168_3840_2160_30fps.mp4": "videos",      # wide 4K view of a mall atrium
    "15007111-uhd_3840_2160_31fps.mp4": "videos",  # dense crowd
}


def main():
    for name, folder in VIDEOS.items():
        path = os.path.join(DATA_DIR, folder, name)
        if os.path.exists(path):
            print(f"have     {name}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"download {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(RELEASE + name, path + ".part")
        except OSError as e:
            print(f"failed ({e})")
            continue
        os.replace(path + ".part", path)
        print(f"{os.path.getsize(path) / 1e6:.0f} MB")
    print(f"Videos are in {DATA_DIR}. Next: python index_videos.py")


if __name__ == "__main__":
    sys.exit(main())
