"""Download the full set of sample videos (the repo itself only ships 4 small ones in videos/).

They are attached to the GitHub release "sample-videos" (too big for the repo) and are saved into DATA_DIR, in the
folders the evaluation's ground truth expects (camera id = folder_file name). Then index them:

    python download_samples.py
    python index_videos.py
"""

import os
import sys
import urllib.request

from common import DATA_DIR

RELEASE = "https://github.com/nadashaker/lost_and_found/releases/download/sample-videos/"
VIDEOS = {  # file -> folder under DATA_DIR
    "144443-784570594_medium.mp4": "videos",
    "30291-380713825_medium.mp4": "videos",
    "31873-389248736_medium.mp4": "videos",
    "14393758-hd_1920_1080_30fps.mp4": "videos",
    "14959168_3840_2160_30fps.mp4": "videos",
    "15007111-uhd_3840_2160_31fps.mp4": "videos",
    "19172722-hd_1080_1920_30fps.mp4": "ObjectTracking",
    "4750056-hd_1920_1080_25fps.mp4": "ObjectTracking",
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
