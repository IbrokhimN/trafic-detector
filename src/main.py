import sys
from pathlib import Path

from photo_mode import analyze_photo
from video_mode import run_video_mode

SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def is_image_file(src):
    if isinstance(src, int):
        return False
    return Path(str(src)).suffix.lower() in IMAGE_EXTENSIONS

if __name__ == "__main__":
    if is_image_file(SOURCE):
        analyze_photo(str(SOURCE))
    else:
        run_video_mode(SOURCE)
