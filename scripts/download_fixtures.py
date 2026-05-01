"""Download test fixture audio files."""

import os
import urllib.request

FIXTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures"))

FILES = {
    "new_order_confusion_60seconds.wav": "https://github.com/user-attachments/files/26225505/new_order_confusion_60seconds.wav",
    "Vesuvius_v2_edit_60s.wav": "https://github.com/user-attachments/files/26225504/Vesuvius_v2_edit_60s.wav",
}


def main():
    for name, url in FILES.items():
        path = os.path.join(FIXTURES_DIR, name)
        if os.path.exists(path):
            print(f"  exists: {name}")
            continue
        print(f"  downloading: {name}")
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  saved: {name} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
