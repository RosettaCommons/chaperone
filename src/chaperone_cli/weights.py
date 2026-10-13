#!/usr/bin/env python3
"""Download model weights that a container does not bundle.

Usage:
    chaperone weights install rfd2
"""

import os
import sys
import urllib.request
from pathlib import Path

# name -> (settings field naming the target directory, file name, URL)
WEIGHTS = {
    "rfd2": (
        "RFD2_WEIGHTS_DIR",
        "RFD_173.pt",
        "https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt",
    ),
}


def install(name: str) -> Path:
    from chaperone import get_setting_field

    setting, filename, url = WEIGHTS[name]
    directory = get_setting_field(setting, default=None)
    if not directory:
        raise SystemExit(f"Set {setting} to the directory to download {filename} into.")
    dest = Path(directory).expanduser() / filename
    if dest.is_file():
        print(f"{dest} already exists, skipping")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)
    os.replace(part, dest)
    return dest


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "install" or argv[1] not in WEIGHTS:
        print(f"Usage: chaperone weights install {{{'|'.join(WEIGHTS)}}}")
        return 1
    install(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
