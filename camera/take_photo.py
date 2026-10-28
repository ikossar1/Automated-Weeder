"""
Take a photo with the Pi camera and file it where mapping expects it.

    python3 take_photo.py            ask what to call the bed
    python3 take_photo.py basil      call it basil, no questions

The camera writes the same filename every time, so a capture either
overwrites the last one or piles up as photo1, photo2, photo_final.
Naming it on the way in is one question, and it is what lets a map
months later be traced back to the frame it came from.

    <bed>_<number>.jpg      basil_1.jpg, basil_2.jpg, bed_7.jpg

The bed part is yours to choose and should say where the photo was
taken. The number is the next free one for that bed, so nothing is
ever overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PHOTOS = HERE.parent / "mapping" / "photos"
NAME_RE = re.compile(r"^([a-z]+[a-z0-9]*)_(\d+)$")
IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def next_number(bed):
    """The next free number for this bed, so nothing is overwritten."""
    used = set()
    for q in PHOTOS.glob("*"):
        m = NAME_RE.match(q.stem.lower())
        if m and m.group(1) == bed:
            used.add(int(m.group(2)))
    n = 1
    while n in used:
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Photograph a bed.")
    ap.add_argument("bed", nargs="?", help="what to call this bed")
    ap.add_argument("--delay", type=int, default=2000,
                    help="milliseconds to let the exposure settle "
                         "(default 2000)")
    ap.add_argument("--list", action="store_true",
                    help="show the photos already taken, then stop")
    args = ap.parse_args(argv)

    PHOTOS.mkdir(parents=True, exist_ok=True)

    if args.list:
        shots = sorted((q for q in PHOTOS.glob("*")
                        if q.suffix.lower() in IMAGES),
                       key=lambda q: -q.stat().st_mtime)
        if not shots:
            print("No photos yet.")
            return 0
        print(f"In {PHOTOS}, newest first:")
        for q in shots[:20]:
            ok = "" if NAME_RE.match(q.stem.lower()) else "   <- odd name"
            print(f"   {q.name}{ok}")
        return 0

    bed = args.bed or input("What is this bed called? [bed]: ").strip()
    bed = re.sub(r"[^a-z0-9]+", "", bed.lower()) or "bed"
    out = PHOTOS / f"{bed}_{next_number(bed)}.jpg"

    cmd = ["rpicam-still", "-n", "-t", str(args.delay), "-o", str(out)]
    print("  " + " ".join(cmd))
    try:
        r = subprocess.run(cmd)
    except FileNotFoundError:
        sys.exit("rpicam-still not found - this has to run on the Pi.")
    if r.returncode != 0:
        sys.exit(f"The camera returned {r.returncode}. Is it connected?")

    print(f"\n  Saved {out.name} in mapping/photos/")
    print(f"  Map it with:  cd ../mapping && python3 new_bed.py")
    print(f"  and choose {out.name} from the list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
