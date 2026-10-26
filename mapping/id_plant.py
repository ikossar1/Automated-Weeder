"""
Name the plant in a photo, with BioCLIP, without running the whole map.

This is the bench test for the identification step on its own. Point it at
a photo of a single plant and it says what BioCLIP thinks it is and whether
that counts as a crop.

    python id_plant.py photos/spider_1.jpg
    python id_plant.py photos/*.jpg --crops "spider plant,heartleaf philodendron"

--crops narrows the list of names BioCLIP may choose from, exactly as the
mapping run does. It matters more than anything else here: the model picks
a winner from the names it is given, so a name that is not on the list can
never win. Leave it off to score against all 69 candidates.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from PIL import Image

import bioclip as B

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import safestop


def main(argv=None):
    ap = argparse.ArgumentParser(description="Identify plants in photos.")
    ap.add_argument("photos", nargs="*", help="image files")
    ap.add_argument("--crops", default="",
                    help="comma-separated crop names to narrow to")
    ap.add_argument("--top", type=int, default=5,
                    help="how many guesses to print (default 5)")
    ap.add_argument("--list-crops", action="store_true",
                    help="print the crop names you can pass to --crops")
    args = ap.parse_args(argv)

    if args.list_crops:
        for name, cat, common in B.crop_choices():
            print(f"  {name:32s} {cat:10s} {common}")
        return 0

    keep = [c.strip() for c in args.crops.split(",") if c.strip()]
    candidates = B.build_candidates(keep)
    crops = [n for n, c in candidates if c == "crop"]

    paths = []
    for pat in args.photos:
        hits = sorted(pathlib.Path().glob(pat)) if any(
            ch in pat for ch in "*?[") else [pathlib.Path(pat)]
        paths += [h for h in hits if h.is_file()]
    if not paths:
        sys.exit("No photos matched.")

    print(f"Candidates : {len(candidates)} names, {len(crops)} of them crops")
    if keep:
        print(f"Narrowed to: {', '.join(crops)}")
    print("Loading BioCLIP (first run downloads about 1 GB)...")
    bc = B.Classifier(candidates=candidates)
    print(f"Running on : {bc.device}\n")

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"{path.name}: cannot read ({exc})")
            continue

        r = bc.classify(img, top_k=args.top)
        print(f"{path.name}")
        print(f"  verdict : {r['category']}")
        for name, p in r["top"]:
            common = B.COMMON.get(name, "")
            bar = "#" * int(round(p * 30))
            print(f"    {p:5.3f} {bar:<30s} {name}"
                  + (f"  ({common})" if common else ""))
        print()
    return 0


if __name__ == "__main__":
    with safestop.guard("plant identification"):
        sys.exit(main())
