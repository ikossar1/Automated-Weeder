"""
Start here when a new photo arrives.

    python new_bed.py

Asks which photo, how big the plot is, opens the corner picker, then runs
the scan. One command instead of remembering three.

WHY A GUIDED START
    Every new scene needs the same four things decided before anything can
    run: which image, how big the plot really is, where its corners landed
    in that image, and what to call the output. Three of those are
    measurements only you can make, and the fourth is a name.

    Getting them wrong does not throw an error - it produces a confident,
    wrong map. So this asks for them up front rather than letting a stale
    value sit in a config file from the last bed. It is also why pressing
    Run in the editor is the wrong way in: those configs hold whatever
    photo was there last time.

PHOTOS LIVE IN ONE PLACE
    photos/, named bed_YYYYMMDD_note. See photos/NAMING.md for why.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import presets as P
import safestop

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HERE = pathlib.Path(__file__).parent
PHOTOS = HERE / "photos"
NAME_RE = re.compile(r"^([a-z]+[a-z0-9]*)_(\d+)$")


def find_images():
    """Photos in photos/, plus any dropped elsewhere by mistake."""
    PHOTOS.mkdir(exist_ok=True)
    kept = [p for p in PHOTOS.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    kept.sort(key=lambda p: -p.stat().st_mtime)

    stray = []
    for p in HERE.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            continue
        if set(p.parts) & {"maps", "handoff", "examples", "photos"}:
            continue
        stray.append(p)
    stray.sort(key=lambda p: -p.stat().st_mtime)
    return kept, stray


def check_name(path):
    """Does it follow NAMING.md? Returns (ok, a run name to suggest)."""
    m = NAME_RE.match(pathlib.Path(path).stem.lower())
    if m:
        bed, number = m.groups()
        return True, f"{bed}_{number}"
    return False, pathlib.Path(path).stem.replace(" ", "_").lower()


def ask(prompt, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not raw:
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  that is not a number, try again")


def next_number(bed):
    """The next free number for this bed, so nothing is overwritten."""
    used = set()
    for q in PHOTOS.iterdir():
        if not q.is_file():
            continue
        m = NAME_RE.match(q.stem.lower())
        if m and m.group(1) == bed:
            used.add(int(m.group(2)))
    n = 1
    while n in used:
        n += 1
    return n


def adopt(path):
    """
    Move a photo into photos/ and name it properly, in place.

    The camera writes photo.jpg every time, so without this each capture
    either overwrites the last one or accumulates as photo1, photo2,
    photo_final. Naming it on the way in costs one question and means
    every map can be traced back to the frame it came from.
    """
    path = pathlib.Path(path)
    ok, _ = check_name(path)
    if ok and path.parent == PHOTOS:
        return path

    print("")
    print(f"  {path.name} does not follow NAMING.md.")
    print("  Photos are named <bed>_<number>, e.g. basil_1.jpg")
    bed = input("  What should this bed be called? [bed]: ").strip().lower()
    bed = re.sub(r"[^a-z0-9]+", "", bed) or "bed"

    target = PHOTOS / f"{bed}_{next_number(bed)}{path.suffix.lower()}"
    PHOTOS.mkdir(exist_ok=True)
    path.rename(target)
    print(f"  Renamed to photos/{target.name}")
    return target


def pick_image():
    kept, stray = find_images()

    if stray:
        print("")
        print("Found outside photos/:")
        for q in stray[:6]:
            print(f"    {q.relative_to(HERE)}")
        if input("Move them into photos/ and name them? (y/N): "
                 ).strip().lower() in ("y", "yes"):
            for q in stray[:6]:
                adopt(q)
            kept, stray = find_images()

    if not kept:
        sys.exit("\nNothing in photos/. Put a photo there named like\n"
                 "    basil_1.jpg\n"
                 "and run this again. See photos/NAMING.md.")

    print("\nPhotos, newest first:\n")
    for i, p in enumerate(kept[:15], 1):
        ok, _ = check_name(p)
        flag = "" if ok else "    <- does not follow NAMING.md"
        print(f"  {i:2d}. {p.name}{flag}")
    print()

    raw = input("Which one? (number, or part of the name): ").strip()

    if raw.isdigit() and 1 <= int(raw) <= len(kept):
        return adopt(kept[int(raw) - 1]).relative_to(HERE)

    if (PHOTOS / raw).exists():
        return (PHOTOS / raw).relative_to(HERE)
    if (HERE / raw).exists():
        return pathlib.Path(raw)

    matches = [q for q in kept if raw.lower() in q.name.lower()]
    if len(matches) == 1:
        print(f"  using {matches[0].name}")
        return adopt(matches[0]).relative_to(HERE)
    if len(matches) > 1:
        sys.exit(f"'{raw}' matches {len(matches)} photos - be more specific")
    sys.exit(f"No photo matching '{raw}'")


def pick_candidates():
    """
    Which crops are planted in this bed.

    BioCLIP picks a winner from a list, so a shorter list is a better
    question. A basil bed judged against 23 crops has 22 ways to be
    confidently wrong; against one crop it has none.

    Weeds are not offered - they always stay in. They are what the crops
    are being told apart FROM, and without them every dandelion has to
    land on a crop name.
    """
    import bioclip

    choices = bioclip.crop_choices()
    print("\nWhich crops are in this bed?")
    print("All the weeds stay in the list either way - this only narrows")
    print("what a plant can be identified AS.\n")

    for i, (name, cat, common) in enumerate(choices, 1):
        note = f"  ({common})" if common else ""
        mark = " *" if cat == "ambiguous" else "  "
        print(f"  {i:2d}.{mark}{name}{note}")

    print("\n   * also grows wild, so it cannot settle crop vs weed alone")
    print("\nNumbers separated by spaces, or Enter for all of them.")

    raw = input("  crops: ").strip().replace(",", " ")
    if not raw:
        print("  using all crops")
        return []

    picked = []
    for bit in raw.split():
        if bit.isdigit() and 1 <= int(bit) <= len(choices):
            picked.append(choices[int(bit) - 1][0])
        else:
            print(f"  ignoring '{bit}' - not a number on the list")

    if not picked:
        print("  nothing valid picked, using all crops")
        return []

    print(f"  using: {', '.join(picked)}")
    return picked


def main():
    ap = argparse.ArgumentParser(description="Map a new photo.")
    P.add_arguments(ap)

    ap.add_argument("--corners", type=float, nargs=8,

                    metavar="N",

                    help="the four corners as x y x y x y x y, "

                         "so no window has to open here")
    args = ap.parse_args()
    if P.handle_flags(args):
        return

    a = P.Answers(P.get(args.preset) if args.preset else None)
    if args.preset and not a.preset:
        sys.exit(f"No preset called {args.preset!r}. Try --list-presets.")

    print("=" * 62)
    print("NEW BED" + (f"   preset: {args.preset}" if args.preset else ""))
    print("=" * 62)

    if "photo" in a.preset:
        image = pathlib.Path(a.record("photo", a.preset["photo"]))
        if not (HERE / image).exists():
            sys.exit(f"Preset names photo {str(image)!r}, which is not there.")
        print(f"Photo: {image}   (preset)")
    else:
        image = pick_image()
        a.record("photo", str(image))
    ok, suggested = check_name(image)
    if not ok:
        print(f"\nNote: '{pathlib.Path(image).name}' is not named the way")
        print("photos/NAMING.md describes. It will still work, but")
        print("bed_number keeps the map folder matched to the photo.")

    print("\nHow big is the plot, in centimetres?")
    print("  Width is along the NEAR edge, the one closest to the camera.")
    print("  Depth is from the near edge to the far one.")
    width = a.ask("width", "  width cm", 30.5, float)
    depth = a.ask("depth", "  depth cm", 35.0, float)

    name = a.ask("name", "\nName for this run", suggested)

    # Corners belong to one photo, so a preset can only replay them when
    # it is the same photo. Reusing yesterday's corners on today's picture
    # produces a clean map of nowhere.
    if args.corners:
        # Picked somewhere else - on a laptop, where a window can open -
        # and handed over as eight numbers. The clicking needs a screen;
        # the mapping does not, so this is the seam between them.
        corners = [str(v) for v in args.corners]
        a.record("corners", corners)
        print(f"{chr(10)}Corners: {' '.join(corners)}   (given on the command line)")
    elif "corners" in a.preset:
        corners = [str(v) for v in a.preset["corners"]]
        a.record("corners", corners)
        print(f"\nCorners: {' '.join(corners)}   (preset)")
    else:
        print("\n" + "-" * 62)
        print("Click the four corners of the plot in the window that opens.")
        print("Order: near-left, near-right, far-right, far-left.")
        print("Press q when the fourth is down.")
        print("-" * 62 + "\n")
        input("Enter when ready...")

        corners_file = HERE / ".corners.json"
        corners_file.unlink(missing_ok=True)

        # Over SSH there is no display, so the OpenCV window cannot
        # open. The browser picker serves the photo from here instead
        # and takes the clicks from wherever you are sitting.
        headless = not (os.environ.get("DISPLAY")
                        or os.environ.get("WAYLAND_DISPLAY"))
        picker = "pick_corners_web.py" if headless else "pick_corners.py"
        if headless:
            print("No display here, so the picker opens in a browser.")
        subprocess.run([sys.executable, str(HERE / picker),
                        str(image), str(width), str(depth),
                        "--out", str(corners_file)], cwd=HERE)

        # The picker writes them to a file and this reads them back.
        # Retyping eight numbers is not just tedious - one transposed
        # digit gives a confident wrong map with nothing to catch it.
        if not corners_file.exists():
            sys.exit("\nNo corners were picked, so there is nothing to scan."
                     "\nRun again and click all four before closing it.")

        corners = [str(v) for v in
                   json.loads(corners_file.read_text())["corners"]]
        corners_file.unlink(missing_ok=True)
        a.record("corners", corners)
    print(f"\nCorners: {' '.join(corners)}")

    grid = a.ask("grid", "\nGrid, as COLSxROWS", "5x6")
    merge = a.ask("merge", "Merge radius cm", 4.0, float)

    cmd = [sys.executable, str(HERE / "scan.py"), str(image),
           "--corners", *corners,
           "--plot", str(width), str(depth),
           "--grid", grid,
           "--merge", str(merge),
           "--crops", "--out", name]

    # YOLOE's only job is finding plants. Keep the word vague - "plant"
    # beat "lettuce" 0.41 to 0.14 on our own photos, and "poppy" scored
    # 0.00 on a poppy. Species come later, from BioCLIP, which is built
    # for it and has the New England crop and weed lists.
    print("\nThe detector only has to FIND plants, not name them.")
    print("Keep the word general - naming happens later, in BioCLIP.")
    prompts = a.ask("prompts", "  words", "plant")

    print("\nMinimum confidence. Lower finds more, including rubbish.")
    print("0.15 is a good start: on our basil bed 0.25 missed a real")
    print("plant that scored 0.24.")
    conf = a.ask("conf", "  confidence", 0.15, float)

    cmd += ["--prompts", prompts, "--conf", str(conf), "--imgsz", "1600"]

    print("\nName the plants with BioCLIP, and keep only the crops?")
    print("  y      - YOLOE finds them, BioCLIP says what they are, and")
    print("           the map shows crops only. Weeds are dropped.")
    print("           Anything it is unsure about counts as a crop.")
    print("  Enter  - skip it. Every plant goes on the map, unlabelled.")
    print("           Needs open_clip_torch installed; first run downloads")
    print("           about 1 GB.")
    if a.confirm("bioclip", "\nuse BioCLIP? (y / Enter): "):
        cmd += ["--bioclip", "--only", "crop"]
        if "candidates" in a.preset:
            chosen = a.record("candidates", a.preset["candidates"])
            print(f"  crops: {', '.join(chosen) or 'all'}   (preset)")
        else:
            chosen = a.record("candidates", pick_candidates())
        if chosen:
            cmd += ["--candidates", ",".join(chosen)]

    print("\n" + "-" * 62 + "\n")
    subprocess.run(cmd, cwd=HERE)

    print("\n" + "=" * 62)
    print(f"Check maps/{name}/{name}_corners.png FIRST.")
    print("If the green outline is not on your plot, every number is wrong.")
    print("\nThen plan a spray run:")
    print("  cd ../spraying")
    print(f"  python simulate.py ../mapping/maps/{name}/{name}.json \\")
    print(f"      --bed 0 0 {width:g} {depth:g} --rail-length 35 \\")
    print(f"      --throw 30.5 --height 14 --offset-across 3.6 \\")
    print(f"      --continuous --grid {grid} --grid-only \\")
    print(f"      --out plans/{name}")
    print("=" * 62)


if __name__ == "__main__":
    with safestop.guard("calibration and mapping"):
        main()
