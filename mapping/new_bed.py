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
    value sit in a config file from the last bed.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HERE = pathlib.Path(__file__).parent


def find_images():
    """Every photo under this folder, newest first."""
    found = []
    for p in HERE.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            if "maps" in p.parts or "handoff" in p.parts or "examples" in p.parts:
                continue
            found.append(p)
    found.sort(key=lambda p: -p.stat().st_mtime)
    return found


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


def pick_image():
    images = find_images()
    if not images:
        sys.exit("No photos found under this folder. Put one in and re-run.")

    print("\nPhotos here, newest first:\n")
    for i, p in enumerate(images[:15], 1):
        rel = p.relative_to(HERE)
        print(f"  {i:2d}. {rel}")
    print()

    raw = input("Which one? (number, or type a filename): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(images):
        return images[int(raw) - 1].relative_to(HERE)

    direct = HERE / raw
    if direct.exists():
        return pathlib.Path(raw)

    matches = [p for p in images if raw.lower() in p.name.lower()]
    if len(matches) == 1:
        print(f"  using {matches[0].relative_to(HERE)}")
        return matches[0].relative_to(HERE)
    if len(matches) > 1:
        sys.exit(f"'{raw}' matches {len(matches)} photos - be more specific")
    sys.exit(f"No photo matching '{raw}'")


def main():
    print("=" * 62)
    print("NEW BED")
    print("=" * 62)

    image = pick_image()

    print("\nHow big is the plot, in centimetres?")
    print("  Width is along the NEAR edge, the one closest to the camera.")
    print("  Depth is from the near edge to the far one.")
    width = ask("  width cm", 30.5, float)
    depth = ask("  depth cm", 35.0, float)

    name = ask("\nName for this run", image.stem.replace(" ", "_"))

    use_results = input(
        "\nIs there a results.csv from Delaney for this photo? (y/N): "
    ).strip().lower().startswith("y")
    results = ask("  path to results.csv", "") if use_results else ""

    print("\n" + "-" * 62)
    print("Now click the four corners of the plot in the window that opens.")
    print("Order: near-left, near-right, far-right, far-left.")
    print("Press q when the fourth is down.")
    print("-" * 62 + "\n")
    input("Enter when ready...")

    subprocess.run([sys.executable, str(HERE / "pick_corners.py"),
                    str(image), str(width), str(depth)], cwd=HERE)

    print("\nPaste the eight corner numbers from above.")
    print("(space separated, in the order it printed them)")
    while True:
        raw = input("corners: ").strip().replace(",", " ")
        bits = raw.split()
        if len(bits) == 8 and all(b.lstrip("-").replace(".", "").isdigit()
                                  for b in bits):
            corners = bits
            break
        print("  need exactly eight numbers")

    cmd = [sys.executable, str(HERE / "scan.py"), str(image),
           "--corners", *corners,
           "--plot", str(width), str(depth),
           "--grid", ask("\nGrid, as COLSxROWS", "5x2"),
           "--merge", str(ask("Merge radius cm", 4.0, float)),
           "--crops", "--out", name]

    if results:
        cmd += ["--results", results]
    else:
        cmd += ["--prompts", ask("Words for the detector", "plant,leaf"),
                "--conf", str(ask("Minimum confidence", 0.25, float)),
                "--imgsz", "1600"]

    print("\n" + "-" * 62)
    print("Running:")
    print("  " + " ".join(pathlib.Path(c).name if "\\" in c or "/" in c else c
                          for c in cmd[1:]))
    print("-" * 62 + "\n")
    subprocess.run(cmd, cwd=HERE)

    print("\n" + "=" * 62)
    print(f"Check maps/{name}/{name}_corners.png FIRST.")
    print("If the green outline is not on your plot, every number is wrong.")
    print("\nThen plan a spray run:")
    print(f"  cd ../spraying")
    print(f"  python simulate.py ../mapping/maps/{name}/{name}.json \\")
    print(f"      --bed 0 0 {width:g} {depth:g} --rail-length 35 \\")
    print(f"      --throw 30.5 --height 14 --continuous")
    print("=" * 62)


if __name__ == "__main__":
    main()
