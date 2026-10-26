"""
Accuracy check - turn the geometry into one defensible number.

    python validate.py truth.csv --height 35 --tilt 60

This is the script that matters most for your report. Everything else
*computes* positions; this one proves they are right. Without it you have a
pipeline that produces confident numbers you cannot defend, which in a design
review is worse than no numbers.

HOW TO DO IT
    1. Put something small and visible at a measured spot on the soil - a
       washer, a golf tee, a coin. Tape measure from the corner stake, and
       write down the real x and y in centimetres.
    2. Repeat for 10-15 spots. Spread them across the WHOLE frame, near and
       far, left and right, and especially into the corners. Error is not
       uniform and an average over the middle of the frame flatters you.
    3. Photograph them with the camera at a known height and tilt.
    4. Find each marker's pixel. `python validate.py --click photo.jpg` will
       let you click them, or read them off any image viewer.
    5. Fill in truth.csv:

           image,u,v,x_cm,y_cm
           grid.jpg,742,1518,10.0,0.0
           grid.jpg,1105,1402,30.0,12.0

    6. Run it. You get mean error, worst error, and whether the error grows
       with distance.

WHAT THE RESULT MEANS
    Mean error is the headline. Worst error is what sets your row tolerance -
    size that band off your WORST case, not your average, or you will hoe a
    crop eventually.

    If error climbs steeply with distance, mount the camera steeper or ignore
    the far part of the frame. Both are legitimate; a tighter working window
    scanned in more frames is a sound engineering trade, and saying so with
    numbers behind it is exactly what a design review wants.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys

import numpy as np

import geometry


def read_truth(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "image": (r.get("image") or "").strip(),
                    "u": float(r["u"]), "v": float(r["v"]),
                    "x_cm": float(r["x_cm"]), "y_cm": float(r["y_cm"]),
                    "heading_deg": float(r.get("heading_deg") or 0.0),
                    "robot_x_cm": float(r.get("robot_x_cm") or 0.0),
                    "robot_y_cm": float(r.get("robot_y_cm") or 0.0),
                    "pan_deg": (float(r["pan_deg"]) if r.get("pan_deg") else None),
                    "tilt_deg": (float(r["tilt_deg"]) if r.get("tilt_deg") else None),
                })
            except (KeyError, TypeError, ValueError) as e:
                sys.exit(f"Bad row in {path}: {r}\n  {e}\n"
                         f"Needs columns: image,u,v,x_cm,y_cm")
    if not rows:
        sys.exit(f"No usable rows in {path}")
    return rows


def click_points(image):
    """Open an image and print the pixels you click. Needs a desktop."""
    try:
        import cv2
    except ImportError:
        sys.exit("--click needs OpenCV:  python -m pip install opencv-python")

    img = cv2.imread(str(image))
    if img is None:
        sys.exit(f"Cannot read {image}")

    h, w = img.shape[:2]
    shrink = min(1.0, 1100.0 / max(w, h))
    view = cv2.resize(img, None, fx=shrink, fy=shrink) if shrink < 1 else img.copy()

    picked = []

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            u, v = x / shrink, y / shrink
            picked.append((u, v))
            print(f"{pathlib.Path(image).name},{u:.0f},{v:.0f},<x_cm>,<y_cm>")
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("click each marker  -  q when done", view)

    print("Click each marker. Press q when done.\n")
    print("image,u,v,x_cm,y_cm")
    cv2.imshow("click each marker  -  q when done", view)
    cv2.setMouseCallback("click each marker  -  q when done", on_click)

    while cv2.waitKey(50) & 0xFF not in (ord("q"), 27):
        if cv2.getWindowProperty("click each marker  -  q when done", 0) < 0:
            break
    cv2.destroyAllWindows()

    print(f"\n{len(picked)} point(s). Fill in the real x_cm,y_cm for each.")
    return picked


def plot_error(distances, errors, path="validation_error.png", mode="light"):
    """Error against distance from the camera. One series, so no legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    surface = "#fcfcfb" if mode == "light" else "#1a1a19"
    ink = "#0b0b0b" if mode == "light" else "#ffffff"
    soft = "#52514e" if mode == "light" else "#c3c2b7"
    blue = "#2a78d6" if mode == "light" else "#3987e5"

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.scatter(distances, errors, s=80, c=blue,
               edgecolors=surface, linewidths=2.0, zorder=3)

    if len(distances) >= 3:
        slope, intercept = np.polyfit(distances, errors, 1)
        xs = np.linspace(min(distances), max(distances), 50)
        ax.plot(xs, slope * xs + intercept, lw=2.0, color=soft,
                alpha=0.55, zorder=2)
        ax.text(0.03, 0.94, f"{slope:+.3f} cm error per cm of distance",
                transform=ax.transAxes, fontsize=8.5, color=soft)

    ax.set_title("Position error against distance from camera",
                 fontsize=12, color=ink, loc="left", pad=12)
    ax.set_xlabel("ground distance from camera  (cm)", fontsize=9, color=soft)
    ax.set_ylabel("error  (cm)", fontsize=9, color=soft)

    ax.grid(True, lw=0.5, alpha=0.18, color=soft)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(soft)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=soft, labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure pixel-to-ground accuracy.")
    ap.add_argument("truth", nargs="?", help="CSV of known marker positions")
    ap.add_argument("--click", metavar="IMAGE",
                    help="click markers in an image to get their pixels")

    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--height", type=float)
    ap.add_argument("--tilt", type=float)
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--forward", type=float, default=0.0)
    ap.add_argument("--left", type=float, default=0.0)
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)

    if args.click:
        click_points(args.click)
        return
    if not args.truth:
        ap.error("give a truth CSV, or --click IMAGE to collect pixels first")
    if args.height is None or args.tilt is None:
        ap.error("--height and --tilt are required")

    rows = read_truth(args.truth)

    cal = pathlib.Path(args.calibration)
    if cal.exists():
        intr = geometry.Intrinsics.load(cal)
    elif args.hfov is not None:
        from PIL import Image
        first = rows[0]["image"]
        if first and pathlib.Path(first).exists():
            w, h = Image.open(first).size
        else:
            sys.exit(f"Need {first} on disk to guess intrinsics from --hfov")
        intr = geometry.Intrinsics.from_fov(w, h, args.hfov)
    else:
        sys.exit(f"No {cal}. Run calibrate.py, or pass --hfov.")

    mount = geometry.CameraMount(args.height, args.tilt, args.pan,
                                 args.forward, args.left)
    proj = geometry.GroundProjector(intr, mount)

    print(f"Lens  : {intr}")
    print(f"Mount : height {args.height} cm, tilt {args.tilt} deg, "
          f"pan {args.pan} deg\n")

    hdr = (f"{'marker':>6}  {'pixel':>13}  {'predicted cm':>15}  "
           f"{'actual cm':>15}  {'error':>7}")
    print(hdr)
    print("-" * len(hdr))

    errors, distances, skipped = [], [], 0

    for i, r in enumerate(rows, 1):
        pose = geometry.RobotPose(r["robot_x_cm"], r["robot_y_cm"],
                                  r["heading_deg"])
        m = mount.at(tilt_deg=r["tilt_deg"], pan_deg=r["pan_deg"])

        xy = proj.to_garden([(r["u"], r["v"])], pose, m)[0]

        if not np.isfinite(xy[0]):
            print(f"{i:6d}  ({r['u']:5.0f},{r['v']:5.0f})  "
                  f"{'ray missed soil':>15}")
            skipped += 1
            continue

        err = math.hypot(xy[0] - r["x_cm"], xy[1] - r["y_cm"])
        dist = math.hypot(xy[0] - pose.x_cm, xy[1] - pose.y_cm)
        errors.append(err)
        distances.append(dist)

        print(f"{i:6d}  ({r['u']:5.0f},{r['v']:5.0f})  "
              f"({xy[0]:6.1f},{xy[1]:6.1f})  "
              f"({r['x_cm']:6.1f},{r['y_cm']:6.1f})  {err:6.2f}")

    if not errors:
        sys.exit("\nNo markers projected onto the soil. Check height, tilt "
                 "sign, and that the pixels are really in the image.")

    e = np.array(errors)
    print("\n" + "=" * len(hdr))
    print(f"Markers used    : {len(e)}" +
          (f"   ({skipped} skipped)" if skipped else ""))
    print(f"Mean error      : {e.mean():6.2f} cm")
    print(f"Median error    : {np.median(e):6.2f} cm")
    print(f"Worst error     : {e.max():6.2f} cm")
    print(f"Std deviation   : {e.std():6.2f} cm")

    if len(e) >= 3:
        slope = np.polyfit(distances, errors, 1)[0]
        print(f"Error vs range  : {slope:+.3f} cm per cm of distance")

    print(f"\nSuggested row tolerance: {e.max() * 1.5:.1f} cm")
    print("  That is worst error x 1.5. Size this band off the WORST case,")
    print("  not the mean - the mean is what you hit on a good day, and the")
    print("  cost of getting it wrong is hoeing a crop.")

    if intr.approximate:
        print("\nNOTE: uncalibrated lens model. Much of the error above is")
        print("probably the lens model, not the geometry. Run calibrate.py")
        print("and measure again before quoting these numbers.")

    p = plot_error(distances, errors, mode="dark" if args.dark else "light")
    if p:
        print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
