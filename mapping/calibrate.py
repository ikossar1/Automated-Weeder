"""
Camera calibration - measure what your lens does, once.

    python calibrate.py calib_photos/

Writes calibration.json, which everything else reads.

WHY A CHECKERBOARD
    The projection maths needs to know two things about your lens: how wide
    it sees (focal length) and how much it bends straight lines (distortion).
    Neither is printed on the part, and neither is quite what the datasheet
    says, because the real answer depends on how your specific lens is
    screwed into your specific sensor.

    A checkerboard is the measuring stick. You know its true geometry
    perfectly - the squares are identical and the lines are straight. So any
    way the photographed board differs from that is the lens's doing, and
    from enough photos OpenCV can solve for exactly what the lens is doing.

    It is not training. There is nothing learned and nothing probabilistic.
    It is solving for about a dozen unknowns, and the answer is the same
    every time.

HOW TO SHOOT IT
    Print a checkerboard - OpenCV's own pattern is fine, and the standard one
    is 10x7 squares, which is 9x6 INNER CORNERS. The inner-corner count is
    what you pass below, and it is the single most common thing people get
    wrong.

    Tape it to something rigid and flat. A curled sheet of paper will quietly
    poison the result.

    Then take 15-25 photos with the camera you will actually use, at the
    resolution you will actually use:
      - fill a good part of the frame
      - tilt the board in different directions, not just flat-on
      - get it into the frame CORNERS, not only the middle. Distortion lives
        at the edges, and if you never photograph the edges the solver has
        nothing to go on there
      - keep it sharp. Blurry photos are worse than fewer photos

    Do not crop or resize the photos afterwards. That changes the very thing
    you are measuring.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("This needs OpenCV:\n    python -m pip install opencv-python")

from geometry import Intrinsics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calibrate(folder, inner_corners=(9, 6), square_mm=25.0, out="calibration.json"):
    folder = pathlib.Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)

    if not files:
        sys.exit(f"No images in {folder}")

    # The board's true geometry, in millimetres. Z is zero - it is flat.
    nx, ny = inner_corners
    board = np.zeros((nx * ny, 3), np.float32)
    board[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_mm

    world_pts, image_pts = [], []
    size = None
    refine = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Looking for a {nx}x{ny} inner-corner board "
          f"({square_mm:.1f} mm squares) in {len(files)} image(s):\n")

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  {f.name:30s} unreadable")
            continue

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = grey.shape[::-1]
        elif grey.shape[::-1] != size:
            print(f"  {f.name:30s} SKIPPED - {grey.shape[::-1]} differs "
                  f"from {size}")
            continue

        ok, corners = cv2.findChessboardCorners(
            grey, (nx, ny),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not ok:
            print(f"  {f.name:30s} board not found")
            continue

        cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), refine)
        world_pts.append(board)
        image_pts.append(corners)
        print(f"  {f.name:30s} ok")

    print(f"\nUsable: {len(world_pts)} of {len(files)}")

    if len(world_pts) < 5:
        sys.exit("Need at least 5 usable views, and 15+ is much better.\n"
                 "If the board was never found, re-check the inner-corner "
                 "count - it is squares minus one in each direction.")
    if len(world_pts) < 12:
        print("WARNING: fewer than 12 views. The distortion terms will be "
              "poorly determined. Shoot more, especially into the corners.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        world_pts, image_pts, size, None, None)

    # Reprojection error: put the known corners back through the solved model
    # and see how far off they land. This is the number that says whether the
    # calibration is any good.
    errors = []
    for i in range(len(world_pts)):
        proj, _ = cv2.projectPoints(world_pts[i], rvecs[i], tvecs[i], K, dist)
        errors.append(cv2.norm(image_pts[i], proj, cv2.NORM_L2)
                      / len(proj))

    print(f"\nImage size          : {size[0]} x {size[1]}")
    print(f"Focal length  fx,fy : {K[0,0]:.1f}, {K[1,1]:.1f} px")
    print(f"Centre        cx,cy : {K[0,2]:.1f}, {K[1,2]:.1f} px")
    print(f"Distortion          : {np.round(dist.ravel(), 4).tolist()}")
    print(f"\nReprojection error  : {rms:.3f} px RMS "
          f"(worst view {max(errors):.3f})")

    if rms < 0.5:
        print("  Good. Use it.")
    elif rms < 1.0:
        print("  Usable, but worth reshooting if you want to quote tight "
              "accuracy numbers.")
    else:
        print("  Too high. Usual causes: a board that is not perfectly flat,")
        print("  blurry photos, or too few angles. Reshoot before relying on it.")

    # Sanity check the implied field of view against reality.
    fov_x = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    fov_y = 2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1])))
    print(f"\nImplied field of view: {fov_x:.1f} deg across, "
          f"{fov_y:.1f} deg down")
    print("  Compare that to your lens's spec. If it is wildly off, the")
    print("  square size or corner count is probably wrong.")

    path = Intrinsics(K, dist, size).save(out)
    print(f"\nWrote {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[2].strip()
                 + "\n\nOptions: python calibrate.py <folder> "
                   "[inner_x] [inner_y] [square_mm]")

    folder = sys.argv[1]
    ix = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    iy = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mm = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0

    calibrate(folder, (ix, iy), mm)
