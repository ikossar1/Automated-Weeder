"""
Click the four corners of your plot.

    python pick_corners.py photos_bed/bed.webp 244 122

A window opens. Click the four corners of the measured plot IN THIS ORDER:

    1. near-left    2. near-right    3. far-right    4. far-left

"Near" means nearest the camera - toward the bottom of the picture. Go round
the edge in order; jumping about gives a folded or mirrored map.

Click the corners of the SOIL, not the outer edge of the timber. The soil is
the surface the plants stand on, and that is the plane the maths solves for.

When four points are down it prints the finished scan.py command, ready to
paste. Press r to start over, q to quit.

Long term you would not do this by hand - you would put an ArUco marker at
each corner and let OpenCV find them in every frame. This is the manual
version of the same four points.
"""
import json
import pathlib
import sys

import cv2
import numpy as np

LABELS = ["1 near-left", "2 near-right", "3 far-right", "4 far-left"]
COLOURS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (72, 73, 227)]
WIN = "Click the four corners of the plot"


def _screen_size():
    """Usable screen in pixels. Falls back to a safe small desktop."""
    try:                                    # Windows
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()
        w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        if w > 200 and h > 200:
            return w, h
    except Exception:
        pass
    try:                                    # macOS / Linux with tkinter
        import tkinter
        r = tkinter.Tk(); r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        if w > 200 and h > 200:
            return w, h
    except Exception:
        pass
    return 1280, 720


def _fit_to_screen(img_w, img_h):
    """
    Scale factor that makes the whole photo fit on screen.

    The window is sized to the image, so an image taller than the monitor
    puts its lower half off the bottom of the screen - and you cannot click
    a corner you cannot see. Phone photos are the usual victim: 4032 px tall
    against a 1080 px screen.

    Leaves room for the title bar and taskbar.
    """
    sw, sh = _screen_size()
    return min(1.0, (sw * 0.90) / img_w, (sh * 0.82) / img_h)


def draw_quad(image_path, corners, out_path):
    """
    Save a copy of the photo with the four corners drawn on it.

    This is how you check the corners were right. The maths will happily
    accept corners that are nowhere near the bed and produce a confident,
    completely wrong map - there is nothing in the numbers that tells you.
    Looking at the quadrilateral does tell you, in one second.

    scan.py writes one of these automatically on every four-corner run.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    q = np.asarray(corners, np.int32).reshape(4, 2)

    shaded = img.copy()
    cv2.fillPoly(shaded, [q], (0, 255, 0))
    img = cv2.addWeighted(shaded, 0.20, img, 0.80, 0)
    cv2.polylines(img, [q], True, (0, 255, 0), 2)

    for (x, y), label in zip(q, LABELS):
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)
        for colour, thick in (((0, 0, 0), 4), ((0, 255, 0), 2)):
            cv2.putText(img, label, (int(x) + 9, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick)

    # Scale down before saving. A full-resolution phone photo written as
    # lossless PNG lands at 20+ MB, which is slow to open and pointless -
    # this is a check-by-eye picture, not data.
    h, w = img.shape[:2]
    shrink = min(1.0, 1600.0 / max(w, h))
    if shrink < 1.0:
        img = cv2.resize(img, (int(w * shrink), int(h * shrink)),
                         interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return str(out_path)


def save_corners(pts, out_path):
    """
    Hand the corners back as a file.

    When new_bed.py starts the picker it passes a path here, reads the file
    and carries on. Nobody retypes eight numbers - which is not just tedious
    but a real chance to transpose a digit and get a confident wrong map
    with nothing to catch it.
    """
    flat = [round(float(v), 1) for p in pts for v in p]
    pathlib.Path(out_path).write_text(json.dumps({"corners": flat}))

    # Printed as well as written, so the four corners can be carried
    # to a machine with no screen to click on.
    print()
    print("Corners. To map this photo somewhere without a window:")
    print("    python3 new_bed.py --corners "
          + " ".join(str(v) for v in flat))
    return flat


def report(pts, path, width, depth, out_path=None):
    flat = " ".join(f"{v:.0f}" for p in pts for v in p)

    # Check the four points really do trace the edge in order.
    #
    # Comparing the diagonals does NOT work - a long thin bed seen at an
    # angle has genuinely unequal diagonals, so that test cries wolf on
    # perfectly good corners.
    #
    # What does work: a rectangle stays CONVEX under perspective, however
    # extreme the angle. Clicking the corners out of order instead traces a
    # bow-tie, which is not convex. Walk the vertices and cross-multiply
    # successive edges - a convex polygon keeps the same sign all the way
    # round, a bow-tie flips.
    a = np.array(pts, float)
    signs = []
    for i in range(4):
        e1 = a[(i + 1) % 4] - a[i]
        e2 = a[(i + 2) % 4] - a[(i + 1) % 4]
        signs.append(np.sign(e1[0] * e2[1] - e1[1] * e2[0]))
    out_of_order = len(set(signs)) > 1

    print()
    print("=" * 64)
    if out_of_order:
        print("WARNING: the diagonals differ a lot, which usually means the")
        print("corners were clicked out of order. Press r and redo them as")
        print("near-left, near-right, far-right, far-left.")
        print()

    cmd = (f'python scan.py "{path}" --corners {flat} '
           f'--plot {width:.0f} {depth:.0f} '
           f'--prompts "plant,leaf" --conf 0.06 --imgsz 1600 '
           f'--merge 8 --grid 6x4 --out bedmap')

    if out_path:
        save_corners(pts, out_path)
        print("Corners captured. Close this window (press q) and the scan")
        print("will carry on by itself - nothing to copy.")
        print("=" * 64)
        return

    print("Paste this into the terminal:")
    print()
    print("  " + cmd)
    print()
    print('Or paste just these eight numbers after "--corners" in launch.json:')
    print()
    print("  " + flat)
    print("=" * 64)


def main():
    if len(sys.argv) < 2:
        sys.exit("python pick_corners.py <image> [width_cm] [depth_cm] "
                 "[--out corners.json]")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    path = args[0]
    width = float(args[1]) if len(args) > 1 else 400.0
    depth = float(args[2]) if len(args) > 2 else 300.0

    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read {path}")

    h, w = img.shape[:2]
    shrink = _fit_to_screen(w, h)
    picked = []

    def redraw():
        view = cv2.resize(img, None, fx=shrink, fy=shrink) if shrink < 1 else img.copy()

        for i, (u, v) in enumerate(picked):
            p = (int(u * shrink), int(v * shrink))
            cv2.circle(view, p, 7, COLOURS[i], -1)
            cv2.circle(view, p, 7, (255, 255, 255), 2)
            for colour, thick in (((255, 255, 255), 3), (COLOURS[i], 2)):
                cv2.putText(view, LABELS[i][0], (p[0] + 11, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick)
            if i:
                q = (int(picked[i - 1][0] * shrink), int(picked[i - 1][1] * shrink))
                cv2.line(view, q, p, (255, 255, 255), 2)

        if len(picked) == 4:
            a = (int(picked[0][0] * shrink), int(picked[0][1] * shrink))
            d = (int(picked[3][0] * shrink), int(picked[3][1] * shrink))
            cv2.line(view, d, a, (255, 255, 255), 2)

        nxt = LABELS[len(picked)] if len(picked) < 4 else "done - press q"
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (30, 30, 30), -1)
        cv2.putText(view, f"Click: {nxt}        r = restart    q = quit",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            u, v = x / shrink, y / shrink
            picked.append((u, v))
            print(f"  {LABELS[len(picked) - 1]:14s} ({u:7.1f}, {v:7.1f})")
            redraw()
            if len(picked) == 4:
                report(picked, path, width, depth, out_path)

    sw, sh = _screen_size()
    print(f"{path}  ({w} x {h} px)")
    print(f"screen {sw} x {sh} - showing the photo at "
          f"{int(w * shrink)} x {int(h * shrink)} so all of it fits")
    print()
    print("Click, in order: near-left, near-right, far-right, far-left")
    print("If a corner is off the edge of the photo, you cannot use it -")
    print("reshoot with the whole plot in frame.")
    print()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    redraw()
    cv2.setMouseCallback(WIN, on_click)

    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            picked.clear()
            print()
            print("-- restarted --")
            redraw()
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
