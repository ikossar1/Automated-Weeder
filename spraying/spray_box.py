"""
Point at a box, spray that box.

    python spray_box.py

Opens the map as a clickable grid. Click a box and it is added to the queue.
Press ENTER and the commands are written: travel to each box with the valve
shut, open it, cover the box, shut it again, move on.

    click  ->  travel (valve OFF)  ->  spray the box (valve ON)  ->  OFF

HOW THIS DIFFERS FROM spray.py
    spray.py plans a route over the whole bed and holds the valve open for
    a whole patch of soil at a time - efficient, and the right thing for a
    real run.

    This one is for testing and for demonstrating. You choose exactly where
    mulch goes, one box at a time, and the valve cycles twice per box. That
    is more valve cycles than a full run needs, but every one of them is
    deliberate and you can see it happen.

    It is also the honest way to show a sceptical person that the machine
    hits what the map says it will: point at a square, watch mulch land in
    that square.

WHAT IT REFUSES
    Boxes holding a crop. The whole point of the map is knowing where they
    are; letting someone click one and spray it anyway would be a strange
    thing to build. Clicking one says so and does nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import presets as P
import safestop

import aim as A
import simulate as S

try:
    import cv2
except ImportError:
    # Only the clicking needs OpenCV. Everything else in here -
    # working out the boxes, planning a route through one - is
    # arithmetic, and the Pi runs it over SSH with no display.
    cv2 = None

HERE = pathlib.Path(__file__).parent
MAPS = HERE.parent / "mapping" / "maps"

DEFAULTS = {
    "rail_length": 35.0, "throw": 30.5, "height": 14.0,
    "offset_across": 3.6, "tilt_min": 18.0, "tilt_max": 80.0,
    "standoff": 8.0,         # cm the rail sits outside the bed edge
    "clearance": 2.0, "cell": 1.5,
    "patch": 5.1,            # 2 inches, the measured spray width
}

# Drawing. BGR, because OpenCV.
BG = (247, 249, 252)
INK = (22, 25, 27)
SOFT = (120, 126, 132)
CLEAR = (255, 255, 255)
CROP = (122, 175, 27)
PICKED = (214, 120, 42)
PLANT = (122, 175, 27)


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
            print("  not a number, try again")


def pick_map():
    if not MAPS.exists():
        sys.exit("No maps yet. Run mapping/new_bed.py first.")

    found = []
    for d in sorted(MAPS.iterdir(), key=lambda p: -p.stat().st_mtime):
        f = d / f"{d.name}.json"
        if f.exists():
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            found.append((d.name, len(doc.get("plants", []))))

    if not found:
        sys.exit("No maps yet. Run mapping/new_bed.py first.")

    print("\nMaps, newest first:\n")
    for i, (name, n) in enumerate(found[:15], 1):
        print(f"  {i:2d}. {name:20s} {n:2d} plants")
    print()

    raw = input("Which one? (number, or part of the name): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(found):
        return found[int(raw) - 1][0]
    matches = [f for f in found if raw.lower() in f[0].lower()]
    if len(matches) == 1:
        return matches[0][0]
    sys.exit(f"No single map matching '{raw}'")


class Board:
    """The clickable grid."""

    def __init__(self, bed, cells, plants, px_per_cm=14):
        self.bed = bed
        self.cells = cells
        self.plants = plants
        self.k = px_per_cm
        self.picked = []

        x0, y0, x1, y1 = bed
        self.pad = 30
        self.w = max(int((x1 - x0) * self.k) + 2 * self.pad, 560)
        self.h = int((y1 - y0) * self.k) + 2 * self.pad + 44

    def to_px(self, x_cm, y_cm):
        x0, y0, _x1, y1 = self.bed
        return (int(self.pad + (x_cm - x0) * self.k),
                int(self.pad + (y1 - y_cm) * self.k))

    def cell_at(self, px, py):
        for c in self.cells:
            a = self.to_px(c["x0"], c["y1"])
            b = self.to_px(c["x1"], c["y0"])
            if a[0] <= px < b[0] and a[1] <= py < b[1]:
                return c
        return None

    def draw(self):
        img = np.full((self.h, self.w, 3), BG, np.uint8)

        for c in self.cells:
            a = self.to_px(c["x0"], c["y1"])
            b = self.to_px(c["x1"], c["y0"])
            taken = c["count"] > 0
            chosen = c["name"] in self.picked

            if chosen:
                fill = PICKED
            elif taken:
                fill = CROP
            else:
                fill = CLEAR
            cv2.rectangle(img, (a[0] + 1, a[1] + 1), (b[0] - 1, b[1] - 1),
                          fill, -1)
            cv2.rectangle(img, (a[0] + 1, a[1] + 1), (b[0] - 1, b[1] - 1),
                          SOFT, 1)

            label = c["name"]
            colour = (255, 255, 255) if (chosen or taken) else SOFT
            cv2.putText(img, label, (a[0] + 6, a[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

            if taken and not chosen:
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                n = str(c["count"])
                (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.8, 2)
                cv2.putText(img, n, (mid[0] - tw // 2, mid[1] + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "crop", (mid[0] - 16, mid[1] + th // 2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (255, 255, 255), 1, cv2.LINE_AA)

            if chosen:
                # "#2" for the spray order, a bare "2" for a plant count.
                # Both sat in the middle of a box as the same glyph before,
                # which made a queued box and a two-plant box look alike.
                n = f"#{self.picked.index(c['name']) + 1}"
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                (tw, th), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.85, 2)
                cv2.putText(img, n, (mid[0] - tw // 2, mid[1] + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (255, 255, 255), 2, cv2.LINE_AA)

        # No plant circles. A coloured box already says a crop is in there,
        # and the circle only muddies which box you are about to click -
        # a plant overhanging a boundary makes a clear box look occupied.
        # The box colour is the map's own answer; trust it.

        bar = self.h - 36
        cv2.rectangle(img, (0, bar), (self.w, self.h), (235, 238, 242), -1)

        # Swatches, so the colours do not need explaining in the terminal.
        x = 12
        for colour, text in ((CLEAR, "clear"), (CROP, "crop"),
                             (PICKED, "queued")):
            cv2.rectangle(img, (x, bar + 11), (x + 13, bar + 24), colour, -1)
            cv2.rectangle(img, (x, bar + 11), (x + 13, bar + 24), SOFT, 1)
            cv2.putText(img, text, (x + 18, bar + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1, cv2.LINE_AA)
            x += 20 + 8 * len(text) + 14

        msg = (f"{len(self.picked)} queued   ENTER done   r reset   q cancel")
        cv2.putText(img, msg, (x + 6, bar + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, SOFT, 1, cv2.LINE_AA)
        return img


def choose_boxes(bed, cells, plants):
    if cv2 is None:
        sys.exit("Clicking boxes needs OpenCV:"
                 + chr(10) + "    sudo apt install python3-opencv"
                 + chr(10) + "Or use mobility/pick_and_go.py, "
                 + "which needs no window.")
    board = Board(bed, cells, plants)
    win = "Click the boxes to spray"
    state = {"done": False, "cancel": False}

    def on_click(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        c = board.cell_at(x, y)
        if c is None:
            return
        if c["count"] > 0:
            print(f"  {c['name']} holds a crop - not sprayable")
            return
        if c["name"] in board.picked:
            board.picked.remove(c["name"])
            print(f"  {c['name']} removed")
        else:
            board.picked.append(c["name"])
            print(f"  {c['name']} queued  ({len(board.picked)})")
        cv2.imshow(win, board.draw())

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, board.draw())
    cv2.setMouseCallback(win, on_click)

    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10):
            state["done"] = True
            break
        if k == ord("r"):
            board.picked.clear()
            print("  cleared")
            cv2.imshow(win, board.draw())
        if k in (ord("q"), 27):
            state["cancel"] = True
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if state["cancel"]:
        return []
    return board.picked


def build_cells(bed, plants, cols, rows, snap_cm=4.0):
    """
    The map's boxes, and what is in each.

    Same rule the map itself uses: a box holds a crop when a plant's CENTRE
    is in it, and the outer cells reach a little past the plot so a plant
    just outside a measured edge is not lost.
    """
    boxes, occupied = S.map_boxes(bed, plants, cols, rows)
    x0, y0, x1, y1 = bed
    bw, bh = (x1 - x0) / cols, (y1 - y0) / rows

    cells = []
    for j in range(rows):
        for i in range(cols):
            bx0, by0, bx1, by1 = boxes[j][i]
            lo_x = bx0 - snap_cm if i == 0 else bx0
            hi_x = bx1 + snap_cm if i == cols - 1 else bx1
            lo_y = by0 - snap_cm if j == 0 else by0
            hi_y = by1 + snap_cm if j == rows - 1 else by1

            inside = [p for p in plants
                      if lo_x <= p.x_cm < hi_x and lo_y <= p.y_cm < hi_y]
            cells.append({
                "name": f"C{i+1}R{j+1}", "col": i, "row": j,
                "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
                "x_cm": (bx0 + bx1) / 2, "y_cm": (by0 + by1) / 2,
                "count": len(inside),
                "plant_ids": sorted(p.id for p in inside),
            })
    return cells


def box_plan(cells, picked, plants, bed, rail, params):
    """
    Commands for the queue: travel shut, spray the box, shut again.

    Travel is planned as real moves with the valve off rather than a jump,
    because the machine has to physically get there and the time it takes
    is part of the run.
    """
    steps = []
    by_name = {c["name"]: c for c in cells}
    cell = params["cell"]

    def nozzle_for(y_or_x):
        lo, hi = rail["lo"], rail["hi"]
        pos = min(max(y_or_x, lo), hi)
        return pos, S.nozzle_at(rail, pos, params["height"],
                                params["pan_limit"], params["tilt_min"],
                                params["tilt_max"], params["throw"],
                                0.0, params["offset_across"])

    for order, name in enumerate(picked):
        c = by_name.get(name)
        if c is None:
            continue

        # --- travel, valve shut -------------------------------------------
        along = c["y_cm"] if rail["axis"] == "y" else c["x_cm"]
        pos, noz = nozzle_for(along)
        shot = A.aim(noz, (c["x_cm"], c["y_cm"]))
        steps.append([order, round(pos, 1), round(shot.pan_deg, 1),
                      round(shot.tilt_deg, 1), round(c["x_cm"], 1),
                      round(c["y_cm"], 1), 0, 0])

        # --- spray the box ------------------------------------------------
        # Spacing comes from the spray pattern. A patch 5 cm across does
        # not need a point every 1.5 cm - it needs them just under a patch
        # apart so they overlap, and no closer. Packing them tighter only
        # produces moves too short for a DC motor to execute.
        step = max(cell, params.get("patch", 5.1) * 0.8)
        nx = max(1, int(round((c["x1"] - c["x0"]) / step)))
        ny = max(1, int(round((c["y1"] - c["y0"]) / step)))
        xs = np.linspace(c["x0"] + (c["x1"] - c["x0"]) / (2 * nx),
                         c["x1"] - (c["x1"] - c["x0"]) / (2 * nx), nx)
        ys = np.linspace(c["y0"] + (c["y1"] - c["y0"]) / (2 * ny),
                         c["y1"] - (c["y1"] - c["y0"]) / (2 * ny), ny)
        serpentine = []
        for j, yy in enumerate(ys):
            row = list(xs) if j % 2 == 0 else list(xs)[::-1]
            serpentine += [(xx, yy) for xx in row]

        n_on = 0
        for (xx, yy) in serpentine:
            along = yy if rail["axis"] == "y" else xx
            pos, noz = nozzle_for(along)
            sol = A.solve(noz, (xx, yy), plants, params["clearance"])
            if not sol.ok:
                continue
            steps.append([order, round(pos, 1), round(sol.pan_deg, 1),
                          round(sol.tilt_deg, 1), round(xx, 1), round(yy, 1),
                          1, 0])
            n_on += 1

        # --- shut it ------------------------------------------------------
        if n_on and steps:
            last = list(steps[-1])
            last[6] = 0
            steps.append(last)

        print(f"  {name}: {n_on} spray points"
              + ("" if n_on else "   <- nothing reachable, skipped"))

    return steps


def main():
    ap = argparse.ArgumentParser(description="Spray chosen boxes.")
    P.add_arguments(ap)
    args = ap.parse_args()
    if P.handle_flags(args):
        return

    a = P.Answers(P.get(args.preset) if args.preset else None)
    if args.preset and not a.preset:
        sys.exit(f"No preset called {args.preset!r}. Try --list-presets.")

    print("=" * 62)
    print("SPRAY ONE BOX AT A TIME"
          + (f"   preset: {args.preset}" if args.preset else ""))
    print("=" * 62)

    if "map" in a.preset:
        name = a.record("map", a.preset["map"])
        print(f"Map: {name}   (preset)")
        if not (MAPS / name / f"{name}.json").exists():
            sys.exit(f"Preset names map {name!r}, which does not exist.")
    else:
        name = a.record("map", pick_map())
    doc = json.loads((MAPS / name / f"{name}.json").read_text())
    plants, _ = S.load_plants(MAPS / name / f"{name}.json")

    print(f"\n{name}: {len(plants)} plants")
    print("\nThe plot, in centimetres. Must match the mapping run.")
    width = a.ask("width", "  width cm  (near edge)", 35.0, float)
    depth = a.ask("depth", "  depth cm  (near to far)", 30.0, float)
    bed = (0.0, 0.0, width, depth)

    grid = a.ask("grid", "\nGrid, as COLSxROWS", "5x2")
    cols, rows = (int(v) for v in grid.lower().split("x"))

    print("\nThe machine. Enter through these unless something changed.")
    params = {
        "rail_length": a.ask("rail", "  rail travel cm",
                             DEFAULTS["rail_length"], float),
        "throw": a.ask("throw", "  sprayer throw cm",
                       DEFAULTS["throw"], float),
        "height": a.ask("height", "  nozzle height cm",
                        DEFAULTS["height"], float),
        "offset_across": a.ask("offset", "  camera to nozzle cm",
                             DEFAULTS["offset_across"], float),
        "clearance": a.ask("clearance", "  keep-off margin cm",
                           DEFAULTS["clearance"], float),
        "patch": a.ask("patch", "  spray patch width cm",
                       DEFAULTS["patch"], float),
        "standoff": a.ask("standoff", "  rail standoff from bed edge cm",
                          DEFAULTS["standoff"], float),
        "tilt_min": DEFAULTS["tilt_min"], "tilt_max": DEFAULTS["tilt_max"],
        "pan_limit": 70.0, "cell": DEFAULTS["cell"],
    }
    side = a.ask("side",
                 "\nWhich edge is the rail on? (left/right/near/far)",
                 "left")

    rail = S.limit_rail(S.build_rail(bed, side, params["standoff"]),
                        params["rail_length"])

    # Build the grid here rather than importing it from mapping/. The two
    # halves do not import each other - they meet at the map file - and a
    # shortcut now would be the first crack in that.
    cells = build_cells(bed, plants, cols, rows)

    free = [c for c in cells if c["count"] == 0]
    print(f"\n{len(cells)} boxes, {len(free)} with no crop in them.")
    print("\nA window is opening. Click the boxes you want sprayed.")
    print("Green boxes hold a crop and cannot be picked.")
    print("ENTER when done, r to clear, q to cancel.\n")

    picked = choose_boxes(bed, cells, plants)
    if not picked:
        print("\nNothing picked. Stopping.")
        return

    print(f"\nQueued: {', '.join(picked)}\n")
    S.report_reach(bed, rail, params["height"], params["throw"],
                   params["tilt_min"], params["tilt_max"])
    print("")
    steps = box_plan(cells, picked, plants, bed, rail, params)
    if not steps:
        print("\nNothing reachable in those boxes. Stopping.")
        return

    out_dir = HERE / "plans" / f"{name}_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "bed": list(bed),
        "plants": [{"id": p.id, "x": round(p.x_cm, 1), "y": round(p.y_cm, 1),
                    "r": round(A.footprint_radius(p), 1)} for p in plants],
        "rail": {k: (round(v, 1) if isinstance(v, float) else v)
                 for k, v in rail.items()},
        "nozzle": {"height": params["height"], "throw": params["throw"]},
        "stops": sorted({s[1] for s in steps}),
        "boxes": picked,
        "fields": ["box", "rail_pos", "pan", "tilt", "land_x", "land_y",
                   "spraying", "blocked_by"],
        "steps": steps,
    }
    (out_dir / "plan.json").write_text(json.dumps(plan))

    on = sum(s[6] for s in steps)
    cycles = 0
    prev = 0
    for s in steps:
        if s[6] and not prev:
            cycles += 1
        prev = s[6]

    print(f"\nSteps        : {len(steps):,}   spraying on {on:,}")
    print(f"Valve cycles : {cycles}   ({len(picked)} box(es), one each)")
    print(f"Wrote {out_dir / 'plan.json'}")

    if args.save_preset:
        where = P.save(args.save_preset, a.used)
        print(f"Saved preset {args.save_preset!r} to {where.name}")

    import subprocess
    subprocess.run([sys.executable, str(HERE / "motors.py"),
                    f"plans/{name}_boxes/plan.json", "--dc"], cwd=HERE)

    print("\n" + "=" * 62)
    print("The machine's instructions are in:")
    print("  spraying/dc_commands.csv")
    print("\nValve is shut while travelling to each box, open only inside")
    print("it. Nothing has moved - sending the file is a separate step.")
    print("=" * 62)


if __name__ == "__main__":
    with safestop.guard("box selection"):
        main()
