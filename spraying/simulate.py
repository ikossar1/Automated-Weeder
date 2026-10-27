"""
Plan a spray run and record every move, so it can be watched.

    python simulate.py maps/delaney_bed/delaney_bed.json
    python simulate.py maps/delaney_bed/delaney_bed.json --throw 250 --rail left

WHY IT WORKS THIS WAY
    Every angle in the output comes from aim.py - the same functions the
    machine will run. Nothing is re-derived here and nothing is
    reimplemented in the viewer. If the simulation sprays a crop, the real
    thing would too.

    That matters more than it sounds. A simulator that computes its own
    geometry only proves the simulator works. This one is a recording of
    the actual planner, which is why it is worth watching.

THE STRATEGY IT PLANS
    Radial sweeps. Park at a point on the rail, pick a pan angle, sweep
    tilt from near to far along that one line, then step pan and repeat.
    When the line runs out of rail stops, move along and do it again.

    Radial because the shadows are radial: along any one ray from the
    nozzle you are either clear or blocked, and once blocked you stay
    blocked for everything further out. So gating the spray is one test per
    ray, not a lookup per patch.

WHAT COMES OUT
    plan.json   every step: rail position, pan, tilt, where it lands,
                whether the spray is on, and what blocked it if not
    a PNG       the finished coverage, to glance at
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

import aim as A

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import safestop


class MapPlant:
    """A plant read back from a map file."""

    def __init__(self, pid, x, y, size_cm2):
        self.id = pid
        self.x_cm = float(x)
        self.y_cm = float(y)
        self.size_cm2 = float(size_cm2)


def load_plants(map_json):
    d = json.loads(pathlib.Path(map_json).read_text())
    plants = [MapPlant(p["id"], p["x_cm"], p["y_cm"], p.get("size_cm2", 0.0))
              for p in d["plants"]]
    return plants, d


def limit_rail(rail, length_cm, start_cm=None):
    """
    Cut the rail down to the hardware you actually have.

    A 350 mm rail does not span a 3 m bed. Pretending otherwise produces a
    plan the machine cannot execute, so the travel is clamped here and the
    shortfall is reported rather than silently assumed away.

    start_cm places the near end of the rail; left out, it is centred on
    the bed so the reachable patch sits in the middle.
    """
    if length_cm is None:
        return rail

    span = rail["hi"] - rail["lo"]
    length = min(float(length_cm), span)
    if start_cm is None:
        start = rail["lo"] + (span - length) / 2.0
    else:
        start = max(rail["lo"], min(float(start_cm), rail["hi"] - length))

    out = dict(rail)
    out["lo"] = start
    out["hi"] = start + length
    return out


def build_rail(bed, side, standoff=8.0):
    """
    Where the rail runs, and which way the nozzle faces.

    Returns (fixed coordinate, from, to, heading, axis) where axis is the
    one the nozzle slides along.
    """
    x0, y0, x1, y1 = bed
    return {
        "left":   dict(fixed=x0 - standoff, lo=y0, hi=y1, heading=0.0,   axis="y"),
        "right":  dict(fixed=x1 + standoff, lo=y0, hi=y1, heading=180.0, axis="y"),
        "near":   dict(fixed=y0 - standoff, lo=x0, hi=x1, heading=90.0,  axis="x"),
        "far":    dict(fixed=y1 + standoff, lo=x0, hi=x1, heading=270.0, axis="x"),
    }[side]


def nozzle_at(rail, pos, height, pan_limit, tilt_min, tilt_max, throw,
              offset_along=0.0, offset_across=0.0):
    if rail["axis"] == "y":
        x, y = rail["fixed"], pos
    else:
        x, y = pos, rail["fixed"]
    return A.Nozzle(x, y, height, heading_deg=rail["heading"],
                    pan_limit_deg=pan_limit, tilt_min_deg=tilt_min,
                    tilt_max_deg=tilt_max, max_range_cm=throw,
                    offset_along_cm=offset_along,
                    offset_across_cm=offset_across)


def passes_across(bed, side, standoff, n_passes):
    """
    Parallel rail lines across the bed, instead of one on the edge.

    A short-throw sprayer cannot reach across a wide bed from one side, so
    the sprayer has to travel instead. Each pass is its own rail line, and
    the spacing between them has to be inside the throw or stripes of bare
    soil survive between passes.

    n_passes = 1 gives the original single rail on the edge.
    """
    x0, y0, x1, y1 = bed
    rails = []
    for i in range(n_passes):
        f = (i + 0.5) / n_passes
        if side in ("left", "right"):
            at = x0 + f * (x1 - x0)
            head = 0.0 if side == "left" else 180.0
            rails.append(dict(fixed=at, lo=y0, hi=y1, heading=head, axis="y"))
        else:
            at = y0 + f * (y1 - y0)
            head = 90.0 if side == "near" else 270.0
            rails.append(dict(fixed=at, lo=x0, hi=x1, heading=head, axis="x"))
    if n_passes == 1:
        rails = [build_rail(bed, side, standoff)]
    return rails


def plan_run(plants, bed, rail, height=25.0, throw=120.0, pan_limit=70.0,
             tilt_min=12.0, tilt_max=78.0, stops=10, pan_step=4.0,
             tilt_step=2.5, clearance=2.0):
    """
    The whole run, step by step.

    Each step is one (rail position, pan, tilt) the machine would hold, with
    where that lands and whether the spray should be on.
    """
    x0, y0, x1, y1 = bed
    positions = np.linspace(rail["lo"], rail["hi"], stops)

    steps = []
    for si, pos in enumerate(positions):
        noz = nozzle_at(rail, float(pos), height, pan_limit,
                        tilt_min, tilt_max, throw)

        pan = -pan_limit
        while pan <= pan_limit + 1e-9:
            # Near to far along this one ray. Once a plant blocks it,
            # everything beyond is in shadow, so stop spraying and move on.
            tilt = tilt_max
            shadowed = False
            while tilt >= tilt_min - 1e-9:
                land = A.spray_point(noz, pan, tilt)
                if land is None:
                    tilt -= tilt_step
                    continue

                lx, ly = land
                inside = (x0 <= lx <= x1) and (y0 <= ly <= y1)

                on, blocker = False, None
                if inside and not shadowed:
                    shot = A.solve(noz, land, plants, clearance)
                    if shot.ok:
                        on = True
                    else:
                        # Only a blocked PATH shadows the rest of the ray.
                        # Landing on a plant just means skip this one step -
                        # beyond it the ray is clear again.
                        if "in the way" in shot.reason:
                            shadowed = True
                            blocker = shot.reason.split("#")[1].split()[0]

                steps.append([si, round(float(pos), 1), round(pan, 1),
                              round(tilt, 1), round(lx, 1), round(ly, 1),
                              1 if on else 0,
                              int(blocker) if blocker else 0])
                tilt -= tilt_step
            pan += pan_step

    return steps, positions


def coverage(steps, plants, bed, cell=4.0):
    """How much bare soil the plan actually lands on."""
    x0, y0, x1, y1 = bed
    gx = np.arange(x0 + cell / 2, x1, cell)
    gy = np.arange(y0 + cell / 2, y1, cell)
    pts = np.array([(a, b) for b in gy for a in gx])

    on_plant = np.zeros(len(pts), bool)
    for p in plants:
        r = A.footprint_radius(p)
        on_plant |= np.hypot(pts[:, 0] - p.x_cm, pts[:, 1] - p.y_cm) <= r
    soil = ~on_plant

    hit = np.zeros(len(pts), bool)
    sprayed = np.array([[s[4], s[5]] for s in steps if s[6]], float)
    if len(sprayed):
        for i, (px, py) in enumerate(pts):
            if soil[i]:
                hit[i] = (np.hypot(sprayed[:, 0] - px,
                                   sprayed[:, 1] - py) < cell).any()

    # Did it ever land on a plant? That is the failure that matters.
    on_crop = 0
    for s in steps:
        if not s[6]:
            continue
        for p in plants:
            if math.hypot(s[4] - p.x_cm, s[5] - p.y_cm) <= A.footprint_radius(p):
                on_crop += 1
                break

    return pts, soil, hit, on_crop


def draw(path, bed, plants, positions, rail, steps, pts, soil, hit, pct):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.lines import Line2D
    except ImportError:
        return None

    surface, ink, soft = "#fcfcfb", "#0b0b0b", "#52514e"
    blue, orange, grey = "#2a78d6", "#eb6834", "#d9d5cd"
    x0, y0, x1, y1 = bed
    span_x, span_y = x1 - x0, y1 - y0
    scale = 7.5 / max(span_x, span_y)

    fig, ax = plt.subplots(figsize=(max(span_x * scale + 1.6, 3.2),
                                    max(span_y * scale + 1.2, 3.0)))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.add_patch(Rectangle((x0, y0), span_x, span_y, facecolor="none",
                           edgecolor=soft, lw=1.2, zorder=1))

    miss = soil & ~hit
    ax.scatter(pts[hit, 0], pts[hit, 1], s=9, c=blue, marker="s",
               linewidths=0, alpha=.5, zorder=2)
    ax.scatter(pts[miss, 0], pts[miss, 1], s=9, c=orange, marker="s",
               linewidths=0, zorder=3)

    for p in plants:
        ax.add_patch(Circle((p.x_cm, p.y_cm), A.footprint_radius(p),
                            facecolor=grey, edgecolor=soft, lw=1.1, zorder=4))

    if rail["axis"] == "y":
        ax.plot([rail["fixed"]] * 2, [rail["lo"], rail["hi"]],
                color=soft, lw=2, zorder=5)
        ax.scatter([rail["fixed"]] * len(positions), positions, s=22,
                   c=surface, edgecolors=soft, linewidths=1.3, zorder=6)
    else:
        ax.plot([rail["lo"], rail["hi"]], [rail["fixed"]] * 2,
                color=soft, lw=2, zorder=5)
        ax.scatter(positions, [rail["fixed"]] * len(positions), s=22,
                   c=surface, edgecolors=soft, linewidths=1.3, zorder=6)

    pad = 0.06 * max(span_x, span_y)
    ax.set_xlim(min(x0, rail["fixed"] if rail["axis"] == "y" else x0) - pad, x1 + pad)
    ax.set_ylim(min(y0, rail["fixed"] if rail["axis"] == "x" else y0) - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Simulated spray run", fontsize=12, color=ink, loc="left", pad=14)
    ax.text(0.0, 1.012, f"{pct:.0f}% of bare soil covered   -   "
                        f"{len(steps):,} steps",
            transform=ax.transAxes, fontsize=8.5, color=soft)
    ax.set_xlabel("x  (cm)", fontsize=9, color=soft)
    ax.set_ylabel("y  (cm)", fontsize=9, color=soft)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(soft)
    ax.tick_params(colors=soft, labelsize=8)
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=blue,
               markersize=9, label="sprayed"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=orange,
               markersize=9, label="missed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=grey,
               markeredgecolor=soft, markersize=9, label="plant"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=8.5, labelcolor=soft)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan and record a spray run.")
    ap.add_argument("map_json", help="a map written by scan.py")
    ap.add_argument("--bed", nargs=4, type=float, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--side", default="left",
                    choices=["left", "right", "near", "far"],
                    help="which edge the rail runs along")
    ap.add_argument("--height", type=float, default=25.0,
                    help="nozzle height above the soil, cm")
    ap.add_argument("--throw", type=float, default=120.0,
                    help="how far the sprayer reaches, cm. MEASURE THIS - "
                         "everything downstream scales off it")
    ap.add_argument("--pan-limit", type=float, default=70.0)
    ap.add_argument("--tilt-min", type=float, default=12.0)
    ap.add_argument("--tilt-max", type=float, default=78.0)
    ap.add_argument("--continuous", action="store_true",
                    help="plan a route that keeps the valve open for a whole "
                         "patch of soil at a time, instead of sweeping "
                         "angles and gating. Valve cycles drop to one per "
                         "patch of free soil")
    ap.add_argument("--grid", default=None, metavar="COLSxROWS",
                    help="the map's grid, e.g. 5x2. Used with --grid-only")
    ap.add_argument("--grid-only", action="store_true",
                    help="spray only the map boxes that hold no crop, rather "
                         "than every bare centimetre. Coarser, but it keeps "
                         "a whole box between the stream and any plant")
    ap.add_argument("--cell", type=float, default=2.0,
                    help="route resolution in cm, for --continuous")
    ap.add_argument("--offset-along", type=float, default=0.0,
                    help="nozzle offset from the head along its heading, cm")
    ap.add_argument("--offset-across", type=float, default=0.0,
                    help="nozzle offset sideways from the head, cm. The "
                         "camera-to-airbrush gap goes here")
    # How far the rail sits outside the bed edge. This is the cheapest
    # lever on coverage there is: every cm here is a cm taken straight off
    # the far side of the bed, and unlike the throw it costs nothing to
    # change. It was hardcoded at 8 cm, which quietly cost a third of a
    # 30 cm bed.
    ap.add_argument("--standoff", type=float, default=8.0,
                    help="cm the rail sits outside the bed edge "
                         "(default 8; 0 = flush with the edge)")
    ap.add_argument("--rail-length", type=float, default=None,
                    help="how long the rail actually is, cm. Without it the "
                         "rail is assumed to span the whole bed")
    ap.add_argument("--rail-start", type=float, default=None,
                    help="where the rail's near end sits along the edge, cm. "
                         "Default centres it on the bed")
    ap.add_argument("--passes", type=int, default=1,
                    help="parallel traverses across the bed instead of one "
                         "rail on the edge. A short-throw sprayer has to "
                         "travel; this is how many lines it drives")
    ap.add_argument("--stops", type=int, default=10)
    ap.add_argument("--pan-step", type=float, default=4.0)
    ap.add_argument("--tilt-step", type=float, default=2.5)
    ap.add_argument("--clearance", type=float, default=2.0,
                    help="keep-off margin round each plant, cm")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args(argv)

    plants, doc = load_plants(args.map_json)
    if not plants:
        sys.exit("That map has no plants in it.")

    if args.bed:
        bed = tuple(args.bed)
    else:
        xs = [p.x_cm for p in plants]
        ys = [p.y_cm for p in plants]
        bed = (0.0, 0.0, max(xs) * 1.2, max(ys) * 1.2)
        print(f"No --bed given, guessing {bed[2]:.0f} x {bed[3]:.0f} cm "
              f"from the plants")

    rails = [limit_rail(r, args.rail_length, args.rail_start)
             for r in passes_across(bed, args.side, args.standoff,
                                    args.passes)]
    rail = rails[0]
    out_dir = pathlib.Path(args.out or
                           (pathlib.Path(args.map_json).parent / "spray"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plot        : {bed[2]-bed[0]:.0f} x {bed[3]-bed[1]:.0f} cm")
    print(f"Plants      : {len(plants)}")
    travel = rail["hi"] - rail["lo"]
    along = (bed[3] - bed[1]) if rail["axis"] == "y" else (bed[2] - bed[0])
    print(f"Rail        : {args.side} edge, {travel:.0f} cm of travel, "
          f"{args.stops} stops")
    if travel < along - 1e-6:
        print(f"              bed is {along:.0f} cm along that edge - the "
              f"rail covers {100*travel/along:.0f}% of it")
    print(f"Nozzle      : {args.height:.0f} cm up, throws {args.throw:.0f} cm, "
          f"pan +/-{args.pan_limit:.0f}")
    # The reach that actually binds. Two separate things cap it, and it is
    # easy to watch only one.
    #
    # The throw is a SLANT distance - the nozzle is up in the air, so its
    # height is spent on the way down and only sqrt(throw^2 - height^2) is
    # left to cross the ground. A 30 cm throw from 14 cm up crosses 27 cm,
    # not 30. Separately, a nozzle can only aim so shallow: at tilt_min the
    # ray lands at height / tan(tilt_min). Whichever is smaller wins.
    r = reach_into_bed(bed, rail, args.height, args.throw,
                       args.tilt_min, args.tilt_max)
    print(f"Reach       : {r['flat']:.1f} cm across the ground "
          f"({r['limit']} limited), {r['into']:.1f} cm of it inside the bed")
    print(f"              = {r['pct']:.0f}% of the {r['width']:.1f} cm width, "
          f"with the rail {r['standoff']:.1f} cm off the edge")

    if r["short"] > 0.5:
        reach_advice(r, args.height, args.throw, args.tilt_min)

    print("Planning...")

    grid_wh = None
    if args.grid and args.grid.lower() != "none" and args.grid_only:
        grid_wh = tuple(int(v) for v in args.grid.lower().split("x"))

    if args.continuous:
        steps, patches, _free = plan_continuous(
            plants, bed, rail, args.height, args.throw, args.pan_limit,
            args.tilt_min, args.tilt_max, args.cell, args.clearance,
            args.offset_along, args.offset_across, grid_wh)
        positions = sorted({s[1] for s in steps})
        print(f"\nContinuous route: {patches} patch(es) of free soil")
        print(f"                  VALVE CYCLES: {patches}")
        print(f"                  {len(steps):,} path points")
    else:
        steps, positions = [], []
    for ri, r in enumerate(rails if not args.continuous else []):
        st, pos = plan_run(
            plants, bed, r, args.height, args.throw, args.pan_limit,
            args.tilt_min, args.tilt_max, args.stops, args.pan_step,
            args.tilt_step, args.clearance)
        # Keep the pass number in the stop index so the viewer can tell
        # one traverse from the next.
        for row in st:
            row[0] += ri * args.stops
        steps.extend(st)
        positions.extend(pos)

    if args.passes > 1:
        if rails[0]["axis"] == "y":
            gap = (bed[2] - bed[0]) / args.passes
        else:
            gap = (bed[3] - bed[1]) / args.passes
        print(f"Passes      : {args.passes}, {gap:.0f} cm apart"
              + ("   <- wider than the throw, expect stripes"
                 if gap / 2 > args.throw else ""))

    on = sum(s[6] for s in steps)
    print(f"\nSteps       : {len(steps):,}   spraying on {on:,} of them")

    pts, soil, hit, on_crop = coverage(steps, plants, bed)
    pct = 100.0 * hit[soil].sum() / max(soil.sum(), 1)
    print(f"Coverage    : {pct:.1f}% of bare soil")
    print(f"Hit a plant : {on_crop}   <- must be 0")

    plan = {
        "bed": list(bed),
        "plants": [{"id": p.id, "x": round(p.x_cm, 1), "y": round(p.y_cm, 1),
                    "r": round(A.footprint_radius(p), 1)} for p in plants],
        "rail": {k: (round(v, 1) if isinstance(v, float) else v)
                 for k, v in rail.items()},
        "nozzle": {"height": args.height, "throw": args.throw,
                   "pan_limit": args.pan_limit, "tilt_min": args.tilt_min,
                   "tilt_max": args.tilt_max},
        "stops": [round(float(p), 1) for p in positions],
        "coverage_pct": round(pct, 1),
        "hit_plant": on_crop,
        "fields": ["stop", "rail_pos", "pan", "tilt", "land_x", "land_y",
                   "spraying", "blocked_by"],
        "steps": steps,
    }
    jp = out_dir / "plan.json"
    jp.write_text(json.dumps(plan))
    print(f"\nWrote {jp}  ({jp.stat().st_size/1e6:.1f} MB)")

    png = draw(out_dir / "spray_coverage.png", bed, plants, positions, rail,
               steps, pts, soil, hit, pct)
    if png:
        print(f"Wrote {png}")

    if args.continuous:
        rp = draw_route(out_dir / "spray_route.png", bed, plants, rail,
                        steps, patches, args.cell,
                        grid=grid_wh, clearance=args.clearance)
        if rp:
            print(f"Wrote {rp}   <- the one to look at before running")

    if on_crop:
        print("\nWARNING: the plan sprays a plant. Raise --clearance.")
    return plan


# ----------------------------------------------------------------------------
# Continuous planning - keep the valve open
# ----------------------------------------------------------------------------

def map_boxes(bed, plants, cols, rows, clearance=0.0):
    """
    The map's own grid, and which boxes hold a crop.

    Returns (boxes, occupied) where boxes[j][i] is (bx0, by0, bx1, by1) and
    occupied[j][i] says whether any plant sits in it.
    """
    x0, y0, x1, y1 = bed
    bw, bh = (x1 - x0) / cols, (y1 - y0) / rows

    boxes, occupied = [], np.zeros((rows, cols), bool)
    for j in range(rows):
        line = []
        for i in range(cols):
            bx0, by0 = x0 + i * bw, y0 + j * bh
            bx1, by1 = bx0 + bw, by0 + bh
            line.append((bx0, by0, bx1, by1))
            # A box holds a crop when the plant's CENTRE is in it - the same
            # rule the map's own grid uses, so the two always agree.
            #
            # Not "any leaf overlaps", which would mark all four boxes round
            # a plant sitting near a corner and leave almost nothing to
            # spray. Overhanging leaves are handled separately: free_grid
            # also drops every cell within `clearance` of a plant, so the
            # stream keeps its distance even inside a box called clear.
            for p in plants:
                if bx0 <= p.x_cm < bx1 and by0 <= p.y_cm < by1:
                    occupied[j, i] = True
                    break
        boxes.append(line)
    return boxes, occupied


def free_grid(bed, plants, cell, clearance, grid=None):
    """
    Grid of the bed. True where there is bare soil worth spraying.

    `grid` as (cols, rows) restricts spraying to the MAP's boxes that hold
    no crop - the whole box is either in or out. Without it, every square
    centimetre of bare soil is fair game, which covers more ground but
    sprays right up against the plants.
    """
    x0, y0, x1, y1 = bed
    nx = max(1, int(round((x1 - x0) / cell)))
    ny = max(1, int(round((y1 - y0) / cell)))

    boxes = occupied = None
    if grid:
        cols, rows = grid
        boxes, occupied = map_boxes(bed, plants, cols, rows, clearance)
        bw, bh = (x1 - x0) / cols, (y1 - y0) / rows

    free = np.ones((ny, nx), bool)
    for j in range(ny):
        for i in range(nx):
            cx = x0 + (i + 0.5) * cell
            cy = y0 + (j + 0.5) * cell

            if occupied is not None:
                bi = min(int((cx - x0) / bw), cols - 1)
                bj = min(int((cy - y0) / bh), rows - 1)
                if occupied[bj, bi]:
                    free[j, i] = False
                    continue

            for p in plants:
                if math.hypot(cx - p.x_cm, cy - p.y_cm) <= A.footprint_radius(p) + clearance:
                    free[j, i] = False
                    break
    return free, nx, ny


def components(free):
    """
    Label the connected patches of free soil.

    THIS IS THE NUMBER THAT SETS HOW OFTEN THE VALVE MUST CYCLE. The aim
    point can wander anywhere inside one connected patch without crossing a
    plant, so one patch can be done in a single continuous pass with the
    valve open the whole time. Getting to a different patch means crossing
    a plant, and that is the only honest reason to shut it.

    So: valve openings = number of patches. Four plants sitting as islands
    in a bed leave the soil in ONE piece, which is why one opening is
    achievable. A plant touching an edge, or two plants touching each
    other, can cut the soil in half and force a second.
    """
    ny, nx = free.shape
    label = np.zeros((ny, nx), int)
    current = 0

    for sj in range(ny):
        for si in range(nx):
            if not free[sj, si] or label[sj, si]:
                continue
            current += 1
            stack = [(sj, si)]
            label[sj, si] = current
            while stack:
                j, i = stack.pop()
                for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nj, ni = j + dj, i + di
                    if 0 <= nj < ny and 0 <= ni < nx and free[nj, ni] \
                            and not label[nj, ni]:
                        label[nj, ni] = current
                        stack.append((nj, ni))
    return label, current


def tour(label, which):
    """
    One continuous route over every cell of a patch.

    Walks a spanning tree depth first and records the backtracking too, so
    the path never jumps. It crosses some cells more than once - which for
    spraying is only a little extra mulch, and is the price of never
    lifting the pen.
    """
    ny, nx = label.shape
    cells = [(j, i) for j in range(ny) for i in range(nx) if label[j, i] == which]
    if not cells:
        return []

    start = min(cells)
    seen = {start}
    path = [start]

    def walk(j, i):
        # Serpentine bias: prefer across, then along, so the route reads as
        # rows rather than a random scribble.
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and label[nj, ni] == which \
                    and (nj, ni) not in seen:
                seen.add((nj, ni))
                path.append((nj, ni))
                walk(nj, ni)
                path.append((j, i))        # the backtrack, kept

    import sys as _sys
    old = _sys.getrecursionlimit()
    _sys.setrecursionlimit(max(old, len(cells) * 4 + 1000))
    try:
        walk(*start)
    finally:
        _sys.setrecursionlimit(old)
    return path


def plan_continuous(plants, bed, rail, height, throw, pan_limit,
                    tilt_min, tilt_max, cell=2.0, clearance=2.0,
                    offset_along=0.0, offset_across=0.0, grid=None):
    """
    A route that keeps the valve open for a whole patch of soil at a time.

    Instead of sweeping angles and gating the spray as the aim wanders in
    and out of shadow, this plans in GROUND space: work out which soil is
    worth spraying, find the patches it falls into, and walk each patch in
    one unbroken route. The angles come out of aim() afterwards.

    Valve cycles drop from hundreds to one per patch.
    """
    x0, y0, x1, y1 = bed
    free, nx, ny = free_grid(bed, plants, cell, clearance, grid)

    # Drop anything the machine cannot physically reach from the rail.
    reach_lo, reach_hi = rail["lo"], rail["hi"]
    for j in range(ny):
        for i in range(nx):
            if not free[j, i]:
                continue
            cx = x0 + (i + 0.5) * cell
            cy = y0 + (j + 0.5) * cell
            along = cy if rail["axis"] == "y" else cx
            pos = min(max(along, reach_lo), reach_hi)
            noz = nozzle_at(rail, pos, height, pan_limit, tilt_min, tilt_max,
                            throw, offset_along, offset_across)
            if not A.solve(noz, (cx, cy), plants, clearance).ok:
                free[j, i] = False

    label, n_patches = components(free)

    steps, opens = [], 0
    for patch in range(1, n_patches + 1):
        route = tour(label, patch)
        if not route:
            continue
        opens += 1
        for j, i in route:
            cx = x0 + (i + 0.5) * cell
            cy = y0 + (j + 0.5) * cell
            along = cy if rail["axis"] == "y" else cx
            pos = min(max(along, reach_lo), reach_hi)
            noz = nozzle_at(rail, pos, height, pan_limit, tilt_min, tilt_max,
                            throw, offset_along, offset_across)
            shot = A.aim(noz, (cx, cy))
            steps.append([patch - 1, round(pos, 1), round(shot.pan_deg, 1),
                          round(shot.tilt_deg, 1), round(cx, 1), round(cy, 1),
                          1, 0])

    return steps, n_patches, free


def draw_route(path, bed, plants, rail, steps, patches, cell=2.0,
               grid=None, clearance=0.0):
    """
    The route the aim point takes, drawn as one continuous line.

    This is the picture to approve before anything moves. It shows exactly
    where mulch will land and, more usefully, that the line never crosses a
    plant - which is the whole claim the valve plan rests on.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.lines import Line2D
    except ImportError:
        return None

    surface, ink, soft = "#fcfcfb", "#0b0b0b", "#52514e"
    wet, plant_c = "#2a78d6", "#1baf7a"
    x0, y0, x1, y1 = bed
    sx, sy = x1 - x0, y1 - y0
    k = 7.0 / max(sx, sy)

    fig, ax = plt.subplots(figsize=(max(sx * k + 1.4, 3.2), max(sy * k + 1.0, 3.0)))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.add_patch(Rectangle((x0, y0), sx, sy, facecolor="#f1efea",
                           edgecolor=soft, lw=1.3, zorder=1))

    # The map's own boxes underneath, so it is obvious the spray only ever
    # enters the empty ones.
    if grid:
        cols, rows = grid
        boxes, occupied = map_boxes(bed, plants, cols, rows, clearance)
        for j in range(rows):
            for i in range(cols):
                bx0, by0, bx1, by1 = boxes[j][i]
                taken = occupied[j, i]
                ax.add_patch(Rectangle(
                    (bx0, by0), bx1 - bx0, by1 - by0,
                    facecolor="#eb6834" if taken else "#ffffff",
                    alpha=.22 if taken else .55,
                    edgecolor=soft, lw=.8, zorder=2))
                ax.text((bx0 + bx1) / 2, by1 - (by1 - by0) * 0.12,
                        f"C{i+1}R{j+1}", ha="center", va="top",
                        fontsize=7, color=soft, alpha=.75, zorder=3)

    # The route, coloured by which patch it belongs to.
    for patch in sorted({s[0] for s in steps}):
        pts = [(s[4], s[5]) for s in steps if s[0] == patch]
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                lw=1.6, color=wet, alpha=.85, solid_capstyle="round", zorder=6)
        ax.scatter([pts[0][0]], [pts[0][1]], s=70, c=surface,
                   edgecolors=wet, linewidths=2.2, zorder=8)
        ax.annotate("valve on", (pts[0][0], pts[0][1]),
                    textcoords="offset points", xytext=(8, 7),
                    fontsize=8, color=soft, zorder=6)

    for p in plants:
        ax.add_patch(Circle((p.x_cm, p.y_cm), A.footprint_radius(p),
                            facecolor=plant_c, alpha=.85, edgecolor=surface,
                            lw=2, zorder=7))

    if rail["axis"] == "y":
        ax.plot([rail["fixed"]] * 2, [rail["lo"], rail["hi"]],
                color=soft, lw=2.5, zorder=2)
    else:
        ax.plot([rail["lo"], rail["hi"]], [rail["fixed"]] * 2,
                color=soft, lw=2.5, zorder=2)

    pad = 0.08 * max(sx, sy)
    ax.set_xlim(min(x0, rail["fixed"] if rail["axis"] == "y" else x0) - pad, x1 + pad)
    ax.set_ylim(min(y0, rail["fixed"] if rail["axis"] == "x" else y0) - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Spray route - approve before running",
                 fontsize=12, color=ink, loc="left", pad=14)
    ax.text(0.0, 1.012,
            f"{patches} patch(es)  -  valve opens {patches} time(s)  -  "
            f"{len(steps):,} points",
            transform=ax.transAxes, fontsize=8.5, color=soft)
    ax.set_xlabel("x  (cm)", fontsize=9, color=soft)
    ax.set_ylabel("y  (cm)", fontsize=9, color=soft)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(soft)
    ax.tick_params(colors=soft, labelsize=8)
    ax.legend(handles=[
        Line2D([0], [0], color=wet, lw=2, label="spray path"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=plant_c,
               markersize=10, label="plant - never crossed"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#eb6834",
               alpha=.4, markersize=11, label="box holds a crop"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ffffff",
               markeredgecolor=soft, markersize=11, label="box is clear"),
        Line2D([0], [0], color=soft, lw=2.5, label="rail"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=8.5, labelcolor=soft)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return path






def reach_into_bed(bed, rail, height, throw, tilt_min, tilt_max):
    """
    How far across the bed the sprayer can actually land mulch.

    The throw is a SLANT distance - the nozzle is up in the air, so the
    height is part of it and only what is left over reaches across the
    ground. Standing the rail off the bed edge spends more of it again.

        ground it can cover = sqrt(throw^2 - height^2) - standoff

    A 12 inch throw therefore does not cover a 12 inch bed from outside
    it, which is easy to miss when both numbers are the same.
    """
    fixed = rail["fixed"]

    # Which bed edge the rail is parked against. It can be mounted on
    # either side, so take whichever edge it is nearer to as the near one
    # and measure across to the other.
    lo, hi = (bed[0], bed[2]) if rail["axis"] == "y" else (bed[1], bed[3])

    # The head can sit inside the bed rather than off one edge. Then
    # there is no near edge and no far edge - there are two sides, and
    # what has to be reached is whichever is further away.
    inside = lo < fixed < hi
    if inside:
        near_edge = fixed
        far_edge = hi if (hi - fixed) >= (fixed - lo) else lo
    elif abs(fixed - lo) <= abs(fixed - hi):
        near_edge, far_edge = lo, hi
    else:
        near_edge, far_edge = hi, lo
    width = abs(far_edge - near_edge)
    standoff = abs(near_edge - fixed)
    if inside:
        # Report the whole bed, since both sides still have to be covered.
        width = abs(hi - lo)

    if throw <= height:
        flat = 0.0
    else:
        flat = math.sqrt(throw ** 2 - height ** 2)

    # The tilt stops can bite before the throw does, so take whichever
    # limit is tighter.
    if tilt_min > 0:
        flat = min(flat, height / math.tan(math.radians(tilt_min)))

    if inside:
        # Reaching out both ways: what has to be covered is the longer
        # side, and what IS covered is each side capped at the reach.
        needed = max(hi - fixed, fixed - lo)
        into = min(flat, needed)
        covered = min(flat, fixed - lo) + min(flat, hi - fixed)
        short = max(0.0, needed - flat)
        return {
            "width": width, "standoff": 0.0, "flat": flat, "into": into,
            "short": short,
            "pct": max(0.0, min(100.0, covered / width * 100)) if width else 0.0,
            "limit": "tilt" if tilt_min > 0 and height / math.tan(
                math.radians(tilt_min)) < math.sqrt(
                    max(0.0, throw ** 2 - height ** 2)) else "throw",
            "inside": True,
        }

    into = flat - standoff
    return {
        "width": width,
        "standoff": standoff,
        "flat": flat,
        "into": into,
        "short": max(0.0, width - into),
        "pct": max(0.0, min(100.0, into / width * 100)) if width else 0.0,
        "limit": "tilt" if tilt_min > 0 and height / math.tan(
            math.radians(tilt_min)) < math.sqrt(max(0.0, throw ** 2 - height ** 2))
            else "throw",
    }


def report_reach(bed, rail, height, throw, tilt_min, tilt_max):
    """
    Say up front how much of the bed is out of range, and why.

    Shares reach_advice with the planner rather than keeping a second
    copy: the advice has been wrong once already by being worked out two
    different ways in two places.
    """
    r = reach_into_bed(bed, rail, height, throw, tilt_min, tilt_max)
    axis = "x" if rail["axis"] == "y" else "y"

    print(f"  Reach check")
    print(f"    Bed is {r['width']:.1f} cm across, rail stands "
          f"{r['standoff']:.1f} cm off the edge")
    print(f"    Nozzle {height:.1f} cm up, {throw:.1f} cm throw "
          f"-> {r['flat']:.1f} cm of ground ({r['limit']} limited)")
    print(f"    Covers {axis} up to {r['into']:.1f} cm "
          f"= {r['pct']:.0f}% of the bed")

    if r["short"] > 0.5:
        print("")
        reach_advice(r, height, throw, tilt_min)
    return r


def best_height(throw, tilt_min_deg):
    """
    The nozzle height that reaches furthest, and how far that is.

    Two limits fight each other as the nozzle moves up and down:

        throw limit : sqrt(throw^2 - h^2)   - falls as h rises
        tilt limit  : h / tan(tilt_min)     - rises as h rises

    So reach is worst at both extremes and best where they cross. Setting
    them equal gives h = throw * sin(tilt_min), and at that height the
    reach is throw * cos(tilt_min).

    This is why "mount it lower" is not automatically good advice: below
    the crossing point the tilt limit takes over and reach drops again.
    """
    t = math.radians(tilt_min_deg)
    return throw * math.sin(t), throw * math.cos(t)


def reach_advice(r, height, throw, tilt_min):
    """
    What to change when the far side of the bed is out of range.

    Every option is costed through reach_into_bed rather than a one-line
    formula, because which limit binds changes as the numbers move and a
    throw-only estimate quietly recommends the wrong direction.
    """
    span = r["width"] + r["standoff"]
    print(f"              SHORT BY {r['short']:.1f} cm - the far side cannot "
          f"be sprayed from this edge.")
    print(f"              Options, best first:")

    h_opt, reach_opt = best_height(throw, tilt_min)
    options = []

    # Cost tiers, because reach alone would rank "buy a bigger airbrush"
    # above "slide the rail 8 cm". 0 = free, just remount or retune.
    # 1 = a second pass, costs run time. 2 = new hardware.
    if r["standoff"] > 0.5:
        options.append((0, min(r["flat"], r["width"]),
                        f"rail flush with the bed edge instead of "
                        f"{r['standoff']:.1f} cm out",
                        f"gains {r['standoff']:.1f} cm, costs nothing"))

    if abs(height - h_opt) > 0.3:
        gain = (reach_opt - r["standoff"]) - r["into"]
        if gain > 0.2:
            options.append((0, reach_opt - r["standoff"],
                            f"nozzle at {h_opt:.1f} cm rather than "
                            f"{height:.0f} cm",
                            f"best height for a {throw:.0f} cm throw at "
                            f"{tilt_min:.0f} deg, gains {gain:.1f} cm"))

    options.append((1, r["width"],
                    "spray the far side from the opposite edge",
                    "two passes, no mechanical change"))
    need_throw = span / math.cos(math.radians(tilt_min))
    options.append((2, r["width"],
                    f"a throw of {need_throw:.1f} cm",
                    "what it would take to cover the bed as mounted"))

    options.sort(key=lambda o: (o[0], -o[1]))
    for _, _, what, why in options:
        print(f"                - {what}")
        print(f"                  {why}")

    # What the free changes get you together, which is the number that
    # actually decides whether any hardware needs buying.
    free = min(reach_opt, span) - 0.0
    if r["standoff"] > 0.5 or abs(height - h_opt) > 0.3:
        pct = min(100.0, free / r["width"] * 100)
        print(f"              Both free changes together: {free:.1f} cm "
              f"= {pct:.0f}% of the bed.")

    if r["limit"] == "tilt":
        print(f"              Note: the TILT stop is binding here, not the "
              f"throw. A bigger")
        print(f"              airbrush would change nothing until "
              f"--tilt-min drops.")


if __name__ == "__main__":
    with safestop.guard("route planning"):
        main()
