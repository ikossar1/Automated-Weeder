"""
Prove the aiming maths before anything gets machined.

    python verify_aim.py

HOW YOU CAN TEST AIMING WITH NO HARDWARE
    aim() and spray_point() are exact inverses. aim() turns a ground point
    into angles; spray_point() turns angles back into a ground point. Run
    one into the other and you must get back exactly what you started with.

    That is not a plausibility check, it is a proof. If the trigonometry is
    wrong anywhere - a swapped sine, a sign on the pan, degrees where
    radians belong - the point comes back somewhere else, and the error is
    enormous rather than subtle. A correct implementation returns the same
    number to about 1e-13 cm, which is floating-point noise.

    This is the same trick as validate.py on the camera side: build a
    measurement you can check against something you already know, instead
    of eyeballing whether the output looks about right.

    It cannot tell you the nozzle is really 20 cm up, or that the airbrush
    actually throws 35 cm. Those are measurements for the bench. What it
    does tell you is that given correct numbers, the maths does the right
    thing - so when the hardware misbehaves you can stop suspecting the
    code.

WHAT RUNS HERE
    1. round trip        aim -> spray_point over thousands of targets
    2. known answers     cases worked out by hand
    3. limits            targets too near, too far, too far round
    4. line of sight     blocking, not blocking, behind, beyond
    5. coverage          a simulated bed, with a picture to look at
"""
from __future__ import annotations

import math
import random

import numpy as np

import aim as A

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"  [{PASS if condition else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    return bool(condition)


# ----------------------------------------------------------------------------
# 1. The round trip
# ----------------------------------------------------------------------------

def test_round_trip():
    print("\n1. ROUND TRIP  -  aim() then spray_point() must return the target")
    print("   If this passes, the trigonometry is right. There is no way to")
    print("   get the target back from the angles by accident.\n")

    rng = random.Random(11)
    worst, worst_case = 0.0, None
    tested = 0

    for _ in range(4000):
        n = A.Nozzle(rng.uniform(-50, 50), rng.uniform(-50, 50),
                     rng.uniform(5, 60), heading_deg=rng.uniform(-180, 180),
                     pan_limit_deg=180, tilt_min_deg=0.5, tilt_max_deg=89.5,
                     max_range_cm=1e6)

        tx = n.x_cm + rng.uniform(-200, 200)
        ty = n.y_cm + rng.uniform(-200, 200)
        if math.hypot(tx - n.x_cm, ty - n.y_cm) < 0.5:
            continue

        shot = A.aim(n, (tx, ty))
        back = A.spray_point(n, shot.pan_deg, shot.tilt_deg)
        if back is None:
            continue

        err = math.hypot(back[0] - tx, back[1] - ty)
        tested += 1
        if err > worst:
            worst, worst_case = err, (n, tx, ty, shot)

    check(f"{tested} random targets return to within 1e-9 cm",
          worst < 1e-9, f"worst {worst:.3e} cm")

    if worst_case and worst >= 1e-9:
        n, tx, ty, shot = worst_case
        print(f"        worst case: {n}  target ({tx:.3f}, {ty:.3f})  {shot}")


# ----------------------------------------------------------------------------
# 2. Cases you can work out on paper
# ----------------------------------------------------------------------------

def test_known_answers():
    print("\n2. KNOWN ANSWERS  -  cases worked out by hand\n")

    # Nozzle 20 cm up at the origin, pan 0 pointing along +y.
    n = A.Nozzle(0, 0, 20, heading_deg=90, pan_limit_deg=90,
                 tilt_min_deg=1, tilt_max_deg=89, max_range_cm=500)

    s = A.aim(n, (0, 20))
    check("straight ahead at range = height -> pan 0, tilt 45",
          abs(s.pan_deg) < 1e-9 and abs(s.tilt_deg - 45) < 1e-9,
          f"pan {s.pan_deg:.6f}, tilt {s.tilt_deg:.6f}")

    s = A.aim(n, (0, 40))
    want = math.degrees(math.atan2(20, 40))
    check("twice the height out -> tilt = atan(h/2h) = 26.565",
          abs(s.tilt_deg - want) < 1e-9, f"tilt {s.tilt_deg:.6f}")

    s = A.aim(n, (20, 0))
    check("90 deg to the right of heading -> pan -90",
          abs(s.pan_deg + 90) < 1e-9, f"pan {s.pan_deg:.6f}")

    s = A.aim(n, (-20, 0))
    check("90 deg to the left of heading -> pan +90",
          abs(s.pan_deg - 90) < 1e-9, f"pan {s.pan_deg:.6f}")

    # Steeper tilt must always land shorter - the sign nobody checks.
    d1 = A.spray_point(n, 0, 30)
    d2 = A.spray_point(n, 0, 60)
    check("steeper tilt lands nearer",
          math.hypot(*d2) < math.hypot(*d1),
          f"30deg -> {math.hypot(*d1):.1f} cm, 60deg -> {math.hypot(*d2):.1f} cm")

    # Moving the nozzle must move the landing point with it, exactly.
    moved = n.at(x_cm=10)
    p0 = A.spray_point(n, 15, 40)
    p1 = A.spray_point(moved, 15, 40)
    check("sliding the rail 10 cm shifts the landing point 10 cm",
          abs((p1[0] - p0[0]) - 10) < 1e-9 and abs(p1[1] - p0[1]) < 1e-9,
          f"moved ({p1[0]-p0[0]:.6f}, {p1[1]-p0[1]:.6f})")


# ----------------------------------------------------------------------------
# 3. Limits
# ----------------------------------------------------------------------------

def test_limits():
    print("\n3. LIMITS  -  unreachable targets must be refused, with a reason\n")

    n = A.Nozzle(0, 0, 20, heading_deg=90, pan_limit_deg=60,
                 tilt_min_deg=15, tilt_max_deg=75, max_range_cm=50)
    near, far = n.reach()
    print(f"   this nozzle reaches {near:.1f} to {far:.1f} cm\n")

    s = A.aim(n, (0, near * 0.5))
    check("too close is refused", not s.ok and "close" in s.reason, s.reason)

    s = A.aim(n, (0, far * 2))
    check("too far is refused", not s.ok, s.reason)

    s = A.aim(n, (100, 0))
    check("outside the pan range is refused",
          not s.ok and "pan" in s.reason, s.reason)

    mid = (near + far) / 2
    s = A.aim(n, (0, mid))
    check("a target in the middle of the envelope is accepted",
          s.ok, f"{s}")

    s = A.aim(n, (0, 0))
    check("directly underneath is refused rather than dividing by zero",
          not s.ok, s.reason)


# ----------------------------------------------------------------------------
# 4. Line of sight
# ----------------------------------------------------------------------------

class FakePlant:
    def __init__(self, pid, x, y, radius):
        self.id = pid
        self.x_cm = x
        self.y_cm = y
        self.size_cm2 = math.pi * radius * radius


def test_line_of_sight():
    print("\n4. LINE OF SIGHT  -  a plant blocks only when it is in the way\n")

    n = A.Nozzle(0, 0, 20, heading_deg=90, pan_limit_deg=90,
                 tilt_min_deg=1, tilt_max_deg=89, max_range_cm=500)
    target = (0, 40)

    check("plant sitting on the line blocks it",
          A.blocked_by(n, target, [FakePlant(1, 0, 20, 5)], 0) is not None)

    check("plant well to the side does not block",
          A.blocked_by(n, target, [FakePlant(2, 12, 20, 5)], 0) is None)

    check("plant just clear of the line does not block",
          A.blocked_by(n, target, [FakePlant(3, 5.6, 20, 5)], 0) is None)

    check("plant just touching the line does block",
          A.blocked_by(n, target, [FakePlant(4, 4.4, 20, 5)], 0) is not None)

    check("plant BEHIND the nozzle does not block",
          A.blocked_by(n, target, [FakePlant(5, 0, -20, 5)], 0) is None)

    check("plant BEYOND the target does not block",
          A.blocked_by(n, target, [FakePlant(6, 0, 60, 5)], 0) is None)

    check("clearance grows the plant and makes a near miss a block",
          A.blocked_by(n, target, [FakePlant(7, 5.6, 20, 5)], 2.0) is not None)

    check("the target plant itself can be skipped",
          A.blocked_by(n, (0, 20), [FakePlant(8, 0, 20, 5)], 0,
                       skip_ids=(8,)) is None)

    blocker = FakePlant(9, 0, 20, 5)
    s = A.solve(n, target, [blocker], clearance_cm=0)
    check("solve() reports which plant is in the way",
          not s.ok and "#9" in s.reason, s.reason)

    # The gap the simulator found: blocked_by correctly ignores a plant at
    # the far end of the path, so nothing was stopping the sprayer from
    # aiming straight AT a plant and calling it a clear shot.
    sitting = FakePlant(10, 0, 40, 8)
    check("a plant AT the target is not 'in the way'",
          A.blocked_by(n, target, [sitting], 0) is None)
    check("but on_plant() catches it",
          A.on_plant(target, [sitting], 0) is not None)
    s = A.solve(n, target, [sitting], clearance_cm=0)
    check("so solve() refuses to spray a plant",
          not s.ok and "lands on plant #10" in s.reason, s.reason)
    s = A.solve(n, target, [sitting], clearance_cm=0, allow_on_plant=True)
    check("unless you explicitly allow it", s.ok)


# ----------------------------------------------------------------------------
# 5. Coverage on a simulated bed
# ----------------------------------------------------------------------------

def test_coverage(bed=(0, 0, 30.5, 94.0), draw=True):
    print("\n5. COVERAGE  -  a simulated bed, to see what gets reached\n")

    x0, y0, x1, y1 = bed
    plants = [FakePlant(1, 8.0, 20.0, 7.0),
              FakePlant(2, 22.0, 46.0, 9.0),
              FakePlant(3, 12.0, 74.0, 6.0)]

    # Rail along the long edge at x = x0, looking across the bed (+x).
    stops = np.arange(y0 + 4, y1, 8.0)
    nozzles = [A.Nozzle(x0 - 3.0, float(s), 20.0, heading_deg=0.0,
                        pan_limit_deg=70, tilt_min_deg=14, tilt_max_deg=78,
                        max_range_cm=40.0)
               for s in stops]
    print(f"   rail along x = {x0 - 3:.0f} cm, {len(nozzles)} stops every 8 cm")
    print(f"   nozzle 20 cm up, throws 40 cm, pans +/-70 deg\n")

    step = 2.0
    gx = np.arange(x0 + step / 2, x1, step)
    gy = np.arange(y0 + step / 2, y1, step)
    pts = np.array([(a, b) for b in gy for a in gx])

    # A patch is covered if ANY rail stop can reach it.
    covered = np.zeros(len(pts), bool)
    for n in nozzles:
        covered |= A.reachable_mask(n, pts, plants, clearance_cm=1.0)

    # Don't count the plants themselves as bare soil.
    on_plant = np.zeros(len(pts), bool)
    for p in plants:
        r = A.footprint_radius(p)
        on_plant |= (np.hypot(pts[:, 0] - p.x_cm, pts[:, 1] - p.y_cm) <= r)

    soil = ~on_plant
    pct = 100.0 * covered[soil].sum() / max(soil.sum(), 1)

    print(f"   bare soil patches      {soil.sum()}")
    print(f"   reachable from the rail {covered[soil].sum()}  ({pct:.1f}%)")

    one_stop = A.reachable_mask(nozzles[len(nozzles) // 2], pts, plants, 1.0)
    one_pct = 100.0 * one_stop[soil].sum() / max(soil.sum(), 1)
    print(f"   from a SINGLE stop      {one_pct:.1f}%"
          f"   <- why the rail has to move\n")

    check("most of the bare soil is reachable", pct > 80,
          f"{pct:.1f}% covered")
    check("a single stop covers far less than the whole rail",
          one_pct < pct, f"{one_pct:.1f}% vs {pct:.1f}%")

    if draw:
        path = _draw(bed, plants, nozzles, pts, covered, soil, pct)
        if path:
            print(f"\n   picture: {path}")
    return pct


def _draw(bed, plants, nozzles, pts, covered, soil, pct):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.lines import Line2D
    except ImportError:
        print("   (matplotlib not installed - skipping the picture)")
        return None

    surface, ink, soft = "#fcfcfb", "#0b0b0b", "#52514e"
    blue, orange, grey = "#2a78d6", "#eb6834", "#d9d5cd"
    x0, y0, x1, y1 = bed

    fig, ax = plt.subplots(figsize=(5.0, 9.0))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=soft, lw=1.2, zorder=1))

    ok = covered & soil
    no = (~covered) & soil
    ax.scatter(pts[no, 0], pts[no, 1], s=7, c=orange, marker="s",
               linewidths=0, zorder=2)
    ax.scatter(pts[ok, 0], pts[ok, 1], s=7, c=blue, marker="s",
               linewidths=0, alpha=.55, zorder=2)

    for p in plants:
        ax.add_patch(Circle((p.x_cm, p.y_cm), A.footprint_radius(p),
                            facecolor=grey, edgecolor=soft, lw=1.2, zorder=4))

    ax.plot([nozzles[0].x_cm] * 2, [nozzles[0].y_cm, nozzles[-1].y_cm],
            color=soft, lw=2.0, zorder=3)
    ax.scatter([n.x_cm for n in nozzles], [n.y_cm for n in nozzles],
               s=26, c=surface, edgecolors=soft, linewidths=1.4, zorder=5)

    ax.set_xlim(x0 - 8, x1 + 2)
    ax.set_ylim(y0 - 3, y1 + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Sprayer coverage", fontsize=12, color=ink, loc="left", pad=14)
    ax.text(0.0, 1.012, f"{pct:.0f}% of bare soil reachable   -   "
                        f"orange is shadowed",
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
               markersize=9, label="reachable"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=orange,
               markersize=9, label="shadowed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=grey,
               markeredgecolor=soft, markersize=9, label="plant"),
        Line2D([0], [0], marker="o", color=soft, markerfacecolor=surface,
               markeredgecolor=soft, markersize=7, label="nozzle stop"),
    ], loc="lower left", bbox_to_anchor=(0.0, -0.17), ncol=2,
        frameon=False, fontsize=8.5, labelcolor=soft)

    fig.tight_layout()
    fig.savefig("aim_coverage.png", dpi=200, facecolor=surface,
                bbox_inches="tight")
    plt.close(fig)
    return "aim_coverage.png"


# ----------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("VERIFYING THE AIMING MATHS")
    print("=" * 68)

    test_round_trip()
    test_known_answers()
    test_limits()
    test_line_of_sight()
    test_coverage()

    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 68)
    print(f"{passed} of {len(results)} checks passed")
    if passed == len(results):
        print("\nThe geometry is right. What it does NOT prove:")
        print("  - that the nozzle really sits where you say it does")
        print("  - that the airbrush really throws as far as you told it")
        print("  - that the motors land on the angles they are commanded")
        print("Those are bench measurements. Everything above them is sound.")
    else:
        print("\nSomething is wrong above. Fix it before building anything.")
    print("=" * 68)
    return passed == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
