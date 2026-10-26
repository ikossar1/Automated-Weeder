"""
Simulate the three motors executing a command stream.

    python motor_sim.py motor_commands.csv

Reads the commands motors.py wrote and works out what the axes actually do:
how far each travels, how fast it has to move, how long the run takes, and
where the time goes.

WHY
    A command list says where to be, not whether the machine can get there.
    This applies a speed and acceleration limit to each axis and reports the
    result, so a plan that needs the pan motor to snap 140 degrees between
    consecutive steps shows up here rather than on the bench.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib


class AxisLimits:
    """Speed and acceleration ceiling for one axis, in steps."""

    def __init__(self, name, max_steps_per_s, accel_steps_per_s2):
        self.name = name
        self.vmax = float(max_steps_per_s)
        self.accel = float(accel_steps_per_s2)

    def move_time(self, steps):
        """Seconds for a move of this many steps, trapezoidal profile."""
        d = abs(float(steps))
        if d == 0:
            return 0.0
        # Distance needed to reach full speed and stop again.
        ramp = self.vmax ** 2 / self.accel
        if d < ramp:
            return 2.0 * math.sqrt(d / self.accel)       # never reaches vmax
        return self.vmax / self.accel + d / self.vmax    # ramp + cruise


DEFAULT_LIMITS = {
    "rail": AxisLimits("rail", max_steps_per_s=8000, accel_steps_per_s2=40000),
    "pan": AxisLimits("pan", max_steps_per_s=4000, accel_steps_per_s2=20000),
    "tilt": AxisLimits("tilt", max_steps_per_s=4000, accel_steps_per_s2=20000),
}


def run(csv_path, limits=None, settle_s=0.0):
    limits = limits or DEFAULT_LIMITS
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"{csv_path} has no commands in it")

    axes = ["rail", "pan", "tilt"]

    # Steppers count steps, DC motors with encoders count counts. Same
    # numbers, different column name, so take whichever the file has.
    suffix = "_steps" if f"rail_steps" in rows[0] else "_counts"
    if f"rail{suffix}" not in rows[0]:
        raise SystemExit(f"{csv_path} has no rail_steps or rail_counts column")

    pos = {a: int(rows[0][f"{a}{suffix}"]) for a in axes}
    travel = {a: 0 for a in axes}
    biggest = {a: (0, -1) for a in axes}
    t = 0.0
    timeline = []
    spray_time = 0.0
    valve_flips = 0
    last_spray = None

    for i, r in enumerate(rows):
        want = {a: int(r[f"{a}{suffix}"]) for a in axes}
        deltas = {a: want[a] - pos[a] for a in axes}

        # The axes move together, so the step takes as long as the slowest.
        times = {a: limits[a].move_time(deltas[a]) for a in axes}
        dt = max(times.values()) + settle_s
        slow = max(times, key=times.get)

        for a in axes:
            travel[a] += abs(deltas[a])
            if abs(deltas[a]) > biggest[a][0]:
                biggest[a] = (abs(deltas[a]), i)

        spray = r["spray"] == "1"
        if last_spray is not None and spray != last_spray:
            valve_flips += 1
        last_spray = spray
        if spray:
            spray_time += dt

        t += dt
        pos = want
        timeline.append((t, dt, slow, spray))

    return {
        "commands": len(rows),
        "seconds": t,
        "travel_steps": travel,
        "biggest_jump": biggest,
        "spray_seconds": spray_time,
        "valve_flips": valve_flips,
        "timeline": timeline,
        "limited_by": {a: sum(1 for _, _, s, _ in timeline if s == a) for a in axes},
    }


def report(res, head=None):
    t = res["seconds"]
    print(f"Commands      : {res['commands']:,}")
    print(f"Run time      : {t:,.1f} s  ({t/60:.1f} min)")
    print(f"Spraying for  : {res['spray_seconds']:,.1f} s "
          f"({100*res['spray_seconds']/max(t,1e-9):.0f}% of the run)")
    print(f"Valve changes : {res['valve_flips']:,}")

    print("\nTravel per axis:")
    for a, steps in res["travel_steps"].items():
        line = f"  {a:5s} {steps:>12,} steps"
        if head:
            unit = "cm" if a == "rail" else "deg"
            per = getattr(head, a).steps_per_unit
            line += f"   {steps/per:>12,.0f} {unit}"
        print(line)

    print("\nWhich axis sets the pace:")
    for a, n in sorted(res["limited_by"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:5s} slowest on {n:>7,} of {res['commands']:,} steps"
              f"  ({100*n/res['commands']:.0f}%)")

    print("\nLargest single move per axis:")
    for a, (steps, i) in res["biggest_jump"].items():
        print(f"  {a:5s} {steps:>10,} steps  at command {i:,}")


def draw(res, path="motor_timeline.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    surface, ink, soft = "#fcfcfb", "#0b0b0b", "#52514e"
    blue, orange = "#2a78d6", "#eb6834"

    ts = [p[0] for p in res["timeline"]]
    dts = [p[1] * 1000 for p in res["timeline"]]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    spray = [p[3] for p in res["timeline"]]
    ax.scatter([t for t, s in zip(ts, spray) if s],
               [d for d, s in zip(dts, spray) if s],
               s=3, c=blue, linewidths=0, alpha=.5, label="spraying")
    ax.scatter([t for t, s in zip(ts, spray) if not s],
               [d for d, s in zip(dts, spray) if not s],
               s=3, c=orange, linewidths=0, alpha=.5, label="held off")

    ax.set_title("Time per move", fontsize=12, color=ink, loc="left", pad=12)
    ax.set_xlabel("elapsed  (s)", fontsize=9, color=soft)
    ax.set_ylabel("this move took  (ms)", fontsize=9, color=soft)
    ax.grid(True, lw=.5, alpha=.18, color=soft)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(soft)
    ax.tick_params(colors=soft, labelsize=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=soft, markerscale=3)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Simulate motors running a plan.")
    ap.add_argument("commands", nargs="?", default="motor_commands.csv")
    ap.add_argument("--rail-vmax", type=float, default=8000)
    ap.add_argument("--rot-vmax", type=float, default=4000)
    ap.add_argument("--rail-accel", type=float, default=40000)
    ap.add_argument("--rot-accel", type=float, default=20000)
    ap.add_argument("--plan", default=None,
                    help="the plan.json these commands came from. Needed for "
                         "the valve timing check, which uses landing points")
    ap.add_argument("--valve-open-ms", type=float, default=25.0)
    ap.add_argument("--valve-close-ms", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=0.0,
                    help="extra pause after each move, s")
    args = ap.parse_args()

    limits = {
        "rail": AxisLimits("rail", args.rail_vmax, args.rail_accel),
        "pan": AxisLimits("pan", args.rot_vmax, args.rot_accel),
        "tilt": AxisLimits("tilt", args.rot_vmax, args.rot_accel),
    }

    res = run(args.commands, limits, args.settle)

    try:
        import motors
        head = motors.DEFAULT_HEAD
    except Exception:
        head = None

    report(res, head)

    if args.plan:
        v = valve_lag(args.commands, args.plan, args.valve_open_ms,
                      args.valve_close_ms, limits, args.settle)
        report_valve(v, args.valve_open_ms, args.valve_close_ms)

    p = draw(res)
    if p:
        print(f"\nWrote {p}")


# ----------------------------------------------------------------------------
# Valve timing
# ----------------------------------------------------------------------------

def valve_lag(csv_path, plan_json, open_ms=25.0, close_ms=25.0, limits=None,
              settle_s=0.0):
    """
    How far the aim point moves while the valve is still catching up.

    THE PROBLEM
        The plan says "spray at step 412, stop at step 530". A valve does
        not switch the instant it is told to - it takes a few tens of
        milliseconds to open and usually a little longer to close. The head
        keeps moving the whole time.

        Open late and the start of the patch is missed. Close late and
        mulch keeps coming after the aim point has moved on - which is the
        dangerous one, because the aim point moves on TOWARD whatever was
        next, and sometimes that is a plant.

    WHAT THIS MEASURES
        For every place the valve changes state, how far the landing point
        travels during the lag, and whether a late close lands on a plant.

    THE FIX, WHICH IS FREE
        Command the valve EARLY by its own lag - open `open_ms` before the
        step that needs it, close `close_ms` before the last spraying step
        ends. The lag is then absorbed instead of accumulated. That is a
        change in the motor code, not here, but this tells you how much
        lead each edge needs.
    """
    limits = limits or DEFAULT_LIMITS
    plan = json.loads(pathlib.Path(plan_json).read_text())
    steps = plan["steps"]
    plants = plan.get("plants", [])

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    suffix = "_steps" if "rail_steps" in rows[0] else "_counts"
    axes = ["rail", "pan", "tilt"]

    pos = {a: int(rows[0][f"{a}{suffix}"]) for a in axes}
    edges = []
    prev_spray = None

    n = min(len(rows), len(steps))
    for i in range(n):
        r = rows[i]
        want = {a: int(r[f"{a}{suffix}"]) for a in axes}
        dt = max(limits[a].move_time(want[a] - pos[a]) for a in axes) + settle_s
        pos = want

        spray = r["spray"] == "1"
        if prev_spray is not None and spray != prev_spray:
            lag = (open_ms if spray else close_ms) / 1000.0

            # Ground speed of the aim point right here.
            j = min(i + 1, n - 1)
            dist = math.hypot(steps[j][4] - steps[i][4], steps[j][5] - steps[i][5])
            speed = dist / dt if dt > 1e-9 else 0.0
            slip = speed * lag

            # Where it would land if the valve were that late.
            land = (steps[j][4], steps[j][5])
            hit = None
            if not spray:               # a late CLOSE is the risky one
                for p in plants:
                    if math.hypot(land[0] - p["x"], land[1] - p["y"]) <= p["r"] + slip:
                        hit = p["id"]
                        break

            edges.append({"i": i, "opening": spray, "slip_cm": slip,
                          "speed_cms": speed, "on_plant": hit})
        prev_spray = spray

    opens = [e for e in edges if e["opening"]]
    closes = [e for e in edges if not e["opening"]]
    risky = [e for e in closes if e["on_plant"]]

    return {
        "edges": len(edges),
        "worst_open_cm": max((e["slip_cm"] for e in opens), default=0.0),
        "worst_close_cm": max((e["slip_cm"] for e in closes), default=0.0),
        "mean_close_cm": (sum(e["slip_cm"] for e in closes) / len(closes)
                          if closes else 0.0),
        "late_close_on_plant": len(risky),
        "which_plants": sorted({e["on_plant"] for e in risky}),
    }


def report_valve(v, open_ms, close_ms):
    print(f"\nValve timing  ({open_ms:.0f} ms to open, {close_ms:.0f} ms to close)")
    print(f"  State changes      : {v['edges']:,}")
    print(f"  Worst slip opening : {v['worst_open_cm']:.2f} cm")
    print(f"  Worst slip closing : {v['worst_close_cm']:.2f} cm")
    print(f"  Average on close   : {v['mean_close_cm']:.2f} cm")

    if v["late_close_on_plant"]:
        print(f"  LATE CLOSE LANDS ON A PLANT: {v['late_close_on_plant']} time(s)"
              f"  plants {v['which_plants']}")
        print(f"  Command the valve shut {close_ms:.0f} ms early, or slow the "
              f"sweep where it matters.")
    else:
        print(f"  No late close reaches a plant.")


if __name__ == "__main__":
    main()


