"""
The whole spraying half, one command.

    python spray.py

Picks a map, asks for the machine's measurements, plans the route, shows it
to you, and only then writes the motor and valve commands.

    map  ->  route  ->  [YOU APPROVE]  ->  dc_commands.csv

WHY THERE IS A STOP IN THE MIDDLE
    Everything before the approval only writes pictures and a plan file.
    Everything after it writes the instructions a machine executes.

    The planner cannot tell you the numbers you gave it were wrong. Feed it
    a plot size that does not match the bed, or a throw longer than the
    sprayer manages, and it produces a clean route that is fiction. The
    route picture is the only thing that catches that, which is why it is
    put in front of you before the commands exist.

NOTHING HERE MOVES ANYTHING
    The last step writes a CSV. Sending that to the hardware is a separate,
    deliberate act.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import presets as P
import safestop

HERE = pathlib.Path(__file__).parent
MAPS = HERE.parent / "mapping" / "maps"

# Measured on this machine. Change the defaults here if the build changes,
# rather than retyping them every run.
DEFAULTS = {
    "rail_length": 35.0,     # 350 mm carriage travel
    "throw": 30.5,           # 12 inches, measured level
    "height": 14.0,          # nozzle above the soil
    "offset_across": 3.6,    # 36 mm, camera lens to nozzle
    "tilt_min": 18.0,
    "tilt_max": 80.0,
    "clearance": 2.0,        # keep-off margin round each plant
    "cell": 1.5,             # route resolution
}


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
        sys.exit(f"No maps yet. Run mapping/new_bed.py first.")

    found = []
    for d in sorted(MAPS.iterdir(), key=lambda p: -p.stat().st_mtime):
        f = d / f"{d.name}.json"
        if f.exists():
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            found.append((d.name, f, len(doc.get("plants", [])),
                          doc.get("counts", {})))

    if not found:
        sys.exit("No maps yet. Run mapping/new_bed.py first.")

    print("\nMaps, newest first:\n")
    for i, (name, _f, n, counts) in enumerate(found[:15], 1):
        crops = counts.get("crop", 0)
        note = f"  ({crops} crop)" if crops else ""
        print(f"  {i:2d}. {name:20s} {n:2d} plants{note}")
    print()

    raw = input("Which one? (number, or part of the name): ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(found):
        return found[int(raw) - 1]

    matches = [f for f in found if raw.lower() in f[0].lower()]
    if len(matches) == 1:
        print(f"  using {matches[0][0]}")
        return matches[0]
    if len(matches) > 1:
        sys.exit(f"'{raw}' matches {len(matches)} maps - be more specific")
    sys.exit(f"No map matching '{raw}'")


def run(cmd):
    print()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"\nThat step failed (exit {r.returncode}). Stopping.")


def main():
    ap = argparse.ArgumentParser(description="Plan and write a spray run.")
    P.add_arguments(ap)
    args = ap.parse_args()
    if P.handle_flags(args):
        return

    a = P.Answers(P.get(args.preset) if args.preset else None)
    if args.preset and not a.preset:
        sys.exit(f"No preset called '{args.preset}'. "
                 f"Try --list-presets.")

    print("=" * 62)
    print("SPRAY RUN" + (f"   preset: {args.preset}" if args.preset else ""))
    print("=" * 62)

    if "map" in a.preset:
        name = a.preset["map"]
        a.record("map", name)
        print(f"\nMap: {name}   (preset)")
        f = MAPS / name / f"{name}.json"
        if not f.exists():
            sys.exit(f"The preset names map '{name}', which does not exist.")
        n_plants = len(json.loads(f.read_text()).get("plants", []))
    else:
        name, map_json, n_plants, _counts = pick_map()
        a.record("map", name)

    print(f"\n{name}: {n_plants} plants")
    print("\nThe plot, in centimetres. These MUST match what you gave the")
    print("mapping run, or every position shifts.")
    width = a.ask("width", "  width cm  (near edge)", 35.0, float)
    depth = a.ask("depth", "  depth cm  (near to far)", 30.0, float)

    print("\nThe machine. Enter through these unless something changed.")
    rail = a.ask("rail", "  rail travel cm", DEFAULTS["rail_length"], float)
    throw = a.ask("throw", "  sprayer throw cm", DEFAULTS["throw"], float)
    height = a.ask("height", "  nozzle height cm", DEFAULTS["height"], float)
    offset = a.ask("offset", "  camera to nozzle cm",
                   DEFAULTS["offset_across"], float)
    clearance = a.ask("clearance", "  keep-off margin round plants cm",
                      DEFAULTS["clearance"], float)

    side = ask("\nWhich edge is the rail on? (left/right/near/far)", "left")
    grid = a.ask("grid", "Grid, as COLSxROWS  (blank for none)", "5x2")

    out = f"plans/{name}"
    cmd = [sys.executable, str(HERE / "simulate.py"),
           f"../mapping/maps/{name}/{name}.json",
           "--bed", "0", "0", str(width), str(depth),
           "--side", side,
           "--rail-length", str(rail),
           "--throw", str(throw),
           "--height", str(height),
           "--tilt-min", str(DEFAULTS["tilt_min"]),
           "--tilt-max", str(DEFAULTS["tilt_max"]),
           "--offset-across", str(offset),
           "--clearance", str(clearance),
           "--continuous", "--cell", str(DEFAULTS["cell"]),
           "--out", out]
    if grid and grid.lower() != "none":
        cmd += ["--grid", grid, "--grid-only"]

    run(cmd)

    route = HERE / out / "spray_route.png"
    print("\n" + "-" * 62)
    print(f"  {route}")
    print("-" * 62)
    print("Open it. One unbroken line that never enters a crop box is what")
    print("you want. If it crosses a plant, say no and check the numbers.")

    if not input("\nApprove this route? (yes / no): ").strip().lower() \
            in ("y", "yes"):
        print("\nStopped. No commands were written.")
        if args.save_preset:
            print("Preset not saved - the route was not approved.")
        return

    # Only save settings that produced a route you accepted. A preset is a
    # claim that these numbers work, so a rejected run must not make one.
    if args.save_preset:
        where = P.save(args.save_preset, a.used)
        print(f"\nSaved preset '{args.save_preset}' to {where.name}")

    run([sys.executable, str(HERE / "motors.py"), f"{out}/plan.json", "--dc"])
    run([sys.executable, str(HERE / "motor_sim.py"), "dc_commands.csv",
         "--plan", f"{out}/plan.json"])

    print("\n" + "=" * 62)
    print("The machine's instructions are in:")
    print("  spraying/dc_commands.csv")
    print("\nNothing has moved. Sending that file to the hardware is a")
    print("separate step.")
    print("=" * 62)


if __name__ == "__main__":
    with safestop.guard("spray planning"):
        main()
