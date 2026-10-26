"""
The whole job, with a look before each irreversible step.

    python run.py

    photo  ->  map  ->  [YOU APPROVE]  ->  spray route  ->  [YOU APPROVE]
           ->  motor and valve commands

WHY TWO GATES
    Both halves can produce a confident, wrong answer with no error.

    The map can be wrong because the four corners were clicked badly - the
    maths cannot tell, and a plant ends up metres from where it is. So look
    at the corner overlay before trusting any position.

    The route can be wrong because the plot size, the rail length or the
    throw do not match the real machine. So look at the drawn route before
    anything moves. The route picture shows the line the aim takes and
    every plant it must avoid; if the line crosses a plant, stop.

    After the second gate the commands are written and a machine can be
    told to execute them. That is the point to be sure.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
MAPPING = ROOT / "mapping"
SPRAYING = ROOT / "spraying"


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


def approve(question, files):
    """Show what was produced, then ask. Anything but yes stops."""
    print()
    print("-" * 62)
    for f in files:
        p = pathlib.Path(f)
        mark = "  " if p.exists() else "  (missing) "
        print(f"{mark}{p}")
    print("-" * 62)
    print("Open those and check them before answering.")
    answer = input(f"\n{question} (yes / no): ").strip().lower()
    if answer not in ("y", "yes"):
        print("\nStopped. Nothing further was run.")
        raise SystemExit(0)
    print()


def run(cmd, cwd):
    print("  $ " + " ".join(
        pathlib.Path(c).name if ("/" in c or "\\" in c) else c for c in cmd[1:]))
    print()
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"\nThat step failed (exit {r.returncode}). Stopping.")


def main():
    print("=" * 62)
    print("AUTOMATED WEEDER  -  full run")
    print("=" * 62)

    # ---- 1. map ----------------------------------------------------------
    print("\nSTEP 1  Map the bed\n")
    name = ask("Run name", "bed1")
    map_json = MAPPING / "maps" / name / f"{name}.json"

    if map_json.exists():
        reuse = input(f"\nmaps/{name}/ already exists. Reuse it? (Y/n): ")
        fresh = reuse.strip().lower().startswith("n")
    else:
        fresh = True

    if fresh:
        print("\nHanding over to new_bed.py - it will ask for the photo,")
        print("the plot size, and the corners.\n")
        run([sys.executable, str(MAPPING / "new_bed.py")], MAPPING)
        name = ask("\nConfirm the run name you used", name)
        map_json = MAPPING / "maps" / name / f"{name}.json"

    if not map_json.exists():
        sys.exit(f"No map at {map_json} - nothing to plan from.")

    approve("Is the map correct?", [
        MAPPING / "maps" / name / f"{name}_corners.png",
        MAPPING / "maps" / name / f"{name}_grid.png",
        MAPPING / "maps" / name / f"{name}.csv",
    ])

    # ---- 2. route --------------------------------------------------------
    print("STEP 2  Plan the spray route\n")
    width = ask("Plot width cm (near edge)", 30.5, float)
    depth = ask("Plot depth cm (near to far)", 35.0, float)
    rail_len = ask("Rail travel cm", 35.0, float)
    throw = ask("Sprayer throw cm", 30.5, float)
    height = ask("Nozzle height cm", 14.0, float)
    offset = ask("Camera to nozzle offset cm", 3.6, float)
    clearance = ask("Keep-off margin round plants cm", 2.0, float)
    print()

    plan_dir = SPRAYING / "plans" / name
    run([sys.executable, str(SPRAYING / "simulate.py"),
         f"../mapping/maps/{name}/{name}.json",
         "--bed", "0", "0", str(width), str(depth),
         "--side", "left",
         "--rail-length", str(rail_len),
         "--throw", str(throw),
         "--height", str(height),
         "--offset-across", str(offset),
         "--clearance", str(clearance),
         "--continuous", "--cell", "2",
         "--out", f"plans/{name}"], SPRAYING)

    approve("Does the route look right? This is the last check before "
            "commands are written.",
            [plan_dir / "spray_route.png", plan_dir / "spray_coverage.png"])

    # ---- 3. commands -----------------------------------------------------
    print("STEP 3  Write motor and valve commands\n")
    run([sys.executable, str(SPRAYING / "motors.py"),
         f"plans/{name}/plan.json", "--dc"], SPRAYING)
    run([sys.executable, str(SPRAYING / "motor_sim.py"),
         "dc_commands.csv", "--plan", f"plans/{name}/plan.json"], SPRAYING)

    print("\n" + "=" * 62)
    print("Done. The machine's instructions are in:")
    print(f"  spraying/dc_commands.csv")
    print("\nNothing has moved. Sending that file to the hardware is a")
    print("separate, deliberate step.")
    print("=" * 62)


if __name__ == "__main__":
    main()
