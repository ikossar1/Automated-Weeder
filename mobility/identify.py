"""
Find out which motor drives which axis, by driving each one and asking.

Every number in this project rests on knowing that pins 23/24 move THIS
and the stepper moves THAT. Get it wrong and the machine still runs -
it just drives the wrong motor for every command, which looks like bad
calibration rather than a wiring map that does not match reality.

It has been guessed twice from the jog script's key names. This settles
it by moving one motor at a time and asking what moved, then writing the
answer to axis_map.json, which run_plan loads in preference to its own
defaults.

    python3 identify.py              drive each motor and ask
    python3 identify.py --show       print the saved map
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_plan as RP

MAP_FILE = HERE / "axis_map.json"

AXES = {
    "1": ("rail", "slides along the bed, in a straight line"),
    "2": ("pan", "swings the nozzle left and right"),
    "3": ("tilt", "tips the nozzle up and down"),
}


def ask_what_moved(label):
    print(f"\n  Watch the machine. Driving {label} now.")
    print("  What moved?")
    for key, (name, what) in AXES.items():
        print(f"    {key}  {name:5s} - {what}")
    print("    0  nothing moved")
    while True:
        try:
            said = input("  > ").strip()
        except EOFError:
            return None
        if said == "0":
            return None
        if said in AXES:
            return AXES[said][0]
        print("  Type 1, 2, 3 or 0.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Work out the axis wiring.")
    ap.add_argument("--show", action="store_true", help="print the saved map")
    ap.add_argument("--seconds", type=float, default=1.5,
                    help="how long to drive each motor (default 1.5)")
    ap.add_argument("--duty", type=float, default=40.0,
                    help="power to drive at, percent (default 40)")
    args = ap.parse_args(argv)

    if args.show:
        if MAP_FILE.exists():
            print(json.dumps(json.loads(MAP_FILE.read_text()), indent=2))
        else:
            print(f"No map saved yet. Run: python3 {pathlib.Path(__file__).name}")
        return 0

    print("=" * 62)
    print("WHICH MOTOR IS WHICH")
    print("=" * 62)
    print("Each motor runs briefly. Say what you saw move.")
    print("Stand clear, and keep a hand near the power.\n")
    try:
        input("Enter when ready, Ctrl+C to stop: ")
    except EOFError:
        return 1

    hw = RP.Hardware(dry=False)
    found = {}
    try:
        # the two DC motors, by pin pair
        for pins, label in ((RP.PINS["rail"], "the motor on pins 23 and 24"),
                            (RP.PINS["pan"], "the motor on pins 27 and 22")):
            hw._motor(pins[0], args.duty)
            RP.wait(hw, args.seconds)
            hw._motor(pins[0], 0.0)
            RP.wait(hw, 0.4)
            name = ask_what_moved(label)
            if name:
                found[name] = {"kind": "dc", "pins": list(pins)}

        # the stepper
        hw.tilt_to(hw.tilt_deg + 40.0)
        RP.wait(hw, 0.4)
        name = ask_what_moved("the stepper on pins 25 and 6")
        if name:
            found[name] = {"kind": "stepper",
                           "step_pin": RP.TILT_STEP_PIN,
                           "dir_pin": RP.TILT_DIR_PIN}
    finally:
        hw.cleanup()

    print("\n" + "=" * 62)
    if not found:
        print("Nothing was identified. Check power and wiring.")
        return 1

    for axis in ("rail", "pan", "tilt"):
        got = found.get(axis)
        print(f"  {axis:5s} : {got if got else 'NOT FOUND'}")

    missing = [a for a in ("rail", "pan", "tilt") if a not in found]
    if missing:
        print(f"\n  Still unaccounted for: {', '.join(missing)}")
        print("  Saving what was found anyway.")

    MAP_FILE.write_text(json.dumps(found, indent=2))
    print(f"\n  Written to {MAP_FILE.name}. run_plan will use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
