"""
Stage 4. Read the command file and drive the machine.

    python run_plan.py --dry-run          rehearse it, no hardware touched
    python run_plan.py                    for real

This is the only code in the project that moves anything. Everything before
it writes files.

HOW POSITION IS KNOWN, WITHOUT ENCODERS
    The motors have no feedback, so nothing can measure where an axis is.
    Two things are trusted instead:

      The rail has a hard stop. Driving into it for longer than the full
      travel leaves the carriage against it whatever happened before, so
      that position is exact and self-correcting. It is homed there at the
      start of every run, and again before every box.

      Pan and tilt have no stop, so a person sets them to a marked neutral
      and the run waits for confirmation. The code cannot find neutral by
      itself - it has no way to detect where it is. Within about 3 degrees
      the spray pattern absorbs the error; past 5 it starts missing the box
      it aimed at.

    Re-homing before each box is what stops error accumulating. Every move
    is measured from the stop, never from wherever the last move happened
    to finish, so a bad move spoils one box instead of every box after it.

WHY MOVES MUST NOT BE SHORT
    A DC motor with no speed control is either off or at full tilt, and it
    takes time to get there. A 0.06 second pulse is all acceleration and
    travels an unpredictable distance. Anything under MIN_MOVE_S is refused
    here rather than executed badly - if the plan is full of them, the plan
    needs coarser spacing, not a braver driver.

STOPPING
    Ctrl+C, or any error, shuts the valve FIRST and then the motors. The
    other order leaves air running onto one spot while the head sits still.
    A physical switch in the air line is better than either; this is the
    software backstop, not the safety case.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# Wiring. These are the numbers to check first when something moves the
# wrong way - they are facts about the build, not about the maths.
# ---------------------------------------------------------------------------

PINS = {
    #          forward pin, reverse pin
    "rail": (23, 24),       # Motor 1 - IN1 HIGH drives toward +x (right)
    "pan":  (27, 22),       # Motor 2 - IN3 HIGH rotates clockwise from above
    "tilt": (5, 6),         # Motor 3 - IN5 HIGH raises the nozzle (longer shot)
}
VALVE_PIN = 13

# Set True for any axis wired so that the "forward" pin drives it backwards.
INVERT = {"rail": False, "pan": False, "tilt": False}

# Measured, not guessed. Run --calibrate-rail to get the first one.
SPEED = {"rail": 4.0, "pan": 20.0, "tilt": 20.0}      # cm/s, deg/s, deg/s
SPRAY_PATCH_CM = 5.1         # 2 inches, measured off the airbrush
WORKING_RANGE_CM = 20.0      # typical nozzle-to-soil distance

RAIL_TRAVEL_CM = 35.0        # 350 mm, stop to stop
HOME_EXTRA = 1.4             # drive this much longer than a full traverse
MIN_MOVE_S = 0.15            # below this a move is all acceleration
SETTLE_S = 0.08              # let it stop before the next command

PAN_LIMIT = 90.0             # +/- from the neutral datum
TILT_LIMIT = 80.0


# ---------------------------------------------------------------------------

class Hardware:
    """The pins. Or, with --dry-run, a convincing impression of them."""

    def __init__(self, dry=False, log=None, real_time=False):
        self.dry = dry
        # A dry run skips the waiting by default, so a 15 minute plan can be
        # checked in a second. --real-time rehearses at the true pace, which
        # is what you want the day before, standing next to the machine.
        self.real_time = real_time or not dry
        self.log = log if log is not None else []
        self.gpio = None

        if dry:
            print("  DRY RUN - no pins are touched\n")
            return

        try:
            import RPi.GPIO as GPIO
        except ImportError:
            sys.exit("RPi.GPIO not available. Run with --dry-run on a laptop.")

        self.gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for a, b in PINS.values():
            GPIO.setup(a, GPIO.OUT)
            GPIO.setup(b, GPIO.OUT)
        GPIO.setup(VALVE_PIN, GPIO.OUT)
        self.all_off()

    def _write(self, pin, high):
        if self.dry:
            return
        self.gpio.output(pin, self.gpio.HIGH if high else self.gpio.LOW)

    def drive(self, axis, direction):
        """direction is +1, -1 or 0. Nothing else is accepted."""
        fwd, rev = PINS[axis]
        if INVERT[axis]:
            direction = -direction
        self._write(fwd, direction > 0)
        self._write(rev, direction < 0)

    def stop(self, axis):
        self.drive(axis, 0)

    def valve(self, on):
        self._write(VALVE_PIN, on)
        self.log.append(("valve", "open" if on else "shut"))

    def all_off(self):
        """Valve first, always. See the note at the top of this file."""
        self.valve(False)
        for axis in PINS:
            self.stop(axis)

    def cleanup(self):
        self.all_off()
        if not self.dry and self.gpio:
            self.gpio.cleanup()


# ---------------------------------------------------------------------------

def wait(hw, seconds):
    if seconds > 0 and getattr(hw, "real_time", True):
        time.sleep(seconds)


def move(hw, moves, settle=SETTLE_S):
    """
    Run several axes at once, each stopping at its own deadline.

    The H-bridges are independent, so the axes genuinely move together and
    the step takes as long as the slowest one, not the sum.
    """
    live = []
    now = time.monotonic()
    for axis, direction, seconds in moves:
        if direction == 0 or seconds <= 0:
            continue
        if seconds < MIN_MOVE_S:
            print(f"      {axis} {seconds:.3f}s - too short to drive, holding")
            continue
        hw.drive(axis, direction)
        live.append((axis, now + seconds))

    if not live:
        return 0.0

    for axis, deadline in sorted(live, key=lambda t: t[1]):
        wait(hw, deadline - time.monotonic())
        hw.stop(axis)

    wait(hw, settle)
    return max(d for _, d in live) - now + settle


def home_rail(hw):
    """
    Drive into the stop and stay there long enough to be certain.

    Overshoot is the point: whatever position it started from, it finishes
    pressed against the stop. That is the one position this machine can
    know exactly.
    """
    seconds = (RAIL_TRAVEL_CM / SPEED["rail"]) * HOME_EXTRA
    print(f"    homing rail  ({seconds:.1f}s against the stop)")
    hw.drive("rail", -1)
    wait(hw, seconds)
    hw.stop("rail")
    wait(hw, SETTLE_S * 3)


def ask_datum():
    print("\n" + "=" * 62)
    print("SET THE NEUTRAL DATUM")
    print("=" * 62)
    print("Pan and tilt have no stop, so they cannot find zero on their own.")
    print("\n  Set PAN  to its neutral mark  (straight across the bed)")
    print("  Set TILT to its neutral mark")
    print("\nWithin about 3 degrees is fine - the spray pattern covers that.")
    print("Past 5 and it starts missing the box it aimed at.")
    answer = input("\nBoth set? (yes / no): ").strip().lower()
    if answer not in ("y", "yes"):
        sys.exit("\nStopped. Nothing moved.")
    print()


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no commands in it.")

    need = {"rail_dir", "pan_dir", "tilt_dir",
            "rail_secs", "pan_secs", "tilt_secs", "spray"}
    missing = need - set(rows[0])
    if missing:
        sys.exit(f"{path} is missing columns: {sorted(missing)}\n"
                 f"Produce it with:  python motors.py <plan.json> --dc")
    return rows


def skipped_cost():
    """
    How far off a skipped move leaves the aim, at worst.

    A move under MIN_MOVE_S is not executed, so the aim just stays where it
    was. That only matters if the gap is large against the spray pattern -
    and the longest move that can be skipped is MIN_MOVE_S at full speed,
    so the worst case is a fixed number rather than a guess.
    """
    straight = SPEED["rail"] * MIN_MOVE_S
    turn = SPEED["pan"] * MIN_MOVE_S
    swung = WORKING_RANGE_CM * math.tan(math.radians(turn))
    return max(straight, swung)


def summarise(rows):
    short = sum(1 for r in rows
                if 0 < max(float(r["rail_secs"]), float(r["pan_secs"]),
                           float(r["tilt_secs"])) < MIN_MOVE_S)
    total = sum(max(float(r["rail_secs"]), float(r["pan_secs"]),
                    float(r["tilt_secs"])) + SETTLE_S for r in rows)
    # The valve starts shut, so the first spraying row is itself a change.
    # Counting only row-to-row transitions reported 0 changes for a plan
    # that sprays from the first row to the last, which reads as "nothing
    # happens" when in fact the valve opens once and stays open.
    on_rows = sum(1 for r in rows if str(r["spray"]).strip() == "1")
    state = "0"
    changes = 0
    for r in rows:
        s = str(r["spray"]).strip()
        if s != state:
            changes += 1
            state = s
    boxes = len({r.get("box") for r in rows})
    homing = RAIL_TRAVEL_CM / SPEED["rail"] * HOME_EXTRA

    print(f"  Rows            : {len(rows):,}")
    print(f"  Boxes           : {boxes}")
    print(f"  Spraying on     : {on_rows} of {len(rows)} rows")
    print(f"  Valve changes   : {changes}")
    print(f"  Driving         : {total:.0f} s")
    print(f"  Homing          : {boxes} x {homing:.0f} s = {boxes * homing:.0f} s")
    print(f"  Rough total     : {(total + boxes * homing) / 60:.1f} min")

    # A plan where the valve never opens will drive every motor, take the
    # full run time and lay no mulch at all. That is the quietest way this
    # can waste a session, so it is called out before anything moves
    # rather than discovered afterwards.
    if on_rows == 0:
        print("")
        print("  NOTHING WILL BE SPRAYED - no row in this plan opens the")
        print("  valve. The usual cause is that every box picked was out of")
        print("  the sprayer's reach, which leaves only the travel moves.")
        print("  Re-plan the spraying pathway and check its reach report.")

    if short:
        worst = skipped_cost()
        print("")
        print(f"  {short} move(s) are under {MIN_MOVE_S}s, so they are not run -")
        print(f"  the aim holds its last position instead. That leaves it off")
        print(f"  by at most {worst:.1f} cm, against a {SPRAY_PATCH_CM:.1f} cm spray patch.")
        if worst < SPRAY_PATCH_CM / 2:
            print(f"  The soil still gets covered, so this is fine as it is.")
        else:
            print(f"  That is too much of the patch. Either slow the motors or")
            print(f"  space the aim points further apart.")
    return short


def run(hw, rows, rehome_every_box=True):
    valve_is_open = False
    last_box = None
    t0 = time.monotonic()

    for i, r in enumerate(rows):
        box = r.get("box", r.get("i", ""))

        if rehome_every_box and box != last_box:
            if valve_is_open:
                hw.valve(False)
                valve_is_open = False
            print(f"\n  Box {box}")
            home_rail(hw)
            last_box = box

        want_valve = r["spray"] == "1"

        # The valve closes BEFORE a move that is not meant to be spraying,
        # and opens only once the head is already where it should be.
        if valve_is_open and not want_valve:
            hw.valve(False)
            valve_is_open = False

        took = move(hw, [
            ("rail", int(r["rail_dir"]), float(r["rail_secs"])),
            ("pan",  int(r["pan_dir"]),  float(r["pan_secs"])),
            ("tilt", int(r["tilt_dir"]), float(r["tilt_secs"])),
        ])

        if want_valve and not valve_is_open:
            hw.valve(True)
            valve_is_open = True

        if i % 10 == 0 or took > 0.5:
            print(f"    {i:4d}/{len(rows)}  "
                  f"rail {r.get('rail_cm','?'):>6}  "
                  f"pan {r.get('pan_deg','?'):>7}  "
                  f"tilt {r.get('tilt_deg','?'):>6}  "
                  f"valve {'OPEN' if valve_is_open else 'shut'}")

    if valve_is_open:
        hw.valve(False)
    print(f"\n  Finished in {time.monotonic() - t0:.0f} s")


def calibrate_rail(hw):
    """Time a full traverse. This is the only speed number that is measured."""
    print("\nRail calibration")
    print("  It will home, then drive the full length. Time how long the")
    print("  carriage is actually moving, and stop it with Ctrl+C when it")
    print("  reaches the far stop.\n")
    input("  Enter to start...")

    home_rail(hw)
    print(f"\n  Driving out. Watch it. Ctrl+C when it reaches the far end.")
    start = time.monotonic()
    hw.drive("rail", +1)
    try:
        time.sleep(RAIL_TRAVEL_CM / SPEED["rail"] * 2)
    except KeyboardInterrupt:
        pass
    took = time.monotonic() - start
    hw.stop("rail")

    print(f"\n  {RAIL_TRAVEL_CM:.0f} cm in {took:.2f} s")
    print(f"  -> SPEED['rail'] = {RAIL_TRAVEL_CM / took:.2f}   (cm per second)")
    print(f"\n  Put that at the top of this file.")


def main():
    ap = argparse.ArgumentParser(description="Drive the machine from a plan.")
    ap.add_argument("commands", nargs="?",
                    default="../spraying/dc_commands.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearse without touching any pin")
    ap.add_argument("--real-time", action="store_true",
                    help="with --dry-run, wait the real durations instead "
                         "of running through instantly")
    ap.add_argument("--no-rehome", action="store_true",
                    help="do not home before each box. Faster, and error "
                         "then accumulates across the whole run")
    ap.add_argument("--calibrate-rail", action="store_true",
                    help="time a full traverse to measure the rail speed")
    args = ap.parse_args()

    hw = Hardware(dry=args.dry_run, real_time=args.real_time)

    try:
        if args.calibrate_rail:
            calibrate_rail(hw)
            return

        path = pathlib.Path(args.commands)
        if not path.exists():
            sys.exit(f"No command file at {path}")

        rows = load_rows(path)

        print("=" * 62)
        print("RUN PLAN" + ("   (dry run)" if args.dry_run else ""))
        print("=" * 62)
        print(f"  File            : {path}")
        summarise(rows)

        if not args.dry_run:
            ask_datum()
            if input("Start the run? (yes / no): ").strip().lower() \
                    not in ("y", "yes"):
                sys.exit("\nStopped. Nothing moved.")

        run(hw, rows, rehome_every_box=not args.no_rehome)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 62)
        print("STOPPED - valve shut first, then motors")
        print("=" * 62)
    finally:
        hw.cleanup()
        print("\nAll pins low. Valve shut.")


if __name__ == "__main__":
    main()
