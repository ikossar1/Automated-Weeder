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
    "rail": (23, 24),       # Motor 1 - 23 HIGH drives toward +x (right)
    "pan":  (27, 22),       # Motor 2 - 27 HIGH rotates clockwise from above
}

# Tilt is a stepper on an EasyDriver, not a DC motor.
#
# This is the one axis that knows where it is. A DC motor is told to run
# for a length of time and we hope; a stepper is told to take a number of
# steps and it takes them. So tilt alone is immune to the speed guesses
# that make the other two axes approximate, and none of the duration
# machinery below - MIN_MOVE_S, MAX_TURN_S - applies to it.
TILT_STEP_PIN = 25
TILT_DIR_PIN = 6
TILT_STEP_DELAY = 0.002      # seconds per half pulse, from the jog script

# MEASURE THIS. The default assumes a 200 step motor driven at the
# EasyDriver's 1/8 microstep default, turning the nozzle directly:
#     200 * 8 = 1600 steps per turn, / 360 = 4.44 per degree
# Any gearing between motor and nozzle multiplies it. To check: run
# --tilt-sweep, count the degrees it actually moved, and divide.
TILT_STEPS_PER_DEG = 4.44
TILT_DIR_RAISES = True       # True if DIR high tilts the nozzle UP

# The valve sits on two pins through an H-bridge channel, not one pin.
#
# Driving 26 high with 5 low energises it; both low releases it. That is
# a spring-return valve: it shuts by itself when power is removed, which
# is the behaviour this code relies on to fail safe.
#
# If yours is a latching valve - one that stays put after a pulse - this
# is WRONG and it will never close. Check by opening it and cutting
# power: a spring-return shuts, a latching one does not.
VALVE_PINS = (26, 5)

# Set True for any axis wired so that the "forward" pin drives it backwards.
INVERT = {"rail": False, "pan": False, "tilt": False}

# Measured, not guessed. Run --calibrate-rail to get the first one.
SPEED = {"rail": 4.0, "pan": 20.0, "tilt": 20.0}      # cm/s, deg/s, deg/s
SPRAY_PATCH_CM = 5.1         # 2 inches, measured off the airbrush
WORKING_RANGE_CM = 20.0      # typical nozzle-to-soil distance

RAIL_TRAVEL_CM = 35.0        # 350 mm, stop to stop
HOME_EXTRA = 1.4             # drive this much longer than a full traverse
MIN_MOVE_S = 0.15            # below this a move is all acceleration

# No single pan or tilt command may drive longer than this.
#
# The plan works out its durations from SPEED below. If the real motor is
# faster than SPEED says, every turn overshoots, and a bad SPEED can spin
# an axis right round rather than nudging it a few degrees. This cap is
# the backstop: it bounds the damage a wrong SPEED can do to one second
# of travel per command, whatever the plan asked for.
#
# It is a safety net, not a fix. If moves are being capped, SPEED is wrong
# and wants measuring.
MAX_TURN_S = 1.0
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
        # Where the stepper believes it is, in degrees from the datum.
        self.tilt_deg = 0.0

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

        # Release anything a previous run left claimed. A script that was
        # killed rather than closed keeps its pins, and the next run then
        # cannot have them.
        try:
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BCM)

        # initial= matters more than it looks. Without it the Pi 5 GPIO
        # layer reads the pin to find its current level, and that read
        # fails outright if the line is not already allocated. Saying what
        # the level should be skips the read - and starting every pin low
        # is what we want anyway, since low is motors stopped, valve shut.
        pins = ([n for pair in PINS.values() for n in pair]
                + list(VALVE_PINS) + [TILT_STEP_PIN, TILT_DIR_PIN])
        for pin in pins:
            try:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            except Exception as exc:
                sys.exit(
                    f"Could not claim GPIO {pin}: {exc}" + NL + NL
                    + "Something else is holding the pins. Usually that is" + NL
                    + "another script still running - check with:" + NL + NL
                    + "    ps aux | grep -i python" + NL + NL
                    + "and stop it, or reboot the Pi if you are not sure.")
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
        energise, other = VALVE_PINS
        self._write(other, False)
        self._write(energise, on)
        self.log.append(("valve", "open" if on else "shut"))

    def tilt_to(self, degrees):
        """
        Step the nozzle to an absolute angle.

        Absolute, not relative, because a stepper can hold a true position
        and there is no reason to accumulate error on the one axis that
        does not have to. The datum set at the start of the run is what
        this counts from.
        """
        delta = degrees - self.tilt_deg
        steps = int(round(delta * TILT_STEPS_PER_DEG))
        if steps == 0:
            return 0.0

        up = (steps > 0) == TILT_DIR_RAISES
        self._write(TILT_DIR_PIN, up)
        took = abs(steps) * TILT_STEP_DELAY * 2

        if not self.dry:
            time.sleep(0.001)
            for _ in range(abs(steps)):
                self._write(TILT_STEP_PIN, True)
                time.sleep(TILT_STEP_DELAY)
                self._write(TILT_STEP_PIN, False)
                time.sleep(TILT_STEP_DELAY)

        # Only the rounding is lost, and it is lost once rather than
        # compounding, because the next move is worked out from where we
        # actually are.
        self.tilt_deg += steps / TILT_STEPS_PER_DEG
        self.log.append(("tilt", round(self.tilt_deg, 2)))
        return took

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


def move(hw, moves, settle=SETTLE_S, max_turn=MAX_TURN_S):
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
        # The rail runs against a hard stop and is allowed its long homing
        # drive. Pan has no stop and no feedback, so it is capped. Tilt
        # does not come through here at all any more - it is a stepper.
        if axis == "pan" and seconds > max_turn:
            print(f"      {axis} {seconds:.2f}s capped to {max_turn:.2f}s "
                  f"- check SPEED['{axis}']")
            seconds = max_turn
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
    print("Tilt is a stepper and will hold whatever you set here all run;")
    print("pan is a plain motor and drifts, so it is the one to watch.")
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
    # Tilt no longer takes the time the plan guessed for a DC motor: it
    # takes however long its pulses take, which is a known number.
    tilt_at = 0.0
    total = 0.0
    for r in rows:
        want = float(r["tilt_deg"])
        steps = abs(int(round((want - tilt_at) * TILT_STEPS_PER_DEG)))
        tilt_at += steps / TILT_STEPS_PER_DEG if want >= tilt_at else -steps / TILT_STEPS_PER_DEG
        total += (steps * TILT_STEP_DELAY * 2
                  + max(float(r["rail_secs"]), float(r["pan_secs"]))
                  + SETTLE_S)
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


def list_boxes(rows):
    """Boxes in the order the plan visits them, with their row counts."""
    order, counts = [], {}
    for r in rows:
        b = r.get("box", "")
        if b not in counts:
            order.append(b)
            counts[b] = [0, 0]
        counts[b][0] += 1
        counts[b][1] += 1 if str(r["spray"]).strip() == "1" else 0
    return [(b, counts[b][0], counts[b][1]) for b in order]


def choose_boxes(rows):
    """
    Ask which boxes to run, and keep only those rows.

    The plan holds every box the map flagged. Running the lot is right for
    a finished bed, but while testing you usually want one box, watched
    closely, before letting it loose on the rest.
    """
    boxes = list_boxes(rows)
    print("")
    print("  Boxes in this plan:")
    for i, (name, n, on) in enumerate(boxes):
        print(f"    [{i}]  box {name:<4} {n:4d} moves, sprays on {on}")
    print("")
    print("  Which ones? Numbers from the brackets, separated by spaces.")
    print("  Blank runs all of them, q quits.")
    try:
        answer = input("  > ").strip()
    except EOFError:
        answer = ""

    if answer.lower() in ("q", "quit", "n", "no"):
        return None
    if not answer:
        return rows

    wanted = set()
    for bit in answer.replace(",", " ").split():
        try:
            idx = int(bit)
        except ValueError:
            print(f"  '{bit}' is not a number, ignoring it")
            continue
        if 0 <= idx < len(boxes):
            wanted.add(boxes[idx][0])
        else:
            print(f"  there is no box [{idx}], ignoring it")

    if not wanted:
        print("  Nothing selected.")
        return None

    kept = [r for r in rows if r.get("box", "") in wanted]
    print(f"  Running {len(wanted)} box(es), {len(kept)} moves.")
    return kept


def run(hw, rows, rehome_every_box=True, max_turn=MAX_TURN_S, confirm=False):
    valve_is_open = False
    last_box = None
    skipping = None
    t0 = time.monotonic()

    for i, r in enumerate(rows):
        box = r.get("box", r.get("i", ""))

        if rehome_every_box and box != last_box:
            if valve_is_open:
                hw.valve(False)
                valve_is_open = False
            print(f"\n  Box {box}")
            if confirm:
                try:
                    said = input("    Enter to run this box, s to skip, "
                                 "q to stop: ").strip().lower()
                except EOFError:
                    said = ""
                if said == "q":
                    print("    Stopping here.")
                    break
                if said == "s":
                    skipping = box
                    last_box = box
                    continue
            home_rail(hw)
            last_box = box

        want_valve = r["spray"] == "1"

        # The valve closes BEFORE a move that is not meant to be spraying,
        # and opens only once the head is already where it should be.
        if valve_is_open and not want_valve:
            hw.valve(False)
            valve_is_open = False

        if skipping is not None and box == skipping:
            continue
        skipping = None

        # Tilt first, and on its own. Stepping is a tight pulse loop that
        # cannot share the clock with a timed DC move, and the stepper is
        # exact, so there is nothing to gain from overlapping them.
        took = hw.tilt_to(float(r["tilt_deg"]))
        took += move(hw, [
            ("rail", int(r["rail_dir"]), float(r["rail_secs"])),
            ("pan",  int(r["pan_dir"]),  float(r["pan_secs"])),
        ], max_turn=max_turn)

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


def tilt_sweep(hw, degrees=90.0):
    """
    Drive the tilt a stated angle so the real steps-per-degree can be
    measured. Mark where the nozzle points, run it, measure where it
    ended up, then set TILT_STEPS_PER_DEG to:

        current value  *  (degrees asked for / degrees actually moved)
    """
    print(f"  Tilt sweep: asking for {degrees:.0f} degrees at "
          f"{TILT_STEPS_PER_DEG:.2f} steps/deg")
    print(f"  = {int(round(degrees * TILT_STEPS_PER_DEG))} steps")
    print("  Mark where the nozzle points now, then press Enter.")
    try:
        input("  > ")
    except EOFError:
        pass
    hw.tilt_to(hw.tilt_deg + degrees)
    print("")
    print("  Measure the angle it actually moved through, then set")
    print("  TILT_STEPS_PER_DEG near the top of this file to:")
    print("")
    print(f"      {TILT_STEPS_PER_DEG:.2f} * ({degrees:.0f} / measured)")
    print("")
    print("  Until that is done every tilt angle in the plan is a guess.")


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
    ap.add_argument("--pick", action="store_true",
                    help="choose which boxes to run, and confirm each one "
                         "before it moves")
    ap.add_argument("--max-turn", type=float, default=MAX_TURN_S,
                    help=f"longest single pan or tilt drive, seconds "
                         f"(default {MAX_TURN_S})")
    ap.add_argument("--tilt-sweep", type=float, nargs="?", const=90.0,
                    default=None, metavar="DEG",
                    help="drive the tilt this many degrees to measure "
                         "TILT_STEPS_PER_DEG (default 90)")
    ap.add_argument("--calibrate-rail", action="store_true",
                    help="time a full traverse to measure the rail speed")
    args = ap.parse_args()

    hw = Hardware(dry=args.dry_run, real_time=args.real_time)

    try:
        if args.tilt_sweep is not None:
            tilt_sweep(hw, args.tilt_sweep)
            return

        if args.calibrate_rail:
            calibrate_rail(hw)
            return

        path = pathlib.Path(args.commands)
        if not path.exists():
            sys.exit(f"No command file at {path}")

        rows = load_rows(path)

        if args.pick:
            rows = choose_boxes(rows)
            if not rows:
                print("  Nothing to do.")
                return

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

        run(hw, rows, rehome_every_box=not args.no_rehome,

            max_turn=args.max_turn, confirm=args.pick)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 62)
        print("STOPPED - valve shut first, then motors")
        print("=" * 62)
    finally:
        hw.cleanup()
        print("\nAll pins low. Valve shut.")


if __name__ == "__main__":
    main()
