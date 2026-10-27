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
import io
import math
import pathlib
import signal
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
# Seconds per half pulse. With 4.44 steps/deg this sets the tilt rate:
#     1 / (4.44 * 2 * 0.000483) = 233 deg/s
# which is four times the jog script's 56.3 deg/s.
#
# That is 1036 full steps a second, and at that rate time.sleep stops
# being usable - it cannot resolve half a millisecond and its own
# overhead is the same size as the wait. _hold below busy-waits instead,
# which costs a pegged CPU core for the length of a move and buys a step
# rate that is actually what it says.
#
# Steppers also stall if asked to start straight into a high rate, and
# there is no acceleration ramp here. If tilt buzzes without turning,
# that is this number being too small, not the wiring.
TILT_STEP_DELAY = 0.000483

# MEASURE THIS. The default assumes a 200 step motor driven at the
# EasyDriver's 1/8 microstep default, turning the nozzle directly:
#     200 * 8 = 1600 steps per turn, / 360 = 4.44 per degree
# Any gearing between motor and nozzle multiplies it. To check: run
# --tilt-sweep, count the degrees it actually moved, and divide.
TILT_STEPS_PER_DEG = 4.44
TILT_DIR_RAISES = True       # True if DIR high tilts the nozzle UP

# The valve LATCHES. It is not a solenoid you hold open.
#
# One pulse opens it and it stays open with no power. Another pulse the
# other way shuts it. Both pins low does not mean shut - it means stay as
# you are - so the valve can only be closed by actively driving it, never
# by releasing it.
#
# Everything about how this file handles the valve follows from that:
#
#   - we track whether it is open, because the hardware will not tell us
#   - we pulse it shut at startup, because a run that was killed or lost
#     power may have left it open and there is no way to read which
#   - an abort pulses it shut and waits for the pulse to finish, rather
#     than dropping the pins and exiting
#
# A spring-return valve would make all of that unnecessary. This one does
# not forgive a missed command, so nothing here assumes one.
VALVE_OPEN_PIN = 26
VALVE_SHUT_PIN = 5
VALVE_PINS = (VALVE_OPEN_PIN, VALVE_SHUT_PIN)
VALVE_PULSE_S = 1.0          # how long to drive it to flip, from the jog script

# Set True for any axis wired so that the "forward" pin drives it backwards.
INVERT = {"rail": False, "pan": False, "tilt": False}

# Measured, not guessed. Run --calibrate-rail to get the first one.
# How fast each DC axis actually turns, measured with a stopwatch.
SPEED_DEG = {"rail": 50.0, "pan": 42.86}

# rail is cm/s, used only for homing and for turning plan distances into
# durations. tilt is absent on purpose: it is a stepper, and its rate
# comes from TILT_STEP_DELAY.
SPEED = {"rail": 4.0, "pan": SPEED_DEG["pan"]}
SPRAY_PATCH_CM = 5.1         # 2 inches, measured off the airbrush
WORKING_RANGE_CM = 20.0      # typical nozzle-to-soil distance

RAIL_TRAVEL_CM = 35.0        # 350 mm, stop to stop
HOME_EXTRA = 1.4             # drive this much longer than a full traverse
MIN_MOVE_S = 0.15            # below this a move is all acceleration

# Measured on the machine, not taken from the rating. Both DC motors are
# rated 10 RPM - 60 deg/s - and neither runs at it:
#
#     up/down     70 deg in 1.4 s  =  50.0 deg/s
#     left/right 150 deg in 3.5 s  =  42.9 deg/s
#
# The caps are written as the furthest one command may take an axis, in
# degrees, and converted to seconds here. That way a re-measured speed
# updates the cap by itself, instead of leaving a stale number in
# seconds that no longer means the angle it was chosen to mean.
#
# GRACE is slack for the measurements being eyeballed. A cap that sits
# exactly on the measured figure will clip honest moves whenever the
# real speed is a little under what was timed.
MAX_DRIVE_DEG = {"rail": 100.0, "pan": 40.0}
GRACE = 1.2

MAX_DRIVE_S = {a: MAX_DRIVE_DEG[a] / SPEED_DEG[a] * GRACE
               for a in MAX_DRIVE_DEG}
MAX_TURN_S = MAX_DRIVE_S["pan"]      # what --max-turn starts from
SETTLE_S = 0.08              # let it stop before the next command

PAN_LIMIT = 90.0             # +/- from the neutral datum
TILT_LIMIT = 80.0


# ---------------------------------------------------------------------------

def _hold(seconds):
    """
    Wait a very short time, accurately.

    time.sleep is fine down to a millisecond or so and useless below it:
    it rounds up, and the cost of calling it is about the same as the
    wait being asked for. The stepper needs half-millisecond pulses, so
    below a millisecond this spins on the clock instead. It burns a core
    while it runs, which is an acceptable price for a few hundred
    milliseconds of stepping and would not be for anything longer.
    """
    # The threshold is 5 ms, not 1. time.sleep was measured running 35%
    # long at 2 ms, so anything near a millisecond is spun rather than
    # slept - the old 2 ms pulse was quietly a 2.7 ms one.
    if seconds >= 0.005:
        time.sleep(seconds)
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


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
        # True until proven otherwise, so the first all_off really pulses
        # it shut instead of deciding there is nothing to do.
        self.valve_open = True

        if dry:
            print("  DRY RUN - no pins are touched\n")
            # Establish the same known state a real run starts from,
            # or the valve would still be assumed open and every
            # open command skipped as a no-op.
            self.all_off()
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

    def valve(self, on, force=False):
        """
        Flip the latching valve, and only if it is not already there.

        Pulsing costs a second, so a no-op is worth skipping - but `force`
        pulses anyway, which is how startup establishes a known state from
        an unknown one.
        """
        if self.valve_open == on and not force:
            return 0.0

        drive = VALVE_OPEN_PIN if on else VALVE_SHUT_PIN
        other = VALVE_SHUT_PIN if on else VALVE_OPEN_PIN
        self._write(other, False)
        self._write(drive, True)
        if not self.dry:
            time.sleep(VALVE_PULSE_S)
        self._write(drive, False)

        self.valve_open = on
        self.log.append(("valve", "open" if on else "shut"))
        return VALVE_PULSE_S

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
                _hold(TILT_STEP_DELAY)
                self._write(TILT_STEP_PIN, False)
                _hold(TILT_STEP_DELAY)

        # Only the rounding is lost, and it is lost once rather than
        # compounding, because the next move is worked out from where we
        # actually are.
        self.tilt_deg += steps / TILT_STEPS_PER_DEG
        self.log.append(("tilt", round(self.tilt_deg, 2)))
        return took

    def all_off(self):
        """
        Valve first, always. See the note at the top of this file.

        The pulse is forced because a latching valve cannot be read: if a
        previous run died with it open, our idea of its state is wrong,
        and the only safe move is to drive it shut regardless.
        """
        self.valve(False, force=True)
        for axis in PINS:
            self.stop(axis)
        self._write(TILT_STEP_PIN, False)

    def cleanup(self):
        """
        Shut down, and do not let a second Ctrl+C break it.

        Closing a latching valve takes a full second of driven pulse. A
        Ctrl+C landing inside that second would abandon the pulse half
        way and leave the valve open - the exact state the abort exists
        to prevent. So interrupts are held off until it is shut, which is
        a moment of being deliberately unresponsive in exchange for never
        walking away from an open valve.
        """
        try:
            old = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, AttributeError):
            old = None                      # not on the main thread
        try:
            self.all_off()
        finally:
            if old is not None:
                signal.signal(signal.SIGINT, old)
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
        cap = max_turn if axis == "pan" else MAX_DRIVE_S.get(axis)
        if cap is not None and seconds > cap:
            print(f"      {axis} {seconds:.2f}s capped to {cap:.2f}s "
                  f"- check SPEED['{axis}']")
            seconds = cap
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


def load_rows(path=None):
    """
    The plan, from a file if one is named and from inside this file if not.

    Carrying the plan inside the script is what makes this version a
    single file: there is no second file to copy, no folder to be in and
    nothing to go stale against it. Pass a .csv path to override it once
    a newer plan arrives.
    """
    if path is None:
        rows = list(csv.DictReader(io.StringIO(EMBEDDED_PLAN.strip())))
        path = "(built in)"
    else:
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
    ap.add_argument("commands", nargs="?", default=None,
                    help="a .csv plan to use instead of the built-in one")
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

        if args.commands:
            path = pathlib.Path(args.commands)
            if not path.exists():
                sys.exit(f"No command file at {path}")
            rows = load_rows(path)
        else:
            path = "(built in)"
            rows = load_rows()

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
        print("STOPPING - pulsing the valve shut, then the motors")
        print("(the valve latches, so this takes a second - let it finish)")
        print("=" * 62)
    finally:
        hw.cleanup()
        print("\nAll pins low. Valve shut.")


# --------------------------------------------------------------------
# The plan. Rewrite this block to change where the machine goes - or pass
# a .csv on the command line to use a different one without touching it.
# --------------------------------------------------------------------
EMBEDDED_PLAN = """\ni,box,rail_cm,pan_deg,tilt_deg,rail_counts,pan_counts,tilt_counts,rail_dir,pan_dir,tilt_dir,rail_secs,pan_secs,tilt_secs,spray
0,0,5.2,-14.2,32.6,624,-170,391,0,0,0,0.0,0.0,0.0,1
1,0,6.8,-14.2,32.6,816,-170,391,1,0,0,0.2,0.0,0.0,1
2,0,6.8,-12.9,30.2,816,-155,362,0,1,-1,0.0,0.0289,0.0533,1
3,0,6.8,-11.8,28.1,816,-142,337,0,1,-1,0.0,0.0244,0.0467,1
4,0,6.8,-12.9,30.2,816,-155,362,0,-1,1,0.0,0.0244,0.0467,1
5,0,6.8,-14.2,32.6,816,-170,391,0,-1,1,0.0,0.0289,0.0533,1
6,0,6.8,-15.8,35.4,816,-190,425,0,-1,1,0.0,0.0356,0.0622,1
7,0,6.8,-17.7,38.5,816,-212,462,0,-1,1,0.0,0.0422,0.0689,1
8,0,6.8,-20.3,42.1,816,-244,505,0,-1,1,0.0,0.0578,0.08,1
9,0,6.8,-23.6,46.2,816,-283,554,0,-1,1,0.0,0.0733,0.0911,1
10,0,8.2,-23.6,46.2,984,-283,554,1,0,0,0.175,0.0,0.0,1
11,0,8.2,-20.3,42.1,984,-244,505,0,1,-1,0.0,0.0733,0.0911,1
12,0,8.2,-17.7,38.5,984,-212,462,0,1,-1,0.0,0.0578,0.08,1
13,0,8.2,-15.8,35.4,984,-190,425,0,1,-1,0.0,0.0422,0.0689,1
14,0,8.2,-14.2,32.6,984,-170,391,0,1,-1,0.0,0.0356,0.0622,1
15,0,8.2,-15.8,35.4,984,-190,425,0,-1,1,0.0,0.0356,0.0622,1
16,0,9.8,-15.8,35.4,1176,-190,425,1,0,0,0.2,0.0,0.0,1
17,0,9.8,-17.7,38.5,1176,-212,462,0,-1,1,0.0,0.0422,0.0689,1
18,0,9.8,-20.3,42.1,1176,-244,505,0,-1,1,0.0,0.0578,0.08,1
19,0,9.8,-23.6,46.2,1176,-283,554,0,-1,1,0.0,0.0733,0.0911,1
20,0,9.8,-28.1,50.9,1176,-337,611,0,-1,1,0.0,0.1,0.1044,1
21,0,9.8,-34.4,55.9,1176,-413,671,0,-1,1,0.0,0.14,0.1111,1
22,0,9.8,-43.8,61.1,1176,-526,733,0,-1,1,0.0,0.2089,0.1156,1
23,0,9.8,-58.0,65.7,1176,-696,788,0,-1,1,0.0,0.3156,0.1022,1
24,0,11.2,-58.0,65.7,1344,-696,788,1,0,0,0.175,0.0,0.0,1
25,0,11.2,-43.8,61.1,1344,-526,733,0,1,-1,0.0,0.3156,0.1022,1
26,0,11.2,-34.4,55.9,1344,-413,671,0,1,-1,0.0,0.2089,0.1156,1
27,0,11.2,-43.8,61.1,1344,-526,733,0,-1,1,0.0,0.2089,0.1156,1
28,0,11.2,-58.0,65.7,1344,-696,788,0,-1,1,0.0,0.3156,0.1022,1
29,0,12.8,-58.0,65.7,1536,-696,788,1,0,0,0.2,0.0,0.0,1
30,0,11.2,-58.0,65.7,1344,-696,788,-1,0,0,0.2,0.0,0.0,1
31,0,9.8,-58.0,65.7,1176,-696,788,-1,0,0,0.175,0.0,0.0,1
32,0,8.2,-58.0,65.7,984,-696,788,-1,0,0,0.2,0.0,0.0,1
33,0,8.2,-43.8,61.1,984,-526,733,0,1,-1,0.0,0.3156,0.1022,1
34,0,8.2,-34.4,55.9,984,-413,671,0,1,-1,0.0,0.2089,0.1156,1
35,0,8.2,-28.1,50.9,984,-337,611,0,1,-1,0.0,0.14,0.1111,1
36,0,8.2,-34.4,55.9,984,-413,671,0,-1,1,0.0,0.14,0.1111,1
37,0,8.2,-43.8,61.1,984,-526,733,0,-1,1,0.0,0.2089,0.1156,1
38,0,6.8,-43.8,61.1,816,-526,733,-1,0,0,0.175,0.0,0.0,1
39,0,6.8,-58.0,65.7,816,-696,788,0,-1,1,0.0,0.3156,0.1022,1
40,0,6.8,-43.8,61.1,816,-526,733,0,1,-1,0.0,0.3156,0.1022,1
41,0,8.2,-43.8,61.1,984,-526,733,1,0,0,0.175,0.0,0.0,1
42,0,8.2,-58.0,65.7,984,-696,788,0,-1,1,0.0,0.3156,0.1022,1
43,0,9.8,-58.0,65.7,1176,-696,788,1,0,0,0.2,0.0,0.0,1
44,0,9.8,-43.8,61.1,1176,-526,733,0,1,-1,0.0,0.3156,0.1022,1
45,0,9.8,-34.4,55.9,1176,-413,671,0,1,-1,0.0,0.2089,0.1156,1
46,0,9.8,-28.1,50.9,1176,-337,611,0,1,-1,0.0,0.14,0.1111,1
47,0,9.8,-23.6,46.2,1176,-283,554,0,1,-1,0.0,0.1,0.1044,1
48,0,9.8,-20.3,42.1,1176,-244,505,0,1,-1,0.0,0.0733,0.0911,1
49,0,9.8,-17.7,38.5,1176,-212,462,0,1,-1,0.0,0.0578,0.08,1
50,0,9.8,-15.8,35.4,1176,-190,425,0,1,-1,0.0,0.0422,0.0689,1
51,0,8.2,-15.8,35.4,984,-190,425,-1,0,0,0.2,0.0,0.0,1
52,0,8.2,-17.7,38.5,984,-212,462,0,-1,1,0.0,0.0422,0.0689,1
53,0,8.2,-20.3,42.1,984,-244,505,0,-1,1,0.0,0.0578,0.08,1
54,0,8.2,-23.6,46.2,984,-283,554,0,-1,1,0.0,0.0733,0.0911,1
55,0,6.8,-23.6,46.2,816,-283,554,-1,0,0,0.175,0.0,0.0,1
56,0,6.8,-20.3,42.1,816,-244,505,0,1,-1,0.0,0.0733,0.0911,1
57,0,6.8,-17.7,38.5,816,-212,462,0,1,-1,0.0,0.0578,0.08,1
58,0,6.8,-15.8,35.4,816,-190,425,0,1,-1,0.0,0.0422,0.0689,1
59,0,6.8,-14.2,32.6,816,-170,391,0,1,-1,0.0,0.0356,0.0622,1
60,0,5.2,-14.2,32.6,624,-170,391,-1,0,0,0.2,0.0,0.0,1
61,1,6.8,-10.1,24.6,816,-121,295,1,1,-1,0.2,0.0911,0.1778,1
62,1,6.8,-9.4,23.1,816,-113,277,0,1,-1,0.0,0.0156,0.0333,1
63,1,6.8,-10.1,24.6,816,-121,295,0,-1,1,0.0,0.0156,0.0333,1
64,2,23.2,-12.9,30.2,2784,-155,362,1,-1,1,2.05,0.0622,0.1244,1
65,2,24.8,-12.9,30.2,2976,-155,362,1,0,0,0.2,0.0,0.0,1
66,2,24.8,-11.8,28.1,2976,-142,337,0,1,-1,0.0,0.0244,0.0467,1
67,2,24.8,-10.9,26.2,2976,-131,314,0,1,-1,0.0,0.02,0.0422,1
68,2,24.8,-10.1,24.6,2976,-121,295,0,1,-1,0.0,0.0178,0.0356,1
69,2,24.8,-9.4,23.1,2976,-113,277,0,1,-1,0.0,0.0156,0.0333,1
70,2,24.8,-8.8,21.8,2976,-106,262,0,1,-1,0.0,0.0133,0.0289,1
71,2,24.8,-8.3,20.6,2976,-100,247,0,1,-1,0.0,0.0111,0.0267,1
72,2,26.2,-8.3,20.6,3144,-100,247,1,0,0,0.175,0.0,0.0,1
73,2,26.2,-7.8,19.5,3144,-94,234,0,1,-1,0.0,0.0111,0.0244,1
74,2,26.2,-7.4,18.6,3144,-89,223,0,1,-1,0.0,0.0089,0.02,1
75,2,27.8,-7.4,18.6,3336,-89,223,1,0,0,0.2,0.0,0.0,1
76,2,27.8,-7.8,19.5,3336,-94,234,0,-1,1,0.0,0.0089,0.02,1
77,2,27.8,-8.3,20.6,3336,-100,247,0,-1,1,0.0,0.0111,0.0244,1
78,2,27.8,-8.8,21.8,3336,-106,262,0,-1,1,0.0,0.0111,0.0267,1
79,2,27.8,-9.4,23.1,3336,-113,277,0,-1,1,0.0,0.0133,0.0289,1
80,2,27.8,-10.1,24.6,3336,-121,295,0,-1,1,0.0,0.0156,0.0333,1
81,2,27.8,-10.9,26.2,3336,-131,314,0,-1,1,0.0,0.0178,0.0356,1
82,2,27.8,-11.8,28.1,3336,-142,337,0,-1,1,0.0,0.02,0.0422,1
83,2,27.8,-12.9,30.2,3336,-155,362,0,-1,1,0.0,0.0244,0.0467,1
84,2,27.8,-14.2,32.6,3336,-170,391,0,-1,1,0.0,0.0289,0.0533,1
85,2,27.8,-15.8,35.4,3336,-190,425,0,-1,1,0.0,0.0356,0.0622,1
86,2,27.8,-17.7,38.5,3336,-212,462,0,-1,1,0.0,0.0422,0.0689,1
87,2,27.8,-20.3,42.1,3336,-244,505,0,-1,1,0.0,0.0578,0.08,1
88,2,27.8,-23.6,46.2,3336,-283,554,0,-1,1,0.0,0.0733,0.0911,1
89,2,27.8,-28.1,50.9,3336,-337,611,0,-1,1,0.0,0.1,0.1044,1
90,2,27.8,-34.4,55.9,3336,-413,671,0,-1,1,0.0,0.14,0.1111,1
91,2,27.8,-43.8,61.1,3336,-526,733,0,-1,1,0.0,0.2089,0.1156,1
92,2,27.8,-58.0,65.7,3336,-696,788,0,-1,1,0.0,0.3156,0.1022,1
93,2,29.2,-58.0,65.7,3504,-696,788,1,0,0,0.175,0.0,0.0,1
94,2,29.2,-43.8,61.1,3504,-526,733,0,1,-1,0.0,0.3156,0.1022,1
95,2,29.2,-34.4,55.9,3504,-413,671,0,1,-1,0.0,0.2089,0.1156,1
96,2,29.2,-28.1,50.9,3504,-337,611,0,1,-1,0.0,0.14,0.1111,1
97,2,29.2,-23.6,46.2,3504,-283,554,0,1,-1,0.0,0.1,0.1044,1
98,2,29.2,-20.3,42.1,3504,-244,505,0,1,-1,0.0,0.0733,0.0911,1
99,2,29.2,-17.7,38.5,3504,-212,462,0,1,-1,0.0,0.0578,0.08,1
100,2,29.2,-15.8,35.4,3504,-190,425,0,1,-1,0.0,0.0422,0.0689,1
101,2,29.2,-14.2,32.6,3504,-170,391,0,1,-1,0.0,0.0356,0.0622,1
102,2,29.2,-12.9,30.2,3504,-155,362,0,1,-1,0.0,0.0289,0.0533,1
103,2,29.2,-11.8,28.1,3504,-142,337,0,1,-1,0.0,0.0244,0.0467,1
104,2,29.2,-10.9,26.2,3504,-131,314,0,1,-1,0.0,0.02,0.0422,1
105,2,29.2,-10.1,24.6,3504,-121,295,0,1,-1,0.0,0.0178,0.0356,1
106,2,29.2,-9.4,23.1,3504,-113,277,0,1,-1,0.0,0.0156,0.0333,1
107,2,29.2,-8.8,21.8,3504,-106,262,0,1,-1,0.0,0.0133,0.0289,1
108,2,29.2,-8.3,20.6,3504,-100,247,0,1,-1,0.0,0.0111,0.0267,1
109,2,29.2,-7.8,19.5,3504,-94,234,0,1,-1,0.0,0.0111,0.0244,1
110,2,29.2,-7.4,18.6,3504,-89,223,0,1,-1,0.0,0.0089,0.02,1
111,2,30.8,-7.4,18.6,3696,-89,223,1,0,0,0.2,0.0,0.0,1
112,2,30.8,-7.8,19.5,3696,-94,234,0,-1,1,0.0,0.0089,0.02,1
113,2,30.8,-8.3,20.6,3696,-100,247,0,-1,1,0.0,0.0111,0.0244,1
114,2,30.8,-8.8,21.8,3696,-106,262,0,-1,1,0.0,0.0111,0.0267,1
115,2,30.8,-9.4,23.1,3696,-113,277,0,-1,1,0.0,0.0133,0.0289,1
116,2,30.8,-10.1,24.6,3696,-121,295,0,-1,1,0.0,0.0156,0.0333,1
117,2,30.8,-10.9,26.2,3696,-131,314,0,-1,1,0.0,0.0178,0.0356,1
118,2,30.8,-11.8,28.1,3696,-142,337,0,-1,1,0.0,0.02,0.0422,1
119,2,30.8,-12.9,30.2,3696,-155,362,0,-1,1,0.0,0.0244,0.0467,1
120,2,30.8,-14.2,32.6,3696,-170,391,0,-1,1,0.0,0.0289,0.0533,1
121,2,30.8,-15.8,35.4,3696,-190,425,0,-1,1,0.0,0.0356,0.0622,1
122,2,30.8,-17.7,38.5,3696,-212,462,0,-1,1,0.0,0.0422,0.0689,1
123,2,30.8,-20.3,42.1,3696,-244,505,0,-1,1,0.0,0.0578,0.08,1
124,2,30.8,-23.6,46.2,3696,-283,554,0,-1,1,0.0,0.0733,0.0911,1
125,2,30.8,-28.1,50.9,3696,-337,611,0,-1,1,0.0,0.1,0.1044,1
126,2,30.8,-34.4,55.9,3696,-413,671,0,-1,1,0.0,0.14,0.1111,1
127,2,30.8,-43.8,61.1,3696,-526,733,0,-1,1,0.0,0.2089,0.1156,1
128,2,30.8,-58.0,65.7,3696,-696,788,0,-1,1,0.0,0.3156,0.1022,1
129,2,32.2,-58.0,65.7,3864,-696,788,1,0,0,0.175,0.0,0.0,1
130,2,32.2,-43.8,61.1,3864,-526,733,0,1,-1,0.0,0.3156,0.1022,1
131,2,32.2,-34.4,55.9,3864,-413,671,0,1,-1,0.0,0.2089,0.1156,1
132,2,32.2,-28.1,50.9,3864,-337,611,0,1,-1,0.0,0.14,0.1111,1
133,2,32.2,-23.6,46.2,3864,-283,554,0,1,-1,0.0,0.1,0.1044,1
134,2,32.2,-20.3,42.1,3864,-244,505,0,1,-1,0.0,0.0733,0.0911,1
135,2,32.2,-17.7,38.5,3864,-212,462,0,1,-1,0.0,0.0578,0.08,1
136,2,32.2,-15.8,35.4,3864,-190,425,0,1,-1,0.0,0.0422,0.0689,1
137,2,32.2,-14.2,32.6,3864,-170,391,0,1,-1,0.0,0.0356,0.0622,1
138,2,32.2,-12.9,30.2,3864,-155,362,0,1,-1,0.0,0.0289,0.0533,1
139,2,32.2,-11.8,28.1,3864,-142,337,0,1,-1,0.0,0.0244,0.0467,1
140,2,32.2,-10.9,26.2,3864,-131,314,0,1,-1,0.0,0.02,0.0422,1
141,2,32.2,-10.1,24.6,3864,-121,295,0,1,-1,0.0,0.0178,0.0356,1
142,2,32.2,-9.4,23.1,3864,-113,277,0,1,-1,0.0,0.0156,0.0333,1
143,2,32.2,-8.8,21.8,3864,-106,262,0,1,-1,0.0,0.0133,0.0289,1
144,2,32.2,-8.3,20.6,3864,-100,247,0,1,-1,0.0,0.0111,0.0267,1
145,2,32.2,-7.8,19.5,3864,-94,234,0,1,-1,0.0,0.0111,0.0244,1
146,2,32.2,-7.4,18.6,3864,-89,223,0,1,-1,0.0,0.0089,0.02,1
147,2,33.8,-7.4,18.6,4056,-89,223,1,0,0,0.2,0.0,0.0,1
148,2,33.8,-7.8,19.5,4056,-94,234,0,-1,1,0.0,0.0089,0.02,1
149,2,33.8,-8.3,20.6,4056,-100,247,0,-1,1,0.0,0.0111,0.0244,1
150,2,33.8,-8.8,21.8,4056,-106,262,0,-1,1,0.0,0.0111,0.0267,1
151,2,33.8,-9.4,23.1,4056,-113,277,0,-1,1,0.0,0.0133,0.0289,1
152,2,33.8,-10.1,24.6,4056,-121,295,0,-1,1,0.0,0.0156,0.0333,1
153,2,33.8,-10.9,26.2,4056,-131,314,0,-1,1,0.0,0.0178,0.0356,1
154,2,33.8,-11.8,28.1,4056,-142,337,0,-1,1,0.0,0.02,0.0422,1
155,2,33.8,-12.9,30.2,4056,-155,362,0,-1,1,0.0,0.0244,0.0467,1
156,2,33.8,-14.2,32.6,4056,-170,391,0,-1,1,0.0,0.0289,0.0533,1
157,2,33.8,-15.8,35.4,4056,-190,425,0,-1,1,0.0,0.0356,0.0622,1
158,2,33.8,-17.7,38.5,4056,-212,462,0,-1,1,0.0,0.0422,0.0689,1
159,2,33.8,-20.3,42.1,4056,-244,505,0,-1,1,0.0,0.0578,0.08,1
160,2,33.8,-23.6,46.2,4056,-283,554,0,-1,1,0.0,0.0733,0.0911,1
161,2,33.8,-28.1,50.9,4056,-337,611,0,-1,1,0.0,0.1,0.1044,1
162,2,33.8,-34.4,55.9,4056,-413,671,0,-1,1,0.0,0.14,0.1111,1
163,2,33.8,-43.8,61.1,4056,-526,733,0,-1,1,0.0,0.2089,0.1156,1
164,2,33.8,-58.0,65.7,4056,-696,788,0,-1,1,0.0,0.3156,0.1022,1
165,2,33.8,-43.8,61.1,4056,-526,733,0,1,-1,0.0,0.3156,0.1022,1
166,2,33.8,-34.4,55.9,4056,-413,671,0,1,-1,0.0,0.2089,0.1156,1
167,2,33.8,-28.1,50.9,4056,-337,611,0,1,-1,0.0,0.14,0.1111,1
168,2,33.8,-23.6,46.2,4056,-283,554,0,1,-1,0.0,0.1,0.1044,1
169,2,33.8,-20.3,42.1,4056,-244,505,0,1,-1,0.0,0.0733,0.0911,1
170,2,33.8,-17.7,38.5,4056,-212,462,0,1,-1,0.0,0.0578,0.08,1
171,2,33.8,-15.8,35.4,4056,-190,425,0,1,-1,0.0,0.0422,0.0689,1
172,2,33.8,-14.2,32.6,4056,-170,391,0,1,-1,0.0,0.0356,0.0622,1
173,2,33.8,-12.9,30.2,4056,-155,362,0,1,-1,0.0,0.0289,0.0533,1
174,2,33.8,-11.8,28.1,4056,-142,337,0,1,-1,0.0,0.0244,0.0467,1
175,2,33.8,-10.9,26.2,4056,-131,314,0,1,-1,0.0,0.02,0.0422,1
176,2,33.8,-10.1,24.6,4056,-121,295,0,1,-1,0.0,0.0178,0.0356,1
177,2,33.8,-9.4,23.1,4056,-113,277,0,1,-1,0.0,0.0156,0.0333,1
178,2,33.8,-8.8,21.8,4056,-106,262,0,1,-1,0.0,0.0133,0.0289,1
179,2,33.8,-8.3,20.6,4056,-100,247,0,1,-1,0.0,0.0111,0.0267,1
180,2,33.8,-7.8,19.5,4056,-94,234,0,1,-1,0.0,0.0111,0.0244,1
181,2,33.8,-7.4,18.6,4056,-89,223,0,1,-1,0.0,0.0089,0.02,1
182,2,32.2,-7.4,18.6,3864,-89,223,-1,0,0,0.2,0.0,0.0,1
183,2,32.2,-7.8,19.5,3864,-94,234,0,-1,1,0.0,0.0089,0.02,1
184,2,32.2,-8.3,20.6,3864,-100,247,0,-1,1,0.0,0.0111,0.0244,1
185,2,32.2,-8.8,21.8,3864,-106,262,0,-1,1,0.0,0.0111,0.0267,1
186,2,32.2,-9.4,23.1,3864,-113,277,0,-1,1,0.0,0.0133,0.0289,1
187,2,32.2,-10.1,24.6,3864,-121,295,0,-1,1,0.0,0.0156,0.0333,1
188,2,32.2,-10.9,26.2,3864,-131,314,0,-1,1,0.0,0.0178,0.0356,1
189,2,32.2,-11.8,28.1,3864,-142,337,0,-1,1,0.0,0.02,0.0422,1
190,2,32.2,-12.9,30.2,3864,-155,362,0,-1,1,0.0,0.0244,0.0467,1
191,2,32.2,-14.2,32.6,3864,-170,391,0,-1,1,0.0,0.0289,0.0533,1
192,2,32.2,-15.8,35.4,3864,-190,425,0,-1,1,0.0,0.0356,0.0622,1
193,2,32.2,-17.7,38.5,3864,-212,462,0,-1,1,0.0,0.0422,0.0689,1
194,2,32.2,-20.3,42.1,3864,-244,505,0,-1,1,0.0,0.0578,0.08,1
195,2,32.2,-23.6,46.2,3864,-283,554,0,-1,1,0.0,0.0733,0.0911,1
196,2,32.2,-28.1,50.9,3864,-337,611,0,-1,1,0.0,0.1,0.1044,1
197,2,32.2,-34.4,55.9,3864,-413,671,0,-1,1,0.0,0.14,0.1111,1
198,2,32.2,-43.8,61.1,3864,-526,733,0,-1,1,0.0,0.2089,0.1156,1
199,2,32.2,-58.0,65.7,3864,-696,788,0,-1,1,0.0,0.3156,0.1022,1
200,2,30.8,-58.0,65.7,3696,-696,788,-1,0,0,0.175,0.0,0.0,1
201,2,30.8,-43.8,61.1,3696,-526,733,0,1,-1,0.0,0.3156,0.1022,1
202,2,30.8,-34.4,55.9,3696,-413,671,0,1,-1,0.0,0.2089,0.1156,1
203,2,30.8,-28.1,50.9,3696,-337,611,0,1,-1,0.0,0.14,0.1111,1
204,2,30.8,-23.6,46.2,3696,-283,554,0,1,-1,0.0,0.1,0.1044,1
205,2,30.8,-20.3,42.1,3696,-244,505,0,1,-1,0.0,0.0733,0.0911,1
206,2,30.8,-17.7,38.5,3696,-212,462,0,1,-1,0.0,0.0578,0.08,1
207,2,30.8,-15.8,35.4,3696,-190,425,0,1,-1,0.0,0.0422,0.0689,1
208,2,30.8,-14.2,32.6,3696,-170,391,0,1,-1,0.0,0.0356,0.0622,1
209,2,30.8,-12.9,30.2,3696,-155,362,0,1,-1,0.0,0.0289,0.0533,1
210,2,30.8,-11.8,28.1,3696,-142,337,0,1,-1,0.0,0.0244,0.0467,1
211,2,30.8,-10.9,26.2,3696,-131,314,0,1,-1,0.0,0.02,0.0422,1
212,2,30.8,-10.1,24.6,3696,-121,295,0,1,-1,0.0,0.0178,0.0356,1
213,2,30.8,-9.4,23.1,3696,-113,277,0,1,-1,0.0,0.0156,0.0333,1
214,2,30.8,-8.8,21.8,3696,-106,262,0,1,-1,0.0,0.0133,0.0289,1
215,2,30.8,-8.3,20.6,3696,-100,247,0,1,-1,0.0,0.0111,0.0267,1
216,2,30.8,-7.8,19.5,3696,-94,234,0,1,-1,0.0,0.0111,0.0244,1
217,2,30.8,-7.4,18.6,3696,-89,223,0,1,-1,0.0,0.0089,0.02,1
218,2,29.2,-7.4,18.6,3504,-89,223,-1,0,0,0.2,0.0,0.0,1
219,2,29.2,-7.8,19.5,3504,-94,234,0,-1,1,0.0,0.0089,0.02,1
220,2,29.2,-8.3,20.6,3504,-100,247,0,-1,1,0.0,0.0111,0.0244,1
221,2,29.2,-8.8,21.8,3504,-106,262,0,-1,1,0.0,0.0111,0.0267,1
222,2,29.2,-9.4,23.1,3504,-113,277,0,-1,1,0.0,0.0133,0.0289,1
223,2,29.2,-10.1,24.6,3504,-121,295,0,-1,1,0.0,0.0156,0.0333,1
224,2,29.2,-10.9,26.2,3504,-131,314,0,-1,1,0.0,0.0178,0.0356,1
225,2,29.2,-11.8,28.1,3504,-142,337,0,-1,1,0.0,0.02,0.0422,1
226,2,29.2,-12.9,30.2,3504,-155,362,0,-1,1,0.0,0.0244,0.0467,1
227,2,29.2,-14.2,32.6,3504,-170,391,0,-1,1,0.0,0.0289,0.0533,1
228,2,29.2,-15.8,35.4,3504,-190,425,0,-1,1,0.0,0.0356,0.0622,1
229,2,29.2,-17.7,38.5,3504,-212,462,0,-1,1,0.0,0.0422,0.0689,1
230,2,29.2,-20.3,42.1,3504,-244,505,0,-1,1,0.0,0.0578,0.08,1
231,2,29.2,-23.6,46.2,3504,-283,554,0,-1,1,0.0,0.0733,0.0911,1
232,2,29.2,-28.1,50.9,3504,-337,611,0,-1,1,0.0,0.1,0.1044,1
233,2,29.2,-34.4,55.9,3504,-413,671,0,-1,1,0.0,0.14,0.1111,1
234,2,29.2,-43.8,61.1,3504,-526,733,0,-1,1,0.0,0.2089,0.1156,1
235,2,29.2,-58.0,65.7,3504,-696,788,0,-1,1,0.0,0.3156,0.1022,1
236,2,27.8,-58.0,65.7,3336,-696,788,-1,0,0,0.175,0.0,0.0,1
237,2,26.2,-58.0,65.7,3144,-696,788,-1,0,0,0.2,0.0,0.0,1
238,2,26.2,-43.8,61.1,3144,-526,733,0,1,-1,0.0,0.3156,0.1022,1
239,2,26.2,-34.4,55.9,3144,-413,671,0,1,-1,0.0,0.2089,0.1156,1
240,2,26.2,-28.1,50.9,3144,-337,611,0,1,-1,0.0,0.14,0.1111,1
241,2,26.2,-23.6,46.2,3144,-283,554,0,1,-1,0.0,0.1,0.1044,1
242,2,26.2,-20.3,42.1,3144,-244,505,0,1,-1,0.0,0.0733,0.0911,1
243,2,26.2,-17.7,38.5,3144,-212,462,0,1,-1,0.0,0.0578,0.08,1
244,2,26.2,-15.8,35.4,3144,-190,425,0,1,-1,0.0,0.0422,0.0689,1
245,2,26.2,-14.2,32.6,3144,-170,391,0,1,-1,0.0,0.0356,0.0622,1
246,2,26.2,-12.9,30.2,3144,-155,362,0,1,-1,0.0,0.0289,0.0533,1
247,2,26.2,-11.8,28.1,3144,-142,337,0,1,-1,0.0,0.0244,0.0467,1
248,2,26.2,-10.9,26.2,3144,-131,314,0,1,-1,0.0,0.02,0.0422,1
249,2,26.2,-10.1,24.6,3144,-121,295,0,1,-1,0.0,0.0178,0.0356,1
250,2,26.2,-9.4,23.1,3144,-113,277,0,1,-1,0.0,0.0156,0.0333,1
251,2,26.2,-8.8,21.8,3144,-106,262,0,1,-1,0.0,0.0133,0.0289,1
252,2,26.2,-9.4,23.1,3144,-113,277,0,-1,1,0.0,0.0133,0.0289,1
253,2,26.2,-10.1,24.6,3144,-121,295,0,-1,1,0.0,0.0156,0.0333,1
254,2,26.2,-10.9,26.2,3144,-131,314,0,-1,1,0.0,0.0178,0.0356,1
255,2,26.2,-11.8,28.1,3144,-142,337,0,-1,1,0.0,0.02,0.0422,1
256,2,26.2,-12.9,30.2,3144,-155,362,0,-1,1,0.0,0.0244,0.0467,1
257,2,26.2,-14.2,32.6,3144,-170,391,0,-1,1,0.0,0.0289,0.0533,1
258,2,24.8,-14.2,32.6,2976,-170,391,-1,0,0,0.175,0.0,0.0,1
259,2,24.8,-15.8,35.4,2976,-190,425,0,-1,1,0.0,0.0356,0.0622,1
260,2,24.8,-17.7,38.5,2976,-212,462,0,-1,1,0.0,0.0422,0.0689,1
261,2,24.8,-20.3,42.1,2976,-244,505,0,-1,1,0.0,0.0578,0.08,1
262,2,24.8,-17.7,38.5,2976,-212,462,0,1,-1,0.0,0.0578,0.08,1
263,2,24.8,-15.8,35.4,2976,-190,425,0,1,-1,0.0,0.0422,0.0689,1
264,2,24.8,-14.2,32.6,2976,-170,391,0,1,-1,0.0,0.0356,0.0622,1
265,2,26.2,-14.2,32.6,3144,-170,391,1,0,0,0.175,0.0,0.0,1
266,2,26.2,-15.8,35.4,3144,-190,425,0,-1,1,0.0,0.0356,0.0622,1
267,2,26.2,-17.7,38.5,3144,-212,462,0,-1,1,0.0,0.0422,0.0689,1
268,2,26.2,-20.3,42.1,3144,-244,505,0,-1,1,0.0,0.0578,0.08,1
269,2,26.2,-23.6,46.2,3144,-283,554,0,-1,1,0.0,0.0733,0.0911,1
270,2,26.2,-28.1,50.9,3144,-337,611,0,-1,1,0.0,0.1,0.1044,1
271,2,26.2,-34.4,55.9,3144,-413,671,0,-1,1,0.0,0.14,0.1111,1
272,2,26.2,-43.8,61.1,3144,-526,733,0,-1,1,0.0,0.2089,0.1156,1
273,2,24.8,-43.8,61.1,2976,-526,733,-1,0,0,0.175,0.0,0.0,1
274,2,24.8,-58.0,65.7,2976,-696,788,0,-1,1,0.0,0.3156,0.1022,1
275,2,24.8,-43.8,61.1,2976,-526,733,0,1,-1,0.0,0.3156,0.1022,1
276,2,26.2,-43.8,61.1,3144,-526,733,1,0,0,0.175,0.0,0.0,1
277,2,26.2,-58.0,65.7,3144,-696,788,0,-1,1,0.0,0.3156,0.1022,1
278,2,27.8,-58.0,65.7,3336,-696,788,1,0,0,0.2,0.0,0.0,1
279,2,27.8,-43.8,61.1,3336,-526,733,0,1,-1,0.0,0.3156,0.1022,1
280,2,27.8,-34.4,55.9,3336,-413,671,0,1,-1,0.0,0.2089,0.1156,1
281,2,27.8,-28.1,50.9,3336,-337,611,0,1,-1,0.0,0.14,0.1111,1
282,2,27.8,-23.6,46.2,3336,-283,554,0,1,-1,0.0,0.1,0.1044,1
283,2,27.8,-20.3,42.1,3336,-244,505,0,1,-1,0.0,0.0733,0.0911,1
284,2,27.8,-17.7,38.5,3336,-212,462,0,1,-1,0.0,0.0578,0.08,1
285,2,27.8,-15.8,35.4,3336,-190,425,0,1,-1,0.0,0.0422,0.0689,1
286,2,27.8,-14.2,32.6,3336,-170,391,0,1,-1,0.0,0.0356,0.0622,1
287,2,27.8,-12.9,30.2,3336,-155,362,0,1,-1,0.0,0.0289,0.0533,1
288,2,27.8,-11.8,28.1,3336,-142,337,0,1,-1,0.0,0.0244,0.0467,1
289,2,27.8,-10.9,26.2,3336,-131,314,0,1,-1,0.0,0.02,0.0422,1
290,2,27.8,-10.1,24.6,3336,-121,295,0,1,-1,0.0,0.0178,0.0356,1
291,2,27.8,-9.4,23.1,3336,-113,277,0,1,-1,0.0,0.0156,0.0333,1
292,2,27.8,-8.8,21.8,3336,-106,262,0,1,-1,0.0,0.0133,0.0289,1
293,2,27.8,-8.3,20.6,3336,-100,247,0,1,-1,0.0,0.0111,0.0267,1
294,2,27.8,-7.8,19.5,3336,-94,234,0,1,-1,0.0,0.0111,0.0244,1
295,2,27.8,-7.4,18.6,3336,-89,223,0,1,-1,0.0,0.0089,0.02,1
296,2,26.2,-7.4,18.6,3144,-89,223,-1,0,0,0.2,0.0,0.0,1
297,2,26.2,-7.8,19.5,3144,-94,234,0,-1,1,0.0,0.0089,0.02,1
298,2,26.2,-8.3,20.6,3144,-100,247,0,-1,1,0.0,0.0111,0.0244,1
299,2,24.8,-8.3,20.6,2976,-100,247,-1,0,0,0.175,0.0,0.0,1
300,2,24.8,-8.8,21.8,2976,-106,262,0,-1,1,0.0,0.0111,0.0267,1
301,2,23.2,-8.8,21.8,2784,-106,262,-1,0,0,0.2,0.0,0.0,1
302,2,23.2,-9.4,23.1,2784,-113,277,0,-1,1,0.0,0.0133,0.0289,1
303,2,23.2,-10.1,24.6,2784,-121,295,0,-1,1,0.0,0.0156,0.0333,1
304,2,23.2,-9.4,23.1,2784,-113,277,0,1,-1,0.0,0.0156,0.0333,1
305,2,23.2,-8.8,21.8,2784,-106,262,0,1,-1,0.0,0.0133,0.0289,1
306,2,24.8,-8.8,21.8,2976,-106,262,1,0,0,0.2,0.0,0.0,1
307,2,24.8,-9.4,23.1,2976,-113,277,0,-1,1,0.0,0.0133,0.0289,1
308,2,24.8,-10.1,24.6,2976,-121,295,0,-1,1,0.0,0.0156,0.0333,1
309,2,24.8,-10.9,26.2,2976,-131,314,0,-1,1,0.0,0.0178,0.0356,1
310,2,24.8,-11.8,28.1,2976,-142,337,0,-1,1,0.0,0.02,0.0422,1
311,2,24.8,-12.9,30.2,2976,-155,362,0,-1,1,0.0,0.0244,0.0467,1
312,2,23.2,-12.9,30.2,2784,-155,362,-1,0,0,0.2,0.0,0.0,1
"""


if __name__ == "__main__":
    main()
