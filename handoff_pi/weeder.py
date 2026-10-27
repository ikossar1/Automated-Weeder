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
import gc
import math
import pathlib
import signal
import sys
import time

# ---------------------------------------------------------------------------
# Wiring. These are the numbers to check first when something moves the
# wrong way - they are facts about the build, not about the maths.
# ---------------------------------------------------------------------------

# Confirmed by driving each motor and watching, not inferred from the
# jog script's key names - which named them after arrow keys and led to
# the wrong map twice.
PINS = {
    #          forward pin, reverse pin
    "pan":  (23, 24),       # 23 HIGH swings the nozzle one way
    "tilt": (27, 22),       # 27 HIGH tips the nozzle down
}

# The RAIL is the stepper, on an EasyDriver. The two DC motors both
# rotate; nothing else is left to slide the head along the bed, and a
# stepper on a leadscrew is how you would build a linear axis that has
# to know where it is.
#
# This is the good axis. A DC motor is told to run for a length of time
# and we hope. A stepper is told to take 400 steps and takes 400 steps,
# so the rail does not accumulate error and does not need re-homing
# before every box the way a timed drive did.
RAIL_STEP_PIN = 25
RAIL_DIR_PIN = 6
RAIL_DIR_FORWARD = True      # True if DIR high moves away from the home stop

# MEASURE THIS with --rail-sweep.
#
# The default assumes a 200 step motor at the EasyDriver's 1/8 microstep
# default on an 8 mm lead screw:
#     200 * 8 = 1600 steps per turn, 8 mm of travel per turn
#     = 200 steps per mm = 2000 per cm
# A different lead changes it completely - a 2 mm lead is four times
# this - so it is a starting point, not a number to trust.
RAIL_STEPS_PER_CM = 2000.0

# Seconds per half pulse at full speed, and where the ramp starts.
#
# A stepper asked to start straight into a high step rate does not turn
# at all: it has to accelerate into it. Running at 1036 steps a second
# from standstill is well past what a small motor will pull in, and the
# symptom is exactly what was seen - a motor that buzzes and does not
# move. So every move starts slow and speeds up.
RAIL_STEP_DELAY = 0.00040        # fastest, once up to speed
RAIL_START_DELAY = 0.0025        # first step, slow enough to pull in
RAIL_RAMP_STEPS = 120            # steps taken getting from one to the other

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
# pan swung the wrong way on the machine: a box to the left was reached
# by turning right. Inverted here rather than by swapping the wires.
INVERT = {"rail": False, "pan": True, "tilt": False}

# Measured, not guessed. Run --calibrate-rail to get the first one.
# How hard the DC motors are driven, as a percentage of full power.
#
# They have no gearbox slow enough to be useful at full power, so they
# are switched on and off a thousand times a second and only left on for
# this fraction of each cycle. The motor sees the average and turns
# proportionally slower.
#
# Lower is slower, but not forever: below roughly 25% most small DC
# motors cannot overcome their own friction and will buzz without
# turning. If an axis stops moving rather than moving slowly, this is
# too low, not too high.
# 100 Hz rather than a few kHz. A brushed motor turning slowly needs
# each pulse to be long enough to break static friction, and at 1 kHz
# the pulses are too short to do that - the motor hums and sits still.
# A slower switching frequency gives a bigger kick per pulse, which is
# what makes a low duty cycle actually turn the shaft.
PWM_HZ = 100
DUTY = 15.0

# Homing is driven harder than that. It is a blind run into a hard stop
# whose whole purpose is to overshoot, so the slow, careful speed used
# for aiming buys nothing and costs 40 seconds a box. Not full power
# either, because this ends in a collision every single time.
HOME_DUTY = 70.0

# Measured with a stopwatch at FULL power, against the axes as they
# actually are: 23/24 is pan, 27/22 is tilt.
SPEED_DEG_FULL = {"pan": 50.0, "tilt": 42.86}

# What to expect at DUTY. A DC motor is not linear in duty cycle - it
# loses proportionally more to friction the slower it goes - so this is
# an estimate and the plan's angles are only as good as it is. Re-time
# an axis at this duty and put the real figure in if the aim is off.
SPEED_DEG = {a: v * DUTY / 100.0 for a, v in SPEED_DEG_FULL.items()}

# Only the DC axes have a speed in here. The rail is a stepper and its
# rate comes from RAIL_STEP_DELAY, not from a guess about voltage.
SPEED = {"pan": SPEED_DEG["pan"], "tilt": SPEED_DEG["tilt"]}
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
# A hard ceiling in SECONDS on any one drive. The degree figures only
# mean what they say if the speed estimate is right, and it is not:
# moves that should have been a few degrees ran far past. Until each
# axis has been timed at the duty it will really run at, this is what
# bounds the damage, and it is deliberately short.
MAX_DRIVE_S_HARD = 0.35

MAX_DRIVE_DEG = {"pan": 10.0, "tilt": 10.0}

# Above this tilt the shot is near enough vertical that pan stops
# mattering: the nozzle is looking almost straight down, so swinging it
# left or right moves the landing point by very little.
NEAR_VERTICAL_DEG = 75.0

# How many capped drives one move may be split into. Enough to cross the
# full range of either axis, and a bound on how long a single command can
# keep a motor running if a speed is badly wrong.
# ONE drive per command, not a sequence of them. Splitting a move into
# several capped drives assumed the speed estimate was roughly right, so
# repeating until the target was reached would arrive. It is not right -
# a single drive already overshot - and repeating drove further past.
MAX_DRIVE_STEPS = 1
GRACE = 1.2

MAX_DRIVE_S = {a: min(MAX_DRIVE_S_HARD,
                      round(MAX_DRIVE_DEG[a] / SPEED_DEG[a] * GRACE, 2))
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
        self.pwm = {}
        # Where the stepper believes the head is, in cm from the home
        # stop. Set by homing, then kept by counting steps.
        self.rail_cm = 0.0
        # Where the two DC axes are believed to be, in degrees from the
        # datum set by hand at the start of the run. Believed, not known:
        # nothing measures them, so this is the running total of what
        # they were told to do.
        self.angle = {"pan": 0.0, "tilt": 0.0}
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
        self.pwm = {}
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
                + list(VALVE_PINS) + [RAIL_STEP_PIN, RAIL_DIR_PIN])
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
        # Every motor pin is driven through PWM so its speed can be set.
        # The valve and the stepper are not: the valve is open or shut,
        # and a stepper's speed is the gap between pulses, not a voltage.
        try:
            for a, b in PINS.values():
                for pin in (a, b):
                    self.pwm[pin] = GPIO.PWM(pin, PWM_HZ)
                    self.pwm[pin].start(0)
        except Exception as exc:
            # Without PWM the motors only have one speed, so this is not
            # something to carry on quietly from.
            sys.exit(f"Could not set up PWM: {exc}" + NL + NL
                     + "Without it the motors run at full power only." + NL
                     + "On a Pi 5 this usually means the GPIO library:" + NL
                     + "    sudo apt install -y python3-rpi-lgpio")

        self.all_off()

    def _motor(self, pin, duty):
        if self.dry or pin not in self.pwm:
            return
        self.pwm[pin].ChangeDutyCycle(duty)

    def _write(self, pin, high):
        if self.dry:
            return
        self.gpio.output(pin, self.gpio.HIGH if high else self.gpio.LOW)

    def drive(self, axis, direction, duty=None):
        """direction is +1, -1 or 0. Nothing else is accepted."""
        fwd, rev = PINS[axis]
        if INVERT[axis]:
            direction = -direction
        d = DUTY if duty is None else duty
        self._motor(fwd, d if direction > 0 else 0.0)
        self._motor(rev, d if direction < 0 else 0.0)
        self.log.append((axis, direction))

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

    def turn_to(self, axis, degrees, max_turn=None):
        """
        Drive a DC axis to an absolute angle, not by a relative nudge.

        The plan only ever stated differences between consecutive rows,
        which silently assumed the machine already sat at the first row's
        angles. It does not: it sits at the datum. So the first move of a
        box - the big one, from neutral out to the aim point - was never
        commanded and nothing appeared to happen.

        Working from an absolute target fixes that, and lets the duration
        come from the speed measured on this machine rather than from
        whatever the planner assumed.
        """
        cap = max_turn if (max_turn and axis == "pan") else MAX_DRIVE_S[axis]
        want = degrees - self.angle[axis]
        if abs(want) < 0.05:
            return 0.0

        # The cap limits how long ONE drive may run, not how far the axis
        # may travel. A legitimate 80 degree swing from the datum is made
        # of several capped drives, not abandoned after the first - and
        # pausing between them is what keeps a runaway short.
        took = 0.0
        for _ in range(MAX_DRIVE_STEPS):
            delta = degrees - self.angle[axis]
            secs = abs(delta) / SPEED_DEG[axis]
            if secs < MIN_MOVE_S:
                break
            secs = min(secs, cap)

            way = 1 if delta > 0 else -1
            self.drive(axis, way)
            wait(self, secs)
            self.stop(axis)
            wait(self, SETTLE_S)
            self.angle[axis] += way * secs * SPEED_DEG[axis]
            took += secs + SETTLE_S

        left = degrees - self.angle[axis]
        if abs(left) > 1.0:
            print(f"      {axis} stopped {abs(left):.0f} deg short of "
                  f"{degrees:.0f} - raise MAX_DRIVE_STEPS or the duty")
        elif took:
            print(f"      {axis} -> {degrees:+.0f} deg in {took:.1f}s")
        return took

    def rail_to(self, cm):
        """
        Step the head to an absolute position along the bed.

        Absolute, not relative, because a stepper holds a true position
        and there is no reason to let error pile up on the one axis that
        does not have to have any.

        Every move accelerates. A stepper told to start at full rate
        does not turn - it has to be brought up to speed - and the
        symptom of getting that wrong is a motor that buzzes and sits
        still, which looks like a wiring fault and is not one.
        """
        delta = cm - self.rail_cm
        steps = int(round(delta * RAIL_STEPS_PER_CM))
        if steps == 0:
            return 0.0

        forward = (steps > 0) == RAIL_DIR_FORWARD
        self._write(RAIL_DIR_PIN, forward)
        n = abs(steps)
        took = 0.0

        if not self.dry:
            time.sleep(0.001)
            for i in range(n):
                # Ramp up over the first RAIL_RAMP_STEPS, and back down
                # over the last of them, so it neither stalls starting
                # nor overruns stopping.
                near = min(i, n - 1 - i, RAIL_RAMP_STEPS)
                if near >= RAIL_RAMP_STEPS:
                    d = RAIL_STEP_DELAY
                else:
                    f = near / RAIL_RAMP_STEPS
                    d = RAIL_START_DELAY + (RAIL_STEP_DELAY
                                            - RAIL_START_DELAY) * f
                self._write(RAIL_STEP_PIN, True)
                _hold(d)
                self._write(RAIL_STEP_PIN, False)
                _hold(d)
                took += 2 * d
        else:
            took = n * 2 * RAIL_STEP_DELAY

        self.rail_cm += steps / RAIL_STEPS_PER_CM
        self.log.append(("rail", round(self.rail_cm, 3)))
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
        self._write(RAIL_STEP_PIN, False)

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
        # Stop and drop the PWM objects BEFORE closing the chip. Left
        # alive, their __del__ runs afterwards and calls into a handle
        # that is already None, which prints a stack trace at exit that
        # looks like a crash and is not one.
        for pwm in self.pwm.values():
            try:
                pwm.stop()
            except Exception:
                pass
        self.pwm.clear()
        gc.collect()
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
    Drive into the stop once, and call that zero.

    With a DC rail this had to happen before every box, because a timed
    drive drifts and the stop was the only position the machine could
    know. The rail is a stepper now: it holds its count, so homing is a
    once-per-run job and the 17 seconds a box it used to cost is gone.

    Stepping into a hard stop makes the motor skip steps, which is
    exactly what is wanted here - it ends up pressed against the stop
    whatever it did on the way.
    """
    cm = RAIL_TRAVEL_CM * HOME_EXTRA
    print(f"    homing rail  (driving {cm:.0f} cm into the stop)")
    hw.rail_cm = cm                 # pretend we are at the far end ...
    hw.rail_to(0.0)                 # ... so this drives the whole way back
    hw.rail_cm = 0.0                # and the stop is zero by definition
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
    # From here the believed angles are zero, by definition - the datum
    # is wherever the operator has just put them.
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
    # Only pan and tilt can be skipped. The rail is a stepper and takes
    # whatever steps it is given, however few.
    turn = max(SPEED["pan"], SPEED["tilt"]) * MIN_MOVE_S
    return WORKING_RANGE_CM * math.tan(math.radians(turn))


def summarise(rows):
    short = sum(1 for r in rows
                if 0 < max(float(r["rail_secs"]), float(r["pan_secs"]),
                           float(r["tilt_secs"])) < MIN_MOVE_S)
    # The rail is a stepper: its time is a count of pulses, not a guess
    # about how fast a DC motor turns.
    rail_at = 0.0
    total = 0.0
    for r in rows:
        want = float(r["rail_cm"])
        steps = abs(int(round((want - rail_at) * RAIL_STEPS_PER_CM)))
        rail_at = want
        total += (steps * RAIL_STEP_DELAY * 2
                  + max(float(r["pan_secs"]), float(r["tilt_secs"]))
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
    homing = (RAIL_TRAVEL_CM * HOME_EXTRA * RAIL_STEPS_PER_CM
              * RAIL_STEP_DELAY * 2)

    print(f"  Rows            : {len(rows):,}")
    print(f"  Boxes           : {boxes}")
    print(f"  Spraying on     : {on_rows} of {len(rows)} rows")
    print(f"  Valve changes   : {changes}")
    print(f"  Driving         : {total:.0f} s")
    print(f"  Homing          : once, {homing:.0f} s "
          f"(the rail is a stepper and holds its place)")
    print(f"  Rough total     : {(total + homing) / 60:.1f} min")

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


def run(hw, rows, rehome_every_box=True, max_turn=MAX_TURN_S,
        confirm=False, use_rail=True):
    valve_is_open = False
    last_box = None
    last_pan = None
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
            if use_rail:
                home_rail(hw)
            else:
                print("    rail off - not homing")
            last_box = box

        want_valve = r["spray"] == "1"

        # Shut before ANYTHING moves, not only before moves that are not
        # meant to spray. It was staying open across a move whenever the
        # next row also wanted it open, so the machine travelled while
        # spraying - the one thing the travel/spray split exists to stop.
        if valve_is_open:
            hw.valve(False)
            valve_is_open = False

        if skipping is not None and box == skipping:
            continue
        skipping = None

        # A target almost directly under the nozzle has a pan angle that
        # is real but meaningless: a couple of millimetres across the
        # nozzle flips it by 180 degrees, while the spray lands in
        # essentially the same place either way. Swinging the head round
        # to honour that is wasted travel and wasted accuracy, so when
        # the shot is near vertical the pan is left where it is.
        pan_want = float(r["pan_deg"])
        if (last_pan is not None and abs(pan_want - last_pan) > 120.0
                and float(r["tilt_deg"]) > NEAR_VERTICAL_DEG):
            print(f"      pan flip of {abs(pan_want - last_pan):.0f} deg "
                  f"ignored - spraying near vertical")
            pan_want = None
        else:
            last_pan = pan_want

        # Tilt first, and on its own. Stepping is a tight pulse loop that
        # cannot share the clock with a timed DC move, and the stepper is
        # exact, so there is nothing to gain from overlapping them.
        # With the rail off the head stays put and only the aiming axes
        # move. Shots then land wherever the head happens to be, which is
        # wrong on purpose - it is for watching pan and tilt behave, not
        # for spraying a bed.
        took = hw.rail_to(float(r["rail_cm"])) if use_rail else 0.0
        if pan_want is not None:
            took += hw.turn_to("pan", pan_want, max_turn)
        took += hw.turn_to("tilt", float(r["tilt_deg"]))

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


def test_motor(hw, axis="pan", seconds=2.0):
    """
    Run one axis at a few power levels so the difference can be seen.

    If all of these look the same speed, PWM is not reaching the motor
    and the duty cycle is doing nothing - which is a different fault
    from the duty being set wrong, and wants finding before any more
    numbers are tuned.
    """
    if axis not in PINS:
        sys.exit(f"No axis {axis!r}. Try: {', '.join(PINS)}")

    print("")
    print(f"  Testing {axis} on pins {PINS[axis]}")
    print(f"  Each step runs {seconds:.0f}s forward. Watch the speed.")
    if hw.dry:
        print("  (dry run - nothing will actually turn)")
    print("")
    for duty in (100.0, 60.0, 30.0, 15.0):
        print(f"    {duty:5.0f}% ...", end="", flush=True)
        hw.drive(axis, 1, duty=duty)
        wait(hw, seconds)
        hw.stop(axis)
        wait(hw, 0.4)
        print(" stopped")

    print("")
    print("  All four the same speed means PWM is not working:")
    print("    - rpi-lgpio must be installed, not plain RPi.GPIO")
    print("      sudo apt install -y python3-rpi-lgpio")
    print("    - and the code must be the pulled version. Check with:")
    print("      git -C ~/Automated-Weeder log --oneline -1")
    print("")
    print("  Speeds that differ but are all too fast means DUTY wants")
    print("  lowering - try --duty 15.")


def time_axis(hw, axis, seconds=1.0):
    """
    Drive one axis for a known time so its real speed can be worked out.

    Everything downstream is a duration, and a duration is only as good
    as the degrees-per-second it was divided by. That figure is still a
    full-power stopwatch number scaled by the duty cycle, which is a
    guess - and the overshoot on the machine says it is a bad one.
    """
    if axis not in PINS:
        sys.exit(f"No axis {axis!r}. Try: {', '.join(PINS)}")

    print("")
    print(f"  Driving {axis} for {seconds:.2f}s at {DUTY:.0f}% power.")
    print(f"  The code currently believes that is "
          f"{seconds * SPEED_DEG[axis]:.1f} degrees.")
    print("  Mark where it points now, then press Enter.")
    try:
        input("  > ")
    except EOFError:
        pass

    hw.drive(axis, 1)
    wait(hw, seconds)
    hw.stop(axis)

    print("")
    print("  Measure how far it actually turned, then set")
    print(f"  SPEED_DEG_FULL['{axis}'] near the top of this file to:")
    print("")
    print(f"      measured_degrees / {seconds:.2f} * 100 / {DUTY:.0f}")
    print("")
    print("  That is the figure at FULL power - the code scales it down")
    print("  by the duty itself.")


def rail_sweep(hw, cm=10.0):
    """
    Drive the rail a stated distance so the real steps-per-cm can be
    measured. Mark where the head is, run it, measure how far it went:

        RAIL_STEPS_PER_CM  *  (cm asked for / cm actually moved)
    """
    steps = int(round(cm * RAIL_STEPS_PER_CM))
    print(f"  Rail sweep: asking for {cm:.1f} cm at "
          f"{RAIL_STEPS_PER_CM:.0f} steps/cm = {steps:,} steps")
    print("  Mark where the head is now, then press Enter.")
    try:
        input("  > ")
    except EOFError:
        pass
    at = hw.rail_cm
    hw.rail_to(at + cm)
    print("")
    print("  Measure how far it actually moved, then set")
    print("  RAIL_STEPS_PER_CM near the top of this file to:")
    print("")
    print(f"      {RAIL_STEPS_PER_CM:.0f} * ({cm:.1f} / measured)")
    print("")
    print("  If it buzzed without moving, the step rate is too high to")
    print("  start from standstill: raise RAIL_START_DELAY.")


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
    ap.add_argument("--no-rail", action="store_true",
                    help="leave the rail stepper alone and move only pan "
                         "and tilt")
    ap.add_argument("--pick", action="store_true",
                    help="choose which boxes to run, and confirm each one "
                         "before it moves")
    ap.add_argument("--max-turn", type=float, default=MAX_TURN_S,
                    help=f"longest single pan or tilt drive, seconds "
                         f"(default {MAX_TURN_S})")
    ap.add_argument("--test-motor", metavar="AXIS", default=None,
                    help="run one axis at 100/60/30/15%% power to check "
                         "that PWM is working")
    ap.add_argument("--time-axis", metavar="AXIS", default=None,
                    help="drive one axis for a known time to measure its "
                         "real speed")
    ap.add_argument("--drive-seconds", type=float, default=1.0,
                    help="how long --time-axis drives for (default 1.0)")
    ap.add_argument("--rail-sweep", type=float, nargs="?", const=10.0,
                    default=None, metavar="CM",
                    help="drive the rail this far to measure "
                         "RAIL_STEPS_PER_CM (default 10)")
    args = ap.parse_args()

    hw = Hardware(dry=args.dry_run, real_time=args.real_time)

    try:
        if args.test_motor:
            test_motor(hw, args.test_motor)
            return

        if args.time_axis:
            time_axis(hw, args.time_axis, args.drive_seconds)
            return

        if args.rail_sweep is not None:
            rail_sweep(hw, args.rail_sweep)
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
            max_turn=args.max_turn, confirm=args.pick,
            use_rail=not args.no_rail)

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
