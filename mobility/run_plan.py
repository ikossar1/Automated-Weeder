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
INVERT = {"rail": False, "pan": False, "tilt": False}

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
MAX_DRIVE_DEG = {"pan": 10.0, "tilt": 10.0}

# Above this tilt the shot is near enough vertical that pan stops
# mattering: the nozzle is looking almost straight down, so swinging it
# left or right moves the landing point by very little.
NEAR_VERTICAL_DEG = 75.0

# How many capped drives one move may be split into. Enough to cross the
# full range of either axis, and a bound on how long a single command can
# keep a motor running if a speed is badly wrong.
MAX_DRIVE_STEPS = 24
GRACE = 1.2

MAX_DRIVE_S = {a: round(MAX_DRIVE_DEG[a] / SPEED_DEG[a] * GRACE, 2)
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

        # The valve closes BEFORE a move that is not meant to be spraying,
        # and opens only once the head is already where it should be.
        if valve_is_open and not want_valve:
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

        if args.rail_sweep is not None:
            rail_sweep(hw, args.rail_sweep)
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


if __name__ == "__main__":
    main()
