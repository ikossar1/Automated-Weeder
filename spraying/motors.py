"""
Turn aim() output into motor commands.

THE HANDOFF
    aim() speaks geometry: centimetres along the rail, degrees of pan,
    degrees of tilt. Motors speak steps. This file is the only place those
    two meet, so when a gear ratio changes or an axis turns out to run
    backwards, there is exactly one file to edit.

    Nothing here knows how to drive a motor. It produces target positions;
    the motor code decides how to get there.

THE THREE AXES
    rail    linear, along x        - carries the whole head
    pan     rotates about z (up)   - swings the aim left and right
    tilt    rotates about x        - raises and lowers the aim

    ORDERING MATTERS, AND THIS FILE ASSUMES ONE. It takes the tilt stage to
    be mounted ON the pan stage, so tilt pivots in the nozzle's own frame
    after pan has turned it. That is how nearly every pan-tilt head is
    built, and it is what makes pan and tilt independent: pan picks the
    direction, tilt picks the distance, neither disturbs the other.

    If your tilt axis is instead fixed to the world - the barrel pivoting
    about a rail-parallel axis that does not turn with pan - then the two
    are coupled and the angles need a different conversion. Worth checking
    against the actual bracket before cutting metal, because the mechanical
    difference is small and the maths difference is not.

UNITS AND SIGNS
    Every axis gets a steps_per_unit, a travel limit, and an invert flag.
    Those four numbers absorb gear ratios, microstepping, lead screw pitch
    and a motor wired backwards - all the things that are wrong on the
    first build and are then wrong forever if they are scattered through
    the code.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib


class Axis:
    """
    One motor, and how its units map to the real world.

    name            what it is called in the command stream
    steps_per_unit  motor steps per cm (linear) or per degree (rotary)
    lo, hi          travel limits in real units; commands outside are refused
    invert          True if positive steps move the opposite way to
                    positive units
    home            the real-world value the axis sits at after homing, so
                    step 0 means something definite
    """

    def __init__(self, name, steps_per_unit, lo, hi, invert=False, home=0.0):
        self.name = name
        self.steps_per_unit = float(steps_per_unit)
        self.lo = float(lo)
        self.hi = float(hi)
        self.invert = bool(invert)
        self.home = float(home)

    def steps(self, value):
        """Real units -> motor steps. None if the value is out of travel."""
        if not (self.lo - 1e-9 <= value <= self.hi + 1e-9):
            return None
        s = (value - self.home) * self.steps_per_unit
        return int(round(-s if self.invert else s))

    def value(self, steps):
        """Motor steps -> real units. The inverse, for checking."""
        s = -steps if self.invert else steps
        return self.home + s / self.steps_per_unit

    def __repr__(self):
        return (f"Axis({self.name}, {self.steps_per_unit:g} steps/unit, "
                f"{self.lo:g}..{self.hi:g}{', inverted' if self.invert else ''})")


class Head:
    """The three axes together, plus the valve."""

    def __init__(self, rail, pan, tilt):
        self.rail = rail
        self.pan = pan
        self.tilt = tilt

    def command(self, rail_cm, pan_deg, tilt_deg, spray=False):
        """
        One target position for all three motors.

        Returns a dict ready to serialise. `ok` is False when any axis
        cannot reach its target, and `refused` names which - so the caller
        can skip that step instead of sending a command that silently
        clamps.
        """
        parts = {
            "rail": (self.rail, rail_cm),
            "pan": (self.pan, pan_deg),
            "tilt": (self.tilt, tilt_deg),
        }

        steps, refused = {}, []
        for key, (axis, want) in parts.items():
            got = axis.steps(want)
            if got is None:
                refused.append(f"{key} {want:.1f} outside {axis.lo:g}..{axis.hi:g}")
            steps[key] = got

        return {
            "rail_cm": round(rail_cm, 2),
            "pan_deg": round(pan_deg, 2),
            "tilt_deg": round(tilt_deg, 2),
            "rail_steps": steps["rail"],
            "pan_steps": steps["pan"],
            "tilt_steps": steps["tilt"],
            "spray": 1 if spray else 0,
            "ok": not refused,
            "refused": "; ".join(refused),
        }

    def from_shot(self, shot, rail_cm, spray=None):
        """
        Convert an aim.Shot straight into a command.

        If `spray` is not given it follows shot.ok - a shot that cannot be
        taken is not one to open the valve on.
        """
        return self.command(rail_cm, shot.pan_deg, shot.tilt_deg,
                            shot.ok if spray is None else spray)

    def check_round_trip(self, rail_cm, pan_deg, tilt_deg):
        """
        Steps back to units, to catch a wrong steps_per_unit.

        Rounding to whole steps loses a little, and how much tells you your
        resolution: if one step is 0.2 degrees you cannot aim finer than
        that, however exact aim() was.
        """
        c = self.command(rail_cm, pan_deg, tilt_deg)
        if not c["ok"]:
            return None
        return {
            "rail_cm": self.rail.value(c["rail_steps"]) - rail_cm,
            "pan_deg": self.pan.value(c["pan_steps"]) - pan_deg,
            "tilt_deg": self.tilt.value(c["tilt_steps"]) - tilt_deg,
        }

    def resolution(self):
        """Ground movement of one step on each axis, in its own units."""
        return {
            "rail_cm_per_step": 1.0 / self.rail.steps_per_unit,
            "pan_deg_per_step": 1.0 / self.pan.steps_per_unit,
            "tilt_deg_per_step": 1.0 / self.tilt.steps_per_unit,
        }


# A starting point. Every number here is a guess until measured on the
# actual machine - which is the point of keeping them in one place.
DEFAULT_HEAD = Head(
    # 200 steps/rev motor, 16x microstepping, 8 mm per rev lead screw
    # -> 3200 steps per 8 mm -> 400 steps per mm -> 4000 per cm
    rail=Axis("rail", steps_per_unit=4000.0, lo=0.0, hi=300.0),
    # 200 steps/rev, 16x microstep, 5:1 gearbox -> 16000 steps/rev -> 44.4/deg
    pan=Axis("pan", steps_per_unit=44.444, lo=-90.0, hi=90.0),
    tilt=Axis("tilt", steps_per_unit=44.444, lo=5.0, hi=85.0),
)


def export_plan(plan_json, head=None, out_csv="motor_commands.csv",
                relative=False):
    """
    Turn a simulate.py plan into a motor command stream.

    One row per move. The motor code reads this and does not need to know
    anything about plants, cameras or geometry.

    relative=False gives absolute target positions, for a controller that
    tracks where it is. relative=True gives the change since the previous
    command, which is what a step/dir driver wants: the number is the pulse
    count and the sign is the direction pin, so the motor code can drive
    straight off these columns.
    """
    head = head or DEFAULT_HEAD
    plan = json.loads(pathlib.Path(plan_json).read_text())

    fields = ["i", "rail_cm", "pan_deg", "tilt_deg",
              "rail_steps", "pan_steps", "tilt_steps", "spray"]
    if relative:
        fields = ["i", "rail_cm", "pan_deg", "tilt_deg",
                  "d_rail", "d_pan", "d_tilt", "spray"]

    rows, refused, prev = [], 0, None
    for i, st in enumerate(plan["steps"]):
        _stop, rail_cm, pan, tilt, _lx, _ly, spraying, _blk = st
        c = head.command(rail_cm, pan, tilt, bool(spraying))
        if not c["ok"]:
            refused += 1
            continue

        if relative:
            now = (c["rail_steps"], c["pan_steps"], c["tilt_steps"])
            base = prev if prev is not None else now
            rows.append({
                "i": i, "rail_cm": c["rail_cm"], "pan_deg": c["pan_deg"],
                "tilt_deg": c["tilt_deg"],
                "d_rail": now[0] - base[0],
                "d_pan": now[1] - base[1],
                "d_tilt": now[2] - base[2],
                "spray": c["spray"],
            })
            prev = now
        else:
            rows.append({"i": i, **{k: c[k] for k in fields if k != "i"}})

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    return out_csv, len(rows), refused


# ----------------------------------------------------------------------------
# DC motors
# ----------------------------------------------------------------------------

class DCAxis(Axis):
    """
    A DC motor instead of a stepper.

    THE DIFFERENCE THAT MATTERS
        A stepper is told "go 1,333 steps" and goes there. A DC motor is
        told "turn on" and keeps turning until something stops it. It has no
        idea where it is.

        So position control needs one of two things:

        WITH AN ENCODER (do this)
            The encoder counts shaft movement, your motor code runs a PID
            loop driving the count toward the target, and `counts_per_unit`
            below replaces `steps_per_unit`. Everything else in this file is
            unchanged - the target numbers mean encoder counts instead of
            steps. This is the only way to get repeatable aiming.

        WITHOUT AN ENCODER (open loop)
            You drive the motor for a measured length of time and hope.
            run_time() below estimates that from a measured speed, but it is
            a guess: the same duration travels further on a warm motor, a
            fresh battery, or downhill. Fine for "move roughly there",
            not for aiming.

        Pan and tilt decide where mulch lands, so they want encoders. The
        rail is more forgiving - a centimetre out along the rail just shifts
        the pattern, while a degree out at the nozzle moves the landing spot
        several centimetres.

    counts_per_unit   encoder counts per cm or per degree, AFTER gearing
    units_per_second  how fast it moves at full drive, measured not assumed
    min_drive         the drive level below which it stalls instead of
                      turning - every DC motor has one, and it is why small
                      corrections need a minimum kick
    """

    def __init__(self, name, counts_per_unit, lo, hi, units_per_second,
                 invert=False, home=0.0, min_drive=0.25):
        super().__init__(name, counts_per_unit, lo, hi, invert, home)
        self.units_per_second = float(units_per_second)
        self.min_drive = float(min_drive)

    @property
    def counts_per_unit(self):
        return self.steps_per_unit

    def counts(self, value):
        """Real units -> encoder counts. Same maths, clearer name."""
        return self.steps(value)

    def run_time(self, from_value, to_value, drive=1.0):
        """
        Seconds of drive to cover this distance open loop.

        Only meaningful if you measured units_per_second on the real
        machine, under load, at the drive level you will actually use.
        """
        if self.units_per_second <= 0 or drive <= 0:
            return 0.0
        return abs(to_value - from_value) / (self.units_per_second * drive)

    def direction(self, from_value, to_value):
        """+1, -1 or 0 - which way to drive the H-bridge."""
        d = to_value - from_value
        if abs(d) < 1e-9:
            return 0
        fwd = 1 if d > 0 else -1
        return -fwd if self.invert else fwd


class DCHead(Head):
    """Three DC motors. Emits encoder targets and open-loop fallbacks."""

    def command(self, rail_cm, pan_deg, tilt_deg, spray=False,
                prev=None):
        c = super().command(rail_cm, pan_deg, tilt_deg, spray)

        # Rename for honesty: with DC motors these are encoder counts.
        c["rail_counts"] = c.pop("rail_steps")
        c["pan_counts"] = c.pop("pan_steps")
        c["tilt_counts"] = c.pop("tilt_steps")

        if prev is not None:
            pr, pp, pt = prev
            for key, axis, was, now in (("rail", self.rail, pr, rail_cm),
                                        ("pan", self.pan, pp, pan_deg),
                                        ("tilt", self.tilt, pt, tilt_deg)):
                c[f"{key}_dir"] = axis.direction(was, now)
                c[f"{key}_secs"] = round(axis.run_time(was, now), 4)
        return c


DEFAULT_DC_HEAD = DCHead(
    # Measure every one of these. They are placeholders.
    rail=DCAxis("rail", counts_per_unit=120.0, lo=0.0, hi=300.0,
                units_per_second=8.0),
    pan=DCAxis("pan", counts_per_unit=12.0, lo=-90.0, hi=90.0,
               units_per_second=45.0),
    tilt=DCAxis("tilt", counts_per_unit=12.0, lo=5.0, hi=85.0,
                units_per_second=45.0),
)


def export_dc(plan_json, head=None, out_csv="dc_commands.csv"):
    """
    Command stream for DC motors.

    Gives encoder targets for closed loop, plus a direction and a duration
    for each axis so an open-loop build has something to work with.
    """
    head = head or DEFAULT_DC_HEAD
    plan = json.loads(pathlib.Path(plan_json).read_text())

    fields = ["i", "rail_cm", "pan_deg", "tilt_deg",
              "rail_counts", "pan_counts", "tilt_counts",
              "rail_dir", "pan_dir", "tilt_dir",
              "rail_secs", "pan_secs", "tilt_secs", "spray"]

    rows, refused, prev = [], 0, None
    for i, st in enumerate(plan["steps"]):
        _stop, rail_cm, pan, tilt, _lx, _ly, spraying, _blk = st
        c = head.command(rail_cm, pan, tilt, bool(spraying),
                         prev=prev or (rail_cm, pan, tilt))
        if not c["ok"]:
            refused += 1
            continue
        rows.append({k: c.get(k, 0) for k in fields if k != "i"} | {"i": i})
        prev = (rail_cm, pan, tilt)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    return out_csv, len(rows), refused


if __name__ == "__main__":
    import sys

    head = DEFAULT_HEAD
    print("Axes:")
    for a in (head.rail, head.pan, head.tilt):
        print(f"  {a}")

    print("\nResolution - the finest move each axis can make:")
    for k, v in head.resolution().items():
        print(f"  {k:22s} {v:.5f}")

    print("\nA few commands:")
    for rail, pan, tilt in [(0, 0, 45), (150.0, -30.0, 20.0),
                            (300.0, 70.0, 80.0), (310.0, 0.0, 45.0)]:
        c = head.command(rail, pan, tilt, spray=True)
        if c["ok"]:
            print(f"  rail {rail:6.1f}cm pan {pan:+6.1f} tilt {tilt:5.1f}"
                  f"  ->  {c['rail_steps']:>9d} {c['pan_steps']:>7d} "
                  f"{c['tilt_steps']:>6d}")
        else:
            print(f"  rail {rail:6.1f}cm pan {pan:+6.1f} tilt {tilt:5.1f}"
                  f"  ->  REFUSED: {c['refused']}")

    print("\nRound trip - how much is lost rounding to whole steps:")
    err = head.check_round_trip(123.456, -17.3, 42.7)
    for k, v in err.items():
        print(f"  {k:10s} {v:+.6f}")

    if len(sys.argv) > 1:
        if "--dc" in sys.argv:
            out, n, bad = export_dc(sys.argv[1], DEFAULT_DC_HEAD)
            print(f"\nWrote {out}: {n:,} commands for DC motors"
                  + (f", {bad:,} refused as out of travel" if bad else ""))
            raise SystemExit(0)
        rel = "--relative" in sys.argv
        name = "motor_steps.csv" if rel else "motor_commands.csv"
        out, n, bad = export_plan(sys.argv[1], head, name, relative=rel)
        print(f"\nWrote {out}: {n:,} commands"
              + (" as step deltas" if rel else " as absolute positions")
              + (f", {bad:,} refused as out of travel" if bad else ""))


