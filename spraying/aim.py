"""
Aiming the mulch sprayer.

An airbrush throws in a straight line, so this is geometry.py run backwards.
There, a pixel is a ray from the lens that you intersect with the soil.
Here, a spray direction is a ray from the nozzle that you intersect with the
same soil. Same flat-ground assumption, same trigonometry, opposite
direction.

That symmetry is also how it gets tested. aim() and spray_point() are exact
inverses, so composing them has to return what you started with. Any error
is a bug, not a tolerance - see verify_aim.py.

FRAMES AND ANGLES
    Everything is in the garden frame the map already uses: x and y in
    centimetres from the corner stake, z up, soil at z = 0.

    heading   the direction pan = 0 points, measured counter-clockwise from
              the garden's +x axis. A rail along the near edge pointing into
              the bed has heading 90.
    pan       left/right from that heading. Positive is counter-clockwise.
    tilt      down from horizontal. 0 is level, 90 is straight down.
              Bigger tilt means a shorter shot.

WHAT THIS DELIBERATELY DOES NOT MODEL
    Gravity droop. Over the 10-30 cm an airbrush actually throws, the drop
    is small compared with the spray cone's own width, so a straight ray is
    the right model. If you later swap in something that lobs, this file is
    where that changes - and the test in verify_aim.py would catch the
    difference immediately.
"""
from __future__ import annotations

import math

import numpy as np


class Nozzle:
    """
    Where the sprayer is and what it can do.

    x_cm, y_cm     position along the rail, in garden coordinates
    height_cm      nozzle height above the soil
    heading_deg    direction pan = 0 points, CCW from +x
    pan_limit_deg  how far either side of heading it can swing
    tilt_min_deg   shallowest shot it can take (sets the FARTHEST reach)
    tilt_max_deg   steepest shot (sets the NEAREST reach)
    max_range_cm   how far the airbrush actually throws usefully. This is
                   the number to measure on the bench first - everything
                   downstream depends on it and no datasheet will tell you.
    """

    def __init__(self, x_cm, y_cm, height_cm, heading_deg=90.0,
                 pan_limit_deg=75.0, tilt_min_deg=12.0, tilt_max_deg=80.0,
                 max_range_cm=35.0, offset_along_cm=0.0, offset_across_cm=0.0):
        # x_cm, y_cm is where the HEAD is - the thing the rail moves and the
        # thing the camera is mounted to. The nozzle sits a little away from
        # it, and that offset is applied here so every angle downstream is
        # measured from the nozzle rather than the camera.
        #
        # 36 mm sounds negligible. It is not: at a 15 cm shot it is a 1.4
        # degree aiming error, and the mulch lands 3.6 cm from where the map
        # said the bare patch was. The map is built in garden coordinates
        # from the plot corners, so it does not care where the camera was -
        # but the sprayer does.
        self.offset_along_cm = float(offset_along_cm)
        self.offset_across_cm = float(offset_across_cm)

        h = math.radians(heading_deg)
        self.head_x_cm = float(x_cm)
        self.head_y_cm = float(y_cm)
        self.x_cm = float(x_cm) + (offset_along_cm * math.cos(h)
                                   - offset_across_cm * math.sin(h))
        self.y_cm = float(y_cm) + (offset_along_cm * math.sin(h)
                                   + offset_across_cm * math.cos(h))
        self.height_cm = float(height_cm)
        self.heading_deg = float(heading_deg)
        self.pan_limit_deg = float(pan_limit_deg)
        self.tilt_min_deg = float(tilt_min_deg)
        self.tilt_max_deg = float(tilt_max_deg)
        self.max_range_cm = float(max_range_cm)

    def at(self, x_cm=None, y_cm=None):
        """The same nozzle with the HEAD moved. The offset travels with it."""
        return Nozzle(self.head_x_cm if x_cm is None else x_cm,
                      self.head_y_cm if y_cm is None else y_cm,
                      self.height_cm, self.heading_deg, self.pan_limit_deg,
                      self.tilt_min_deg, self.tilt_max_deg, self.max_range_cm,
                      self.offset_along_cm, self.offset_across_cm)

    def reach(self):
        """(nearest, farthest) ground distance the tilt limits allow, cm."""
        near = self.height_cm / math.tan(math.radians(self.tilt_max_deg))
        far = self.height_cm / math.tan(math.radians(self.tilt_min_deg))
        return near, min(far, self.max_range_cm)

    def __repr__(self):
        n, f = self.reach()
        return (f"Nozzle(({self.x_cm:.1f}, {self.y_cm:.1f}) cm, "
                f"h={self.height_cm:.1f}, heading={self.heading_deg:.0f}deg, "
                f"reach {n:.1f}-{f:.1f} cm)")


class Shot:
    """One aiming solution. Check `ok` before trusting the angles."""

    __slots__ = ("pan_deg", "tilt_deg", "ground_cm", "slant_cm",
                 "ok", "reason", "target")

    def __init__(self, pan_deg, tilt_deg, ground_cm, slant_cm,
                 ok, reason, target):
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.ground_cm = ground_cm
        self.slant_cm = slant_cm
        self.ok = ok
        self.reason = reason
        self.target = target

    def __repr__(self):
        state = "ok" if self.ok else f"NO - {self.reason}"
        return (f"Shot(pan {self.pan_deg:+6.1f}deg, tilt {self.tilt_deg:5.1f}deg, "
                f"{self.ground_cm:5.1f} cm out, {state})")


def _wrap180(deg):
    """Fold an angle into -180..180, so pan limits compare sensibly."""
    return (deg + 180.0) % 360.0 - 180.0


def aim(nozzle, target_xy):
    """
    What pan and tilt put the spray on this ground point.

    Always returns a Shot. When the point cannot be reached, the angles are
    still the ones that point at it - `ok` is False and `reason` says why.
    That is deliberate: a planner wants to know whether a target is out of
    range or out of pan, and the angles help you see which way it missed.
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx, dy = tx - nozzle.x_cm, ty - nozzle.y_cm

    ground = math.hypot(dx, dy)
    slant = math.hypot(ground, nozzle.height_cm)

    # Straight down: pan is undefined, so keep the current heading.
    if ground < 1e-9:
        return Shot(0.0, 90.0, 0.0, nozzle.height_cm,
                    False, "directly under the nozzle", (tx, ty))

    pan = _wrap180(math.degrees(math.atan2(dy, dx)) - nozzle.heading_deg)
    tilt = math.degrees(math.atan2(nozzle.height_cm, ground))

    ok, reason = True, ""
    if abs(pan) > nozzle.pan_limit_deg:
        ok, reason = False, f"pan {pan:+.1f} beyond +/-{nozzle.pan_limit_deg:.0f}"
    elif tilt > nozzle.tilt_max_deg:
        ok, reason = False, f"too close - needs tilt {tilt:.1f}"
    elif tilt < nozzle.tilt_min_deg:
        ok, reason = False, f"too far - needs tilt {tilt:.1f}"
    elif ground > nozzle.max_range_cm:
        ok, reason = False, f"{ground:.1f} cm is past the {nozzle.max_range_cm:.0f} cm throw"

    return Shot(pan, tilt, ground, slant, ok, reason, (tx, ty))


def spray_point(nozzle, pan_deg, tilt_deg):
    """
    Where the spray lands for a given pan and tilt. The inverse of aim().

    Returns None for a shot level or above horizontal, which never comes
    down onto the soil.
    """
    if tilt_deg <= 0.0:
        return None

    ground = nozzle.height_cm / math.tan(math.radians(tilt_deg))
    world = math.radians(nozzle.heading_deg + pan_deg)

    return (nozzle.x_cm + ground * math.cos(world),
            nozzle.y_cm + ground * math.sin(world))


# ----------------------------------------------------------------------------
# Line of sight
# ----------------------------------------------------------------------------

def _circle_blocks(ax, ay, bx, by, cx, cy, radius):
    """
    Does the circle at (cx, cy) sit on the segment A->B?

    Only counts if the nearest approach happens BETWEEN the endpoints. A
    plant behind the nozzle blocks nothing, and one past the target is not
    in the way either - it is simply further along the same line.
    """
    vx, vy = bx - ax, by - ay
    seg2 = vx * vx + vy * vy
    if seg2 < 1e-12:
        return False

    t = ((cx - ax) * vx + (cy - ay) * vy) / seg2
    if t <= 0.0 or t >= 1.0:
        return False

    nearest = math.hypot(ax + t * vx - cx, ay + t * vy - cy)
    return nearest < radius


def blocked_by(nozzle, target_xy, plants, clearance_cm=0.0, skip_ids=()):
    """
    The first plant standing between the nozzle and the target, or None.

    Works in plan view: a plant is treated as a circle of its own measured
    footprint radius, grown by `clearance_cm`, and anything crossing that
    circle is blocked.

    THIS IS DELIBERATELY PESSIMISTIC. It ignores plant height, so it assumes
    every plant blocks regardless of how low it is. We do not measure height
    - the map only knows footprint - and over-spraying a crop is worse than
    leaving a patch of soil bare, so the assumption errs the safe way.

    If you later measure heights, this is the one function that changes:
    a plant only blocks when the ray is below its top as it passes over.
    """
    nx, ny = nozzle.x_cm, nozzle.y_cm
    tx, ty = float(target_xy[0]), float(target_xy[1])

    for p in plants:
        if p.id in skip_ids:
            continue
        radius = footprint_radius(p) + clearance_cm
        if radius <= 0:
            continue
        if _circle_blocks(nx, ny, tx, ty, p.x_cm, p.y_cm, radius):
            return p
    return None


def footprint_radius(plant):
    """A plant's radius in cm, from the footprint area the map measured."""
    if getattr(plant, "size_cm2", 0.0) > 0:
        return math.sqrt(plant.size_cm2 / math.pi)
    return 0.0


def on_plant(target_xy, plants, clearance_cm=0.0, skip_ids=()):
    """
    The plant the target lands on, or None.

    Separate from blocked_by on purpose. blocked_by asks whether anything
    stands in the PATH, and correctly ignores a plant sitting at the far
    end - that plant is the destination, not an obstruction. This asks the
    other question: is the destination itself a plant?

    Both are needed. Without this one a sprayer aimed straight at a lettuce
    reports a clear shot, because nothing is between it and the lettuce.
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    for p in plants:
        if p.id in skip_ids:
            continue
        if math.hypot(p.x_cm - tx, p.y_cm - ty) <= footprint_radius(p) + clearance_cm:
            return p
    return None


def solve(nozzle, target_xy, plants=(), clearance_cm=1.0, skip_ids=(),
          allow_on_plant=False):
    """
    aim(), plus both plant checks - the one to call from a planner.

    A Shot that comes back ok is mechanically reachable, has a clear path,
    and is not pointed at a plant.

    allow_on_plant=True skips the last of those, for the case where you
    actually do want to hit the plant.
    """
    shot = aim(nozzle, target_xy)
    if not shot.ok:
        return shot

    if not allow_on_plant:
        sitting = on_plant(target_xy, plants, clearance_cm, skip_ids)
        if sitting is not None:
            shot.ok = False
            shot.reason = f"lands on plant #{sitting.id}"
            return shot

    hit = blocked_by(nozzle, target_xy, plants, clearance_cm, skip_ids)
    if hit is not None:
        shot.ok = False
        shot.reason = f"plant #{hit.id} in the way"
    return shot


def reachable_mask(nozzle, points, plants=(), clearance_cm=1.0):
    """Boolean array - which of these ground points this nozzle can spray."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return np.array([solve(nozzle, p, plants, clearance_cm).ok for p in pts])
