"""
Pixel -> position on the soil.

Turns a pixel in a camera image into a real position on the garden floor,
measured in centimetres from a stake you put in one corner.

THE IDEA IN ONE PARAGRAPH
    A pixel is not a point in space. It is a *direction* - "something is
    somewhere along this line." That is why a single photo looks like it has
    lost a dimension. But you already know one thing about every plant: its
    base sits on the soil. So follow the direction out from the lens until it
    crosses the soil, and stop. There is exactly one such point, so there is
    exactly one answer. No second camera, no depth sensor, no neural network.

    The whole method rests on the soil being flat. Where it bulges or dips,
    the answer drifts - see ACCURACY at the bottom.

THE FOUR FRAMES
    image    pixels. u across, v down, origin at the top-left corner.
    camera   OpenCV convention. x right, y down, z straight out of the lens.
    robot    origin on the soil directly under the robot's centre.
             x forward, y left, z up. The soil is z = 0.
    garden   origin at a stake in one corner of the bed. x, y along the
             ground. This is the frame the map is built in, and the only
             one whose numbers you will ever report.

    Every function here moves a point one step along that chain.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

try:
    import cv2
except ImportError:  # geometry still works, just without lens-distortion correction
    cv2 = None


# ----------------------------------------------------------------------------
# 1. The lens
# ----------------------------------------------------------------------------

class Intrinsics:
    """
    What the lens does to a ray of light.

    Two things: a focal length and centre point (the matrix K), and how much
    the glass bends straight lines near the edges of the frame (dist).
    """

    def __init__(self, K, dist=None, size=None, approximate=False):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64).ravel()
        self.size = tuple(size) if size else None
        self.approximate = approximate

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        """
        Rough intrinsics worked out from the lens's horizontal field of view.

        Use this to get the pipeline running TODAY, before you have shot a
        calibration board. It assumes square pixels, a perfectly centred
        optical axis and zero distortion - none of which is quite true, and
        the last one is badly untrue near the frame edges.

        Expect several centimetres of error. Fine for checking the plumbing;
        not fine for a number you put in the report. Replace it with
        calibrate.py output before you quote any accuracy figure.
        """
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        K = [[f, 0.0, width / 2.0],
             [0.0, f, height / 2.0],
             [0.0, 0.0, 1.0]]
        return cls(K, None, (width, height), approximate=True)

    @classmethod
    def load(cls, path="calibration.json"):
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["K"], d.get("dist"), d.get("size"),
                   approximate=d.get("approximate", False))

    def save(self, path="calibration.json"):
        pathlib.Path(path).write_text(json.dumps({
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "size": list(self.size) if self.size else None,
            "approximate": self.approximate,
        }, indent=2))
        return path

    def rays(self, uv):
        """
        Pixels -> ray directions in the CAMERA frame.

        Undoes the lens distortion, then undoes K. The result is a direction
        per pixel, with z fixed at 1 - length is meaningless, only heading.
        """
        pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)

        if cv2 is not None:
            flat = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        else:
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            p = pts.reshape(-1, 2)
            flat = np.column_stack(((p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy))

        return np.column_stack([flat, np.ones(len(flat))])

    def __repr__(self):
        tag = " APPROXIMATE - not calibrated" if self.approximate else ""
        return (f"Intrinsics(fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, "
                f"cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}{tag})")


# ----------------------------------------------------------------------------
# 2. Where the camera is bolted, and where it is looking
# ----------------------------------------------------------------------------

class CameraMount:
    """
    The camera's position and aim, relative to the robot.

    height_cm   lens height above the soil. Measure this carefully - it
                scales every distance you compute, so a 5% error here is a
                5% error in every answer.
    tilt_deg    downward tilt from horizontal.
                  90 = pointing straight down at the soil
                  45 = angled forward
                   0 = horizontal (useless - the ray never hits the ground)
    pan_deg     swing left or right from robot-forward. Positive = left.
    forward_cm  lens offset ahead of the robot's centre (negative = behind)
    left_cm     lens offset to the robot's left (negative = right)

    tilt and pan are arguments rather than fixed values because yours move.
    Read them off your servos each frame and pass them in.
    """

    # Camera at rest: looking along robot-forward, image-right is robot-right,
    # image-down is robot-down. Columns are where each camera axis ends up.
    _REST = np.array([[0.0,  0.0, 1.0],    # cam z (forward) -> robot x (forward)
                      [-1.0, 0.0, 0.0],    # cam x (right)   -> robot -y (right)
                      [0.0, -1.0, 0.0]])   # cam y (down)    -> robot -z (down)

    def __init__(self, height_cm, tilt_deg, pan_deg=0.0,
                 forward_cm=0.0, left_cm=0.0):
        self.height_cm = float(height_cm)
        self.tilt_deg = float(tilt_deg)
        self.pan_deg = float(pan_deg)
        self.forward_cm = float(forward_cm)
        self.left_cm = float(left_cm)

    def at(self, tilt_deg=None, pan_deg=None):
        """A copy with the servos moved. Mount geometry stays put."""
        return CameraMount(
            self.height_cm,
            self.tilt_deg if tilt_deg is None else tilt_deg,
            self.pan_deg if pan_deg is None else pan_deg,
            self.forward_cm, self.left_cm)

    def rotation(self):
        """Camera frame -> robot frame."""
        t, p = math.radians(self.tilt_deg), math.radians(self.pan_deg)

        tilt_down = np.array([[math.cos(t), 0.0, math.sin(t)],
                              [0.0,         1.0, 0.0],
                              [-math.sin(t), 0.0, math.cos(t)]])

        swing = np.array([[math.cos(p), -math.sin(p), 0.0],
                          [math.sin(p),  math.cos(p), 0.0],
                          [0.0,          0.0,         1.0]])

        return swing @ tilt_down @ self._REST

    def position(self):
        """Lens position in the robot frame."""
        return np.array([self.forward_cm, self.left_cm, self.height_cm])


# ----------------------------------------------------------------------------
# 3. Where the robot is in the garden
# ----------------------------------------------------------------------------

class RobotPose:
    """
    The robot's own position in the garden, in centimetres from the stake.

    heading_deg is the direction the robot faces, counter-clockwise from the
    garden's +x axis. 0 = facing +x, 90 = facing +y.

    In a controlled-environment prototype this comes from your own motor
    commands - you are not estimating it, you are the one who set it. That
    is the single reason this whole approach works without SLAM.
    """

    def __init__(self, x_cm=0.0, y_cm=0.0, heading_deg=0.0):
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)
        self.heading_deg = float(heading_deg)

    def to_garden(self, xy_robot):
        """Robot-frame (x, y) -> garden-frame (x, y). A rotate and a shift."""
        xy = np.asarray(xy_robot, np.float64).reshape(-1, 2)
        h = math.radians(self.heading_deg)
        c, s = math.cos(h), math.sin(h)

        return np.column_stack([
            self.x_cm + xy[:, 0] * c - xy[:, 1] * s,
            self.y_cm + xy[:, 0] * s + xy[:, 1] * c,
        ])

    def __repr__(self):
        return (f"RobotPose(x={self.x_cm:.1f}cm, y={self.y_cm:.1f}cm, "
                f"heading={self.heading_deg:.1f}deg)")


# ----------------------------------------------------------------------------
# 4. The projector - the part you actually call
# ----------------------------------------------------------------------------

class GroundProjector:
    """
    Pixels in, garden coordinates out.

        proj = GroundProjector(intrinsics, mount)
        xy   = proj.to_garden([(812, 1430)], pose)      # -> [[x_cm, y_cm]]

    Returns NaN for any pixel whose ray does not hit the soil in front of
    the robot: sky, the horizon, or anything absurdly far away. Always check
    before you drive a tool to the answer.
    """

    def __init__(self, intrinsics, mount, max_range_cm=300.0):
        self.intrinsics = intrinsics
        self.mount = mount
        self.max_range_cm = float(max_range_cm)

    def to_robot(self, uv, mount=None):
        """Pixels -> (x, y) on the soil in the ROBOT frame. NaN where no hit."""
        mount = mount or self.mount

        rays = self.intrinsics.rays(uv) @ mount.rotation().T   # into robot frame
        lens = mount.position()

        out = np.full((len(rays), 2), np.nan)

        dz = rays[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            # Soil is z = 0, so  lens_z + scale * ray_z = 0.
            scale = -lens[2] / dz

        # Keep only rays aimed downward, landing ahead of us, within range.
        hit = (dz < -1e-9) & (scale > 0) & np.isfinite(scale)
        hit &= scale * np.linalg.norm(rays, axis=1) < self.max_range_cm

        if hit.any():
            s = scale[hit][:, None]
            out[hit, 0] = lens[0] + s[:, 0] * rays[hit, 0]
            out[hit, 1] = lens[1] + s[:, 0] * rays[hit, 1]

        return out

    def to_garden(self, uv, pose=None, mount=None):
        """Pixels -> (x, y) on the soil in the GARDEN frame. NaN where no hit."""
        xy = self.to_robot(uv, mount=mount)
        pose = pose or RobotPose()

        good = ~np.isnan(xy[:, 0])
        out = np.full_like(xy, np.nan)
        if good.any():
            out[good] = pose.to_garden(xy[good])
        return out

    def cm_per_pixel(self, uv, pose=None, mount=None):
        """
        How much ground one pixel covers at this point in the frame.

        This is your resolution limit, and it is NOT constant - a pixel near
        the top of a tilted frame covers far more soil than one at the
        bottom. Worth reporting: it bounds how precisely you can ever place
        the weeding tool, no matter how good the detector is.
        """
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        here = self.to_garden(uv, pose, mount)
        over = self.to_garden(uv + np.array([1.0, 0.0]), pose, mount)
        return np.linalg.norm(over - here, axis=1)


# ----------------------------------------------------------------------------

def load(calibration="calibration.json", **mount_kwargs):
    """
    Convenience: build a projector from a saved calibration file.

        proj = geometry.load(height_cm=35, tilt_deg=60)
    """
    return GroundProjector(Intrinsics.load(calibration), CameraMount(**mount_kwargs))


# ----------------------------------------------------------------------------
# ACCURACY - what actually limits this
#
# In rough order of how much damage each one does:
#
#   1. Camera height.  A direct multiplier on every distance. Measure to the
#      front face of the lens, not the board, and measure twice.
#   2. Tilt angle.  At a shallow tilt a 1-degree error throws the far end of
#      the frame off by centimetres. Steeper is more forgiving; straight down
#      is most forgiving of all.
#   3. Uneven soil.  A bump of height h reads as a position error of roughly
#      h / tan(tilt). At 45 degrees that is 1:1 - a 2 cm clod moves the
#      answer 2 cm. Steep tilt shrinks this too.
#   4. Using uncalibrated intrinsics.  from_fov() ignores lens distortion,
#      which is worst at the frame edges - exactly where your far-field
#      plants are.
#   5. Picking the wrong pixel.  Leaves overhang soil the roots are not in.
#      Take the BOTTOM of the mask, not its centre. detect.py does this.
#
# Points 2 and 3 both argue for mounting the camera steeply, close to
# straight down. The cost is a smaller footprint per frame, so you take more
# frames. For a prototype that is the right trade - accuracy is the thing
# you have to defend, and frame count is just time.
#
# Run validate.py to turn all of this into one measured number.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# 5. The no-camera route - four corners instead of height and tilt
# ----------------------------------------------------------------------------

class PlanarProjector:
    """
    Pixel -> ground, from four known reference points. No camera needed.

    WHEN TO USE THIS INSTEAD OF GroundProjector
        GroundProjector needs to know the camera: focal length, lens
        distortion, mount height, tilt angle. Measure all four and it works
        from any viewpoint.

        This one needs none of that. Instead you mark out a rectangle of known
        size on the ground - stakes at the corners of a 3 x 4 m plot, say -
        and tell it where those four corners landed in the image. From four
        point pairs it solves for the perspective directly.

        The trade: GroundProjector works on any photo once the camera is
        characterised. PlanarProjector works on any camera, but every photo
        needs its four corners identified.

        For a prototype on a measured plot, this is usually the easier road.
        You put four stakes in the ground and you are done - no checkerboard,
        no focal length, no tilt gauge, and it does not care if you swap
        cameras or change the zoom halfway through.

    WHAT IT CANNOT DO
        It assumes the ground inside those corners is flat, exactly as the
        other method does. It also silently absorbs lens distortion into the
        fit rather than correcting it, so a very wide or fisheye lens will
        bow the result - straight rows will come out slightly curved. For a
        normal phone or Pi camera lens the effect is small.

        Accuracy is best inside the quadrilateral and degrades fast outside
        it. Anything beyond your stakes is extrapolation; treat it as a hint,
        not a measurement.
    """

    def __init__(self, image_corners, plot_w_cm, plot_d_cm):
        """
        image_corners   four (u, v) pixels, in this order:
                            1. near-left   corner of the plot
                            2. near-right
                            3. far-right
                            4. far-left
                        "Near" means nearest the camera - the bottom of the
                        image. Going round the edge in order matters; getting
                        it wrong gives a mirrored or folded map.

        plot_w_cm       plot width, left to right   -> becomes garden x
        plot_d_cm       plot depth, near to far     -> becomes garden y
        """
        if cv2 is None:
            raise ImportError("PlanarProjector needs OpenCV:\n"
                              "    python -m pip install opencv-python")

        src = np.asarray(image_corners, np.float32).reshape(4, 2)

        # Near edge becomes y = 0, far edge y = depth. So the map comes out
        # the way you are standing: near the camera is at the bottom.
        dst = np.array([[0.0,       0.0],
                        [plot_w_cm, 0.0],
                        [plot_w_cm, plot_d_cm],
                        [0.0,       plot_d_cm]], np.float32)

        self.H = cv2.getPerspectiveTransform(src, dst)
        self.image_corners = src
        self.plot_w_cm = float(plot_w_cm)
        self.plot_d_cm = float(plot_d_cm)

        # How badly the four points disagree with a perfect rectangle. Any
        # real reprojection error here is you mis-clicking a corner, since
        # four points fit a homography exactly.
        back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), self.H)
        self.fit_error_cm = float(np.abs(back.reshape(4, 2) - dst).max())

    def to_garden(self, uv, pose=None, mount=None):
        """
        Pixels -> (x, y) in centimetres on the plot.

        pose and mount are accepted and ignored, so this can be dropped into
        the same code path as GroundProjector.

        Points that land far outside the plot come back as NaN - they are
        extrapolation past your stakes, and the perspective maths goes
        non-linear out there.
        """
        pts = np.asarray(uv, np.float64).reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2).astype(np.float64)

        # Allow half a plot of slack, then give up.
        slack_x, slack_y = self.plot_w_cm * 0.5, self.plot_d_cm * 0.5
        bad = ((out[:, 0] < -slack_x) | (out[:, 0] > self.plot_w_cm + slack_x) |
               (out[:, 1] < -slack_y) | (out[:, 1] > self.plot_d_cm + slack_y) |
               ~np.isfinite(out).all(axis=1))
        out[bad] = np.nan
        return out

    def to_robot(self, uv, mount=None):
        return self.to_garden(uv)

    def cm_per_pixel(self, uv, pose=None, mount=None):
        """
        Ground covered by one pixel, at this point in the frame.

        On an angled photo this varies enormously - a pixel at the far edge
        can cover ten times the soil of one at the near edge. That ratio is
        your honest statement of how much worse the far field is, and it is
        worth quoting.
        """
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        here = self.to_garden(uv)
        over = self.to_garden(uv + np.array([1.0, 0.0]))
        return np.linalg.norm(over - here, axis=1)

    def __repr__(self):
        return (f"PlanarProjector(plot {self.plot_w_cm:.0f} x "
                f"{self.plot_d_cm:.0f} cm, corner fit "
                f"{self.fit_error_cm:.2e} cm)")


# ----------------------------------------------------------------------------
# 6. Footprint area - how much ground a plant actually covers
# ----------------------------------------------------------------------------

def footprint_area_cm2(projector, polygon, pose=None, mount=None,
                       min_points=6):
    """
    How much ground a detected object covers, in square centimetres.

    Projects every point of the mask outline onto the soil and measures the
    area of the resulting shape.

    WHEN THIS IS TRUE, AND WHEN IT LIES
        It measures the FOOTPRINT - the ground the object sits on - and that
        is only what you see when the camera is looking straight down.

        Tilt the camera and an object's sides come into view. Those side
        pixels are not on the ground, but the projection puts them there
        anyway, smearing the shape away from the camera and inflating the
        area. A 10 cm jar photographed from 45 degrees reads far wider than
        10 cm, because you are seeing its wall as well as its base.

        So: the steeper the camera, the truer the number. Near 90 degrees
        (straight down) it is a real measurement. Below about 60 it is an
        overestimate, and the error grows with the object's height.

        This is the same flat-world assumption the rest of the file makes.
        Here it just shows up as area instead of position.

    Returns None when too few outline points land on the ground to trust.
    """
    if polygon is None:
        return None

    pts = np.asarray(polygon, np.float64).reshape(-1, 2)
    if len(pts) < min_points:
        return None

    ground = projector.to_garden(pts, pose, mount)
    good = ground[~np.isnan(ground[:, 0])]
    if len(good) < min_points:
        return None

    # Shoelace over the projected outline. The mask contour already comes
    # round the shape in order, so it stays in order once projected.
    x, y = good[:, 0], good[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return float(area) if np.isfinite(area) else None


def equivalent_diameter_cm(area_cm2):
    """
    The width of a circle with this area.

    Easier to picture than square centimetres, and easier to compare against
    a tool width: "18 cm across" says more than "254 square centimetres".
    """
    if not area_cm2 or area_cm2 <= 0:
        return None
    return float(2.0 * math.sqrt(area_cm2 / math.pi))


def footprint_from_base(projector, box, pose=None, mount=None):
    """
    Footprint of a plant from a drawn BOX rather than a mask.

    Projecting a whole box onto the soil is wrong. A box's top edge is the
    top of the plant, which is in the air - push it onto the ground and you
    measure the plant's HEIGHT as though it were ground, which on an angled
    photo inflates the area enormously.

    So use only the bottom edge, which is the one part of the box that
    really is on the soil. Project its two ends, measure the ground distance
    between them, and treat that as the diameter of a round footprint.

    Returns (area_cm2, width_cm), or (None, None) if the edge misses the
    ground.
    """
    x1, y1, x2, y2 = box
    ends = projector.to_garden([(x1, y2), (x2, y2)], pose, mount)
    if np.isnan(ends).any():
        return None, None

    width = float(np.linalg.norm(ends[1] - ends[0]))
    if not np.isfinite(width) or width <= 0:
        return None, None

    return math.pi * (width / 2.0) ** 2, width
