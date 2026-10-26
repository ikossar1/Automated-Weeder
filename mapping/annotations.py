"""
Read boxes drawn on an image, instead of running a detector.

WHY THIS EXISTS
    Delaney's tool outputs a photo with the crops already boxed and
    labelled. Running YOLOE over that picture would be re-deriving
    something she has already decided - and worse, our detector would see
    her drawn rectangles as well as the plants.

    So this reads her boxes directly. Her call on what is a crop stands;
    this half of the project only works out where those crops are on the
    ground.

    It produces the same Detection objects detect.py does, so everything
    downstream - projection, merging, grid, crops, maps - is unchanged.

HOW IT FINDS THEM
    The boxes are axis-aligned rectangles in near-pure green, which almost
    nothing in a photo of soil and leaves happens to be. So:

      1. mask the pure-green pixels
      2. find columns and rows that are mostly green - those are the edges
      3. test every pair of verticals against every pair of horizontals,
         and keep the combinations whose four sides are really drawn in

    Connected components will not do, because overlapping boxes merge into
    one blob - two of the four in the test image do exactly that. Working
    from edges keeps them separate.

WHAT IT ASSUMES
    Rectangles, axis-aligned, drawn in a consistent colour, a few pixels
    thick. If her tool changes colour or starts drawing rotated boxes, the
    colour range and the edge test are the two things to revisit.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from detect import Detection

# Her boxes are saturated green - BGR around (16, 243, 18). Testing how far
# green leads the other two channels separates a drawn line from sunlit
# foliage, which plain "is it greenish" does not: a bright leaf trips a
# simple g > 150 test and buries the annotations in noise.
GREEN_LEAD = 120

# A drawn edge runs most of the way across its box, so anything shorter than
# this is a label banner, a tick, or JPEG noise.
MIN_EDGE_PX = 200
MIN_SIDE_PX = 150


def green_mask(img):
    b, g, r = cv2.split(img.astype(np.int16))
    return ((g - np.maximum(r, b)) > GREEN_LEAD).astype(np.uint8)


def _runs(vec, min_len):
    """Contiguous stretches of at least min_len set pixels."""
    out, start = [], None
    for i, v in enumerate(vec):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(vec) - start >= min_len:
        out.append((start, len(vec) - 1))
    return out


def _segments(mask, axis, min_len):
    """
    Drawn lines, as (position, from, to).

    axis 0 scans rows and returns horizontal lines; axis 1 scans columns and
    returns vertical ones.

    This replaces summing green down whole columns and looking for peaks.
    That approach fails exactly where it matters: two boxes whose tops are
    14 px apart merge into one peak, and one of them disappears. A line
    found as its own run keeps its position and its extent.
    """
    n = mask.shape[0] if axis == 0 else mask.shape[1]
    segs = []
    for i in range(n):
        line = mask[i] if axis == 0 else mask[:, i]
        for a, b in _runs(line, min_len):
            segs.append((i, a, b))
    return _merge(segs)


def _merge(segs, pos_tol=8, end_tol=24):
    """Fold the several pixel rows of one drawn line into a single entry."""
    kept = []
    for pos, a, b in sorted(segs):
        for i, (p2, a2, b2) in enumerate(kept):
            if abs(pos - p2) <= pos_tol and abs(a - a2) <= end_tol and abs(b - b2) <= end_tol:
                kept[i] = ((p2 + pos) // 2, min(a, a2), max(b, b2))
                break
        else:
            kept.append((pos, a, b))
    return kept


def _spans(seg, lo, hi, slack=20):
    """Does this line cover the stretch lo..hi?"""
    return seg[1] <= lo + slack and seg[2] >= hi - slack


def find_boxes(image_path, min_edge=None, min_area_frac=None):
    """
    Every drawn rectangle in the image, as (x1, y1, x2, y2).

    Pairs up horizontal lines that share an x-span, then keeps the pairs
    with real vertical lines closing both ends.
    """
    if cv2 is None:
        sys.exit("annotations need OpenCV:  python -m pip install opencv-python")

    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Cannot read {image_path}")

    mask = green_mask(img)
    horiz = _segments(mask, 0, MIN_EDGE_PX)
    vert = _segments(mask, 1, MIN_EDGE_PX)

    found = []
    for i, top in enumerate(horiz):
        for bottom in horiz[i + 1:]:
            if bottom[0] - top[0] < MIN_SIDE_PX:
                continue
            # Same left and right ends, within a drawn line's own width.
            if abs(top[1] - bottom[1]) > 24 or abs(top[2] - bottom[2]) > 24:
                continue

            x1 = min(top[1], bottom[1])
            x2 = max(top[2], bottom[2])
            if x2 - x1 < MIN_SIDE_PX:
                continue

            left = any(abs(v[0] - x1) <= 12 and _spans(v, top[0], bottom[0])
                       for v in vert)
            right = any(abs(v[0] - x2) <= 12 and _spans(v, top[0], bottom[0])
                        for v in vert)
            if left and right:
                found.append((x1, top[0], x2, bottom[0]))

    found.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    kept = []
    for box in found:
        if not any(_iou(box, k) > 0.55 for k in kept):
            kept.append(box)

    kept.sort(key=lambda b: (b[1], b[0]))
    return kept, img


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


class AnnotationReader:
    """
    A stand-in for PlantDetector that reads drawn boxes rather than running
    a model. Same detect() signature, same Detection objects out.

        det = AnnotationReader(label="crop")
        for d in det.detect("delaney_output/6crops.jpg"):
            print(d.label, d.base_px)
    """

    def __init__(self, label="crop", min_edge=0.70, confidence=1.0):
        self.label = label
        self.min_edge = float(min_edge)
        self.confidence = float(confidence)

    def detect(self, image, save=False):
        boxes, _ = find_boxes(image, min_edge=self.min_edge)
        out = []

        for x1, y1, x2, y2 in boxes:
            # No mask, so the base is the bottom-centre of the box. For a
            # box drawn round a plant that is about where the stem meets
            # the soil - a little low if the plant leans, but her box is
            # all the information there is.
            out.append(Detection(
                label=self.label,
                confidence=self.confidence,
                base_px=((x1 + x2) / 2.0, y2),
                box=(x1, y1, x2, y2),
                area_px=(x2 - x1) * (y2 - y1),
                polygon=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                 dtype=np.float64),
                from_box=True,
            ))
        return out


def draw_found(image_path, out_path="_annotations_found.png", min_edge=0.70):
    """Save a picture of what was read, to check it by eye."""
    boxes, img = find_boxes(image_path, min_edge=min_edge)

    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 90, 230), 6)
        cv2.circle(img, (int((x1 + x2) / 2), y2), 16, (255, 255, 255), -1)
        cv2.circle(img, (int((x1 + x2) / 2), y2), 16, (40, 90, 230), 6)
        for colour, thick in (((0, 0, 0), 12), ((40, 90, 230), 5)):
            cv2.putText(img, f"read {i}", (x1 + 10, max(y1 - 18, 46)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, colour, thick)

    h, w = img.shape[:2]
    s = min(1.0, 1500.0 / max(w, h))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img)
    return boxes, str(out_path)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "delaney_output/6crops.jpg"
    boxes, out = draw_found(path)

    print(f"{path}: read {len(boxes)} drawn box(es)\n")
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        print(f"  {i}. box ({x1:5d},{y1:5d}) to ({x2:5d},{y2:5d})"
              f"   {x2-x1:5d} x {y2-y1:5d} px"
              f"   base ({(x1+x2)//2:5d},{y2:5d})")
    print(f"\nCheck it: {out}")
    print("The blue boxes should sit exactly on her green ones, and each")
    print("white dot should be at the bottom-centre, on the soil.")


# ----------------------------------------------------------------------------
# Delaney's pipeline output
# ----------------------------------------------------------------------------

class ResultsReader:
    """
    Read the results.csv that BioClip_pipeline.py writes.

    MUCH BETTER THAN READING THE DRAWN BOXES
        find_boxes() above recovers rectangles from coloured pixels in an
        annotated image. It works, but it is reverse-engineering a picture
        of data back into data, and it can only recover the rectangle - the
        species and the crop/not-crop decision are lost, because they were
        only ever drawn as text.

        Her pipeline already writes all of it as numbers:

            plant_id, category, best_match, confidence, crop_prob,
            x1, y1, x2, y2

        So read that. Exact coordinates, the species name, and her
        crop/not-crop call, with nothing inferred. Keep find_boxes() for
        the case where an annotated image is all there is.

    HER CATEGORIES
        crop, possible crop, not crop, uncertain

        These go straight onto the plants, so --only filters on them with
        no mapping step:

            --only crop                     just the confident ones
            --only "crop,possible crop"     include the maybes

    ONE THING WE DO DIFFERENTLY
        Her CSV has a `centroid`, the middle of the box. We do not use it.
        Position comes from the BOTTOM of the box, because that is where
        the plant meets the soil and the soil is the surface all the
        geometry is built on. A centroid floats somewhere up in the leaves.
    """

    def __init__(self, csv_path, use_category=True):
        self.csv_path = str(csv_path)
        self.use_category = use_category
        self._rows = None

    def _load(self):
        if self._rows is None:
            import csv as _csv
            with open(self.csv_path, newline="", encoding="utf-8-sig") as fh:
                self._rows = list(_csv.DictReader(fh))
        return self._rows

    def detect(self, image=None, save=False):
        """
        Detections from the CSV. `image` is ignored - the boxes are already
        decided - but kept so this drops into the same call as the others.
        """
        out = []
        for r in self._load():
            try:
                x1, y1 = float(r["x1"]), float(r["y1"])
                x2, y2 = float(r["x2"]), float(r["y2"])
            except (KeyError, TypeError, ValueError):
                continue

            d = Detection(
                label=(r.get("best_match") or "plant").strip(),
                confidence=float(r.get("confidence") or 0.0),
                base_px=((x1 + x2) / 2.0, y2),
                box=(x1, y1, x2, y2),
                area_px=(x2 - x1) * (y2 - y1),
                polygon=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                 dtype=np.float64),
                from_box=True,
            )
            d.category = (r.get("category") or "").strip() if self.use_category else ""
            d.plant_id = r.get("plant_id", "")
            d.crop_prob = r.get("crop_prob", "")
            out.append(d)

        out.sort(key=lambda d: -d.confidence)
        return out

    def summary(self):
        from collections import Counter
        cats = Counter((r.get("category") or "?").strip() for r in self._load())
        return dict(cats)
