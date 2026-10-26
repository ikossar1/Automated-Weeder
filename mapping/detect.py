"""
YOLOE -> base pixels.

Wraps the model you already tested in 03_yoloe_test.py and pulls out the one
number per plant that geometry.py needs: the pixel where the stem meets the
soil.

WHY THE BOTTOM OF THE MASK, NOT THE CENTRE
    Leaves overhang. A lettuce's centroid floats out over soil its roots are
    nowhere near - several centimetres of error, and always in a direction
    that depends on which way the leaves happen to lean, so you cannot even
    calibrate it out. The lowest pixel of the mask is where the plant touches
    the ground, and the ground is the only surface the projection maths knows
    about. That is the point your weeding tool needs.

    This is why segmentation matters here and a plain box detector would not
    do. A box's bottom edge is the lowest *leaf*, not the stem.

WHAT WE LEARNED IN TESTING - keep both of these in mind
    Generic words beat species names, badly:
        "plant"  0.408      vs  "lettuce"  0.135
        "flower" 0.24       vs  "poppy"    0.00
    So prompt for vegetation, and let row geometry decide crop from weed.

    Input resolution is the dominant lever. 640 -> 1280 roughly doubled
    confidence on phone-sized photos. 1280 is the default here on purpose.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The weights live wherever you first downloaded them. Check the likely spots.
WEIGHT_SEARCH = [
    pathlib.Path("yoloe-11s-seg.pt"),
    pathlib.Path(__file__).parent / "yoloe-11s-seg.pt",
]


def find_weights(name="yoloe-11s-seg.pt"):
    for cand in WEIGHT_SEARCH:
        if cand.name == name and cand.exists():
            return str(cand)
    for cand in WEIGHT_SEARCH:
        alt = cand.parent / name
        if alt.exists():
            return str(alt)
    sys.exit(f"Could not find {name}. Looked in: "
             f"{[str(c.parent) for c in WEIGHT_SEARCH]}")


def find_image(name, folder="."):
    """Tolerate .jpg/.jpeg mix-ups and Windows' hidden double extensions."""
    here = pathlib.Path(folder)
    if pathlib.Path(name).exists():
        return str(name)

    stem = pathlib.Path(name).stem.lower()
    for cand in sorted(here.glob("*")):
        if cand.is_file() and cand.suffix.lower() in IMAGE_SUFFIXES:
            if cand.stem.lower().split(".")[0] == stem:
                print(f"(note: '{name}' not found, using '{cand.name}')")
                return str(cand)

    found = [c.name for c in here.glob("*")
             if c.is_file() and c.suffix.lower() in IMAGE_SUFFIXES]
    sys.exit(f"No image matching '{name}'. Images here: {found or 'none'}")


class Detection:
    """One plant, as the camera saw it. Still in pixels at this stage."""

    def __init__(self, label, confidence, base_px, box, area_px,
                 polygon=None, from_box=False):
        self.label = label
        self.confidence = float(confidence)
        self.base_px = tuple(float(v) for v in base_px)
        self.box = tuple(float(v) for v in box)
        self.area_px = float(area_px)
        # The mask outline in pixels, kept so the footprint can be measured
        # on the ground. None when the model returned a box but no mask.
        self.polygon = (np.asarray(polygon, np.float64)
                        if polygon is not None else None)
        # True when the outline is a drawn rectangle rather than a real mask.
        # Matters for footprint: a box's top edge is the plant's TOP, which
        # is in the air, so projecting the whole box onto the soil measures
        # the plant's height as if it were ground.
        self.from_box = bool(from_box)

    def __repr__(self):
        return (f"Detection({self.label!r}, {self.confidence:.1%}, "
                f"base_px=({self.base_px[0]:.0f},{self.base_px[1]:.0f}))")


def _base_from_polygon(poly):
    """
    Lowest point of a mask outline.

    Takes the median x across the bottom few percent of the shape rather than
    the single lowest vertex - one stray pixel of mask noise should not decide
    where we send the tool.
    """
    poly = np.asarray(poly, np.float64)
    if poly.ndim != 2 or len(poly) == 0:
        return None

    y_max = poly[:, 1].max()
    height = y_max - poly[:, 1].min()
    band = max(height * 0.05, 2.0)        # bottom 5% of the shape, min 2px

    bottom = poly[poly[:, 1] >= y_max - band]
    return float(np.median(bottom[:, 0])), float(y_max)


class PlantDetector:
    """
    Holds the loaded model so you pay the startup cost once.

        det = PlantDetector(["plant"])
        for d in det.detect("frame_001.jpg"):
            print(d.label, d.base_px)
    """

    def __init__(self, prompts=("plant",), weights="yoloe-11s-seg.pt",
                 conf=0.10, imgsz=1280):
        from ultralytics import YOLOE

        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.conf = float(conf)
        self.imgsz = int(imgsz)

        self.model = YOLOE(find_weights(weights))
        self.model.set_classes(self.prompts, self.model.get_text_pe(self.prompts))

    def detect(self, image, save=False):
        """Run the model on one image. Returns a list of Detection, best first."""
        r = self.model.predict(image, save=save, conf=self.conf,
                               imgsz=self.imgsz, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        polys = list(r.masks.xy) if r.masks is not None else []
        out = []

        for i, b in enumerate(r.boxes):
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            base, area, poly = None, (x2 - x1) * (y2 - y1), None
            if i < len(polys):
                poly = polys[i]
                base = _base_from_polygon(poly)
                if base is not None:
                    area = _polygon_area(poly)

            if base is None:
                # No mask for this one - bottom-centre of the box is the
                # best guess available, and it will read a little long.
                base = ((x1 + x2) / 2.0, y2)

            out.append(Detection(
                label=r.names[int(b.cls)],
                confidence=float(b.conf),
                base_px=base,
                box=(x1, y1, x2, y2),
                area_px=area,
                polygon=poly,
            ))

        out.sort(key=lambda d: -d.confidence)
        return out


def _polygon_area(poly):
    """Shoelace formula - mask area in pixels, for a rough size estimate."""
    p = np.asarray(poly, np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


if __name__ == "__main__":
    # Quick check that masks and base pixels come out sensibly.
    #   python detect.py poppy.jpeg "flower"
    img = find_image(sys.argv[1] if len(sys.argv) > 1 else "test.jpg")
    prompts = (sys.argv[2] if len(sys.argv) > 2 else "plant").split(",")

    det = PlantDetector(prompts)
    found = det.detect(img, save=True)

    print(f"{img}: {len(found)} detection(s)\n")
    for i, d in enumerate(found, 1):
        print(f"  {i:2d}. {d.label:12s} {d.confidence:5.1%}   "
              f"base px ({d.base_px[0]:6.0f},{d.base_px[1]:6.0f})   "
              f"mask area {d.area_px:,.0f} px")

    if found:
        print("\nThe base pixel should sit at the bottom of the plant, where")
        print("the stem meets the soil - not in the middle of the leaves.")
        print("Open the annotated image and check that by eye before trusting it.")
