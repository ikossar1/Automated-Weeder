"""
Cropped frames, one per plant - the handoff to the species classifier.

THE SPLIT OF WORK
    This half of the project answers WHERE: it finds plants, puts them in
    centimetres on the ground, and says which grid cell each one is in. It
    deliberately does not try to answer WHAT, because the detector is bad at
    species (prompting YOLOE with "poppy" scored 0.00 where "flower" scored
    0.24).

    A separate classifier answers WHAT. It does not need the whole photo -
    it needs one plant, centred and cropped. That is what this file writes.

THE ROUND TRIP
    1. scan.py --crops   writes crops/plant_001_C1R2.png and crops.csv,
                         with the species column left blank
    2. your teammate     fills in the species column and sends it back
    3. scan.py --species crops_done.csv
                         reads the names in and relabels the map

    The join key is plant_id. crop_file is there too so the file can be
    matched by name if that is easier on the other end - both identify the
    same plant, so either works.

WHAT A CROP CONTAINS
    The plant's mask box plus a margin, taken from the original photo at
    full resolution. The margin matters: classifiers do better with a little
    context than with a shape cut hard to its outline.

    One crop per plant, not per detection. A plant seen in three frames
    produces one crop - the sighting the detector was most confident about.
"""
from __future__ import annotations

import csv
import pathlib

try:
    import cv2
except ImportError:
    cv2 = None

# Columns written to crops.csv. species is last and blank on purpose: it is
# the one your teammate fills in.
# Columns written to crops.csv. species and category are last and blank on
# purpose - they are the two your teammate fills in.
FIELDS = ["plant_id", "crop_file", "cell", "col", "row",
          "x_cm", "y_cm", "footprint_cm2", "width_cm",
          "confidence", "detector_label", "source_image",
          "species", "category"]


def cell_for(cells, plant):
    """Which grid cell holds this plant. None if it falls outside."""
    for c in cells:
        if c["x0"] <= plant.x_cm <= c["x1"] and c["y0"] <= plant.y_cm <= c["y1"]:
            return c
    return None


def write_crops(out_dir, best, cells, margin=0.18, min_px=64,
                flat=True):
    """
    Save one cropped image per plant, plus the manifest.

    best    {plant_id: (Plant, Detection, source_image_path)} - the sighting
            with the highest confidence for each plant
    cells   output of GardenMap.grid(), for the cell label
    margin  extra context round the box, as a fraction of its size

    Returns (crop_dir, manifest_path, rows_written).
    """
    if cv2 is None:
        print("crops need OpenCV:  python -m pip install opencv-python")
        return None, None, 0

    # flat=True puts the images beside the manifest in one folder, so the
    # whole handoff is a single directory to zip and send. Nothing to
    # navigate into, and crop_file in the CSV is just the filename next to
    # it. flat=False keeps them in a crops/ subfolder instead.
    out_dir = pathlib.Path(out_dir)
    crop_dir = out_dir if flat else out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    by_cell = {p.id: cell_for(cells, p) for p, _, _ in best.values()}
    loaded = {}
    rows = []

    for pid in sorted(best):
        plant, det, src = best[pid]

        if src not in loaded:
            loaded[src] = cv2.imread(str(src))
        img = loaded[src]
        if img is None:
            continue

        h, w = img.shape[:2]
        x1, y1, x2, y2 = det.box
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin

        # Clamp to the image, and keep a floor so a tiny detection does not
        # become a crop too small for a classifier to use.
        cx1 = max(0, int(round(min(x1 - pad_x, (x1 + x2) / 2 - min_px / 2))))
        cy1 = max(0, int(round(min(y1 - pad_y, (y1 + y2) / 2 - min_px / 2))))
        cx2 = min(w, int(round(max(x2 + pad_x, (x1 + x2) / 2 + min_px / 2))))
        cy2 = min(h, int(round(max(y2 + pad_y, (y1 + y2) / 2 + min_px / 2))))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue

        cell = by_cell.get(plant.id)
        tag = cell["name"] if cell else "offgrid"
        name = f"plant_{plant.id:03d}_{tag}.png"

        cv2.imwrite(str(crop_dir / name), img[cy1:cy2, cx1:cx2],
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])

        across = (2.0 * (plant.size_cm2 / 3.14159265) ** 0.5
                  if plant.size_cm2 > 0 else 0.0)

        rows.append({
            "plant_id": plant.id,
            "crop_file": name,
            "cell": tag,
            "col": (cell["col"] + 1) if cell else "",
            "row": (cell["row"] + 1) if cell else "",
            "x_cm": f"{plant.x_cm:.2f}",
            "y_cm": f"{plant.y_cm:.2f}",
            "footprint_cm2": f"{plant.size_cm2:.1f}",
            "width_cm": f"{across:.1f}",
            "confidence": f"{plant.confidence:.4f}",
            "detector_label": plant.label,
            "source_image": pathlib.Path(src).name,
            "species": "",          # your teammate fills these two in
            "category": "",
        })

    manifest = out_dir / "crops.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    return crop_dir, manifest, len(rows)


def read_species(path):
    """
    Read a filled-in manifest back.

    Accepts anything with a species column plus either plant_id or
    crop_file. Everything else in the file is ignored, so your teammate can
    add her own columns - confidence scores, alternative names, notes -
    without breaking this.

    Also reads a `category` column if there is one - that is the A/B sort
    your teammate's logic produces, and it is what --only filters on.

    Returns ({plant_id: (species, category)}, {crop_file: (species, category)}).
    """
    by_id, by_file = {}, {}

    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("species") or "").strip()
            cat = (row.get("category") or "").strip()
            if not name and not cat:
                continue

            pid = (row.get("plant_id") or "").strip()
            if pid.isdigit():
                by_id[int(pid)] = (name, cat)
            cf = (row.get("crop_file") or "").strip()
            if cf:
                by_file[cf] = (name, cat)

    return by_id, by_file


def apply_species(gmap, by_id, by_file=None, crop_names=None):
    """
    Put the species names onto the plants, so the map shows them.

    plant_id wins when both are available; crop_file is the fallback.
    Returns how many plants were named.
    """
    named = 0
    for p in gmap.plants:
        hit = by_id.get(p.id)
        if hit is None and by_file and crop_names:
            hit = by_file.get(crop_names.get(p.id, ""))
        if not hit:
            continue

        name, cat = hit
        if name:
            p.label = name
            named += 1
        if cat:
            p.category = cat
    return named
