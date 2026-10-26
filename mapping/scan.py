"""
Build a garden map from a folder of images.

    python scan.py photos/ --height 35 --tilt 60
    python scan.py photos/ --height 35 --tilt 60 --poses poses.csv --rows 30

Runs YOLOE on every image, projects each plant's base pixel onto the soil,
merges repeat sightings, then writes:

    garden_map.json   every plant, in centimetres from the corner stake
    garden_map.csv    the same as a flat table
    garden_map.png    top-down picture of the bed

POSES
    Each image needs to know where the robot was when it was taken. Supply a
    CSV with a header row:

        image,x_cm,y_cm,heading_deg,pan_deg,tilt_deg
        frame_001.jpg,0,0,0,0,60
        frame_002.jpg,20,0,0,0,60
        frame_003.jpg,40,0,0,-15,60

    Any column except `image` may be left out and will fall back to the
    command-line default. Without --poses every image is treated as taken
    from the origin, which is only right if the robot never moved.

    This file is the one part nobody can generate for you - it comes off your
    own motor commands. Logging it as you capture is much easier than
    reconstructing it afterwards.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

import crops as crops_mod
import detect
import geometry
from garden_map import GardenMap

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import safestop


def read_poses(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image") or "").strip()
            if not name:
                continue
            out[name] = {k: float(v) for k, v in row.items()
                         if k != "image" and v not in (None, "")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a top-down garden map from camera images.")

    ap.add_argument("images", help="image file, or a folder of them")
    ap.add_argument("--results", default=None, metavar="CSV",
                    help="results.csv from Delaney's BioClip pipeline. Uses "
                         "her boxes, species and crop/not-crop call directly "
                         "- no detection run, nothing inferred from pixels")
    ap.add_argument("--annotations", action="store_true",
                    help="read boxes already drawn on the image instead of "
                         "running YOLOE. Use this for a teammate's annotated "
                         "output - her call on what is a crop stands, and we "
                         "only work out where those crops are on the ground")
    ap.add_argument("--annotation-label", default="crop",
                    help="what to call the boxes read from the image "
                         "(default: crop)")

    ap.add_argument("--prompts", default="plant",
                    help="comma-separated words for YOLOE. Generic beats "
                         "specific: 'plant' scored 3x 'lettuce'. (default: plant)")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="model input size. 640 badly degrades large photos")

    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--hfov", type=float, default=None,
                    help="horizontal field of view in degrees. Only used if "
                         "there is no calibration file - gives rough answers")

    ap.add_argument("--height", type=float, default=None,
                    help="lens height above the soil, cm (camera route)")
    ap.add_argument("--tilt", type=float, default=None,
                    help="downward tilt from horizontal, deg. 90 = straight "
                         "down (camera route)")
    ap.add_argument("--pan", type=float, default=0.0)
    ap.add_argument("--forward", type=float, default=0.0,
                    help="lens offset ahead of robot centre, cm")
    ap.add_argument("--left", type=float, default=0.0,
                    help="lens offset left of robot centre, cm")

    ap.add_argument("--poses", default=None, help="CSV of per-image robot poses")
    ap.add_argument("--merge", type=float, default=5.0,
                    help="two detections closer than this are one plant, cm. "
                         "Set it to ~3x your measured position error, NOT to "
                         "your crop spacing (default: 5)")
    ap.add_argument("--rows", type=float, default=None,
                    help="crop row spacing in cm. Supplying it classifies "
                         "crop vs weed by position")
    ap.add_argument("--row-offset", type=float, default=0.0)
    ap.add_argument("--row-tolerance", type=float, default=7.5)
    ap.add_argument("--row-axis", default="x", choices=["x", "y"])

    ap.add_argument("--corners", nargs=8, type=float, default=None,
                    metavar=("U1", "V1", "U2", "V2", "U3", "V3", "U4", "V4"),
                    help="FOUR-CORNER ROUTE - no camera data needed. The "
                         "plot's four corners as pixels, in order: near-left, "
                         "near-right, far-right, far-left")
    ap.add_argument("--plot", nargs=2, type=float, default=None,
                    metavar=("WIDTH_CM", "DEPTH_CM"),
                    help="real size of that plot, e.g. --plot 400 300 for a "
                         "4 m wide by 3 m deep bed")

    ap.add_argument("--grid", default="6x4", metavar="COLSxROWS",
                    help="also draw the bed as a grid of cells, coloured in "
                         "where plants are (default: 6x4). 'none' to skip")
    ap.add_argument("--bed", nargs=4, type=float, default=None,
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="real extent of the bed in cm. Without it the grid "
                         "is fitted to whatever was found, which moves if you "
                         "scan a different area")

    ap.add_argument("--out", default="garden_map",
                    help="name for this run. Everything it produces goes in "
                         "maps/<name>/ (default: garden_map)")
    ap.add_argument("--crops", action="store_true",
                    help="save one cropped image per plant, plus crops.csv "
                         "with a blank species column - the handoff to a "
                         "species classifier")
    ap.add_argument("--crops-dir", default="handoff",
                    help="where crop bundles go (default: handoff). Each run "
                         "gets handoff/<name>/ with the images and the CSV "
                         "together - one folder to zip and send")
    ap.add_argument("--crop-margin", type=float, default=0.18,
                    help="context kept round each crop, as a fraction of the "
                         "box (default: 0.18)")
    ap.add_argument("--edge-margin", type=float, default=5.0,
                    help="how far outside the plot a plant may sit and still "
                         "count, cm (default: 5)")
    ap.add_argument("--bioclip", action="store_true",
                    help="name every plant with BioCLIP and decide crop vs "
                         "weed in the same run. YOLOE finds them, BioCLIP "
                         "says what they are. Needs open_clip_torch")
    ap.add_argument("--candidates", default=None,
                    help="which crops are planted in this bed, comma "
                         "separated, e.g. \"Ocimum basilicum\". Narrows "
                         "what BioCLIP chooses between. Weeds always stay "
                         "in the list. Left out, all crops are considered")
    ap.add_argument("--bioclip-margin", type=float, default=0.15,
                    help="context kept round each crop before classifying")
    ap.add_argument("--strict-crop", action="store_true",
                    help="drop plants the classifier was unsure about. By "
                         "default 'uncertain', 'possible crop' and anything "
                         "unlabelled count as crops and stay on the map - "
                         "leaving a weed is cheaper than mulching a basil")
    ap.add_argument("--only", default=None, metavar="CATEGORIES",
                    help="show only these categories on the map, comma "
                         "separated, e.g. --only A. Categories come from the "
                         "category column of the --species CSV")
    ap.add_argument("--species", default=None, metavar="CSV",
                    help="a filled-in crops.csv coming back. Needs a species "
                         "column plus plant_id or crop_file; the names are "
                         "put on the map")

    ap.add_argument("--formats", default="png",
                    help="image formats for the maps, comma separated. "
                         "png opens anywhere; svg and pdf are vector, so "
                         "they stay sharp at any size - better for the "
                         "report. e.g. --formats png,svg (default: png)")
    ap.add_argument("--out-dir", default="maps",
                    help="where run folders are kept (default: maps)")
    ap.add_argument("--dark", action="store_true", help="render for dark mode")
    args = ap.parse_args(argv)

    # -- images ------------------------------------------------------------
    target = pathlib.Path(args.images)
    if target.is_dir():
        files = sorted(p for p in target.iterdir()
                       if p.suffix.lower() in detect.IMAGE_SUFFIXES)
    else:
        files = [pathlib.Path(detect.find_image(args.images))]

    if not files:
        sys.exit(f"No images found in {target}")

    # -- the projector -----------------------------------------------------
    # Two routes to the same answer:
    #   four corners  - no camera data needed, just a plot of known size
    #   height + tilt - needs the camera characterised, works anywhere
    if args.corners:
        if not args.plot:
            sys.exit("--corners also needs --plot WIDTH_CM DEPTH_CM")

        pts = [(args.corners[i], args.corners[i + 1]) for i in range(0, 8, 2)]
        proj = geometry.PlanarProjector(pts, args.plot[0], args.plot[1])
        mount = geometry.CameraMount(0, 90)      # unused; keeps the loop happy
        intr = None

        print(f"Method      : four corners - no camera data used")
        print(f"Plot        : {args.plot[0]:.0f} x {args.plot[1]:.0f} cm")
        print(f"Corners     : "
              + "  ".join(f"({u:.0f},{v:.0f})" for u, v in pts))

        from PIL import Image
        iw, ih = Image.open(files[0]).size
        near = proj.cm_per_pixel([(iw / 2, ih * 0.92)])[0]
        far = proj.cm_per_pixel([(iw / 2, ih * 0.08)])[0]
        if np.isfinite(near) and np.isfinite(far) and near > 0:
            print(f"Resolution  : {near:.2f} cm/px near, {far:.2f} cm/px far "
                  f"-> {far / near:.1f}x coarser at the back")

        if args.bed is None:
            args.bed = [0.0, 0.0, args.plot[0], args.plot[1]]

    else:
        if args.height is None or args.tilt is None:
            sys.exit("Give one of:\n"
                     "  --corners u1 v1 u2 v2 u3 v3 u4 v4 --plot W D\n"
                     "      (no camera data needed)\n"
                     "  --height H --tilt T [--hfov F]\n"
                     "      (needs the camera characterised)")

        cal = pathlib.Path(args.calibration)
        if cal.exists():
            intr = geometry.Intrinsics.load(cal)
            print(f"Calibration : {cal}")
        else:
            if args.hfov is None:
                sys.exit(f"No {cal} and no --hfov given.\n"
                         f"Either run calibrate.py, or pass your lens's "
                         f"horizontal field of view, e.g. --hfov 70")
            from PIL import Image
            w, h = Image.open(files[0]).size
            intr = geometry.Intrinsics.from_fov(w, h, args.hfov)
            print(f"Calibration : NONE - using --hfov {args.hfov} deg "
                  f"(rough, several cm of error)")

        print(f"Lens        : {intr}")

        mount = geometry.CameraMount(args.height, args.tilt, args.pan,
                                     args.forward, args.left)
        proj = geometry.GroundProjector(intr, mount)

    poses = read_poses(args.poses) if args.poses else {}
    if args.poses:
        print(f"Poses       : {len(poses)} row(s) from {args.poses}")
    else:
        print("Poses       : none given - every frame treated as the origin")

    # -- run ---------------------------------------------------------------
    if args.results:
        import annotations as ann
        det = ann.ResultsReader(args.results)
        print(f"Detector    : reading {args.results}")
        print(f"              her categories: {det.summary()}")
    elif args.annotations:
        import annotations as ann
        print(f"Detector    : reading drawn boxes, labelled "
              f"'{args.annotation_label}'")
        det = ann.AnnotationReader(label=args.annotation_label)
    else:
        print(f"Prompts     : {args.prompts.split(',')}")
        print(f"Loading model...")
        det = detect.PlantDetector(args.prompts.split(","),
                                   conf=args.conf, imgsz=args.imgsz)

    gmap = GardenMap(merge_radius_cm=args.merge)
    classifier = None
    best_shot = {}          # plant_id -> (Plant, Detection, source image)
    print(f"\nScanning {len(files)} image(s):")

    for f in files:
        row = poses.get(f.name, {})
        pose = geometry.RobotPose(row.get("x_cm", 0.0),
                                  row.get("y_cm", 0.0),
                                  row.get("heading_deg", 0.0))
        frame_mount = mount.at(tilt_deg=row.get("tilt_deg"),
                               pan_deg=row.get("pan_deg"))

        found = det.detect(str(f))

        # Name them straight away, so category rides along with the
        # detection and the map can be filtered in the same run.
        if found and args.bioclip:
            if classifier is None:
                import bioclip
                keep = ([c.strip() for c in args.candidates.split(",")]
                        if args.candidates else None)
                cands = bioclip.build_candidates(keep)
                crops = [n for n, c in cands if c in ("crop", "ambiguous")]
                print(f"    BioCLIP choosing between {len(cands)} names "
                      f"({len(crops)} crop, rest weeds)")
                if keep:
                    print(f"    crops: {', '.join(crops)}")
                print("    loading (first run downloads ~1 GB)...")
                classifier = bioclip.Classifier(candidates=cands)
                print("    done")
            from PIL import Image as _Image
            photo = _Image.open(f).convert("RGB")
            for d in found:
                import bioclip as _bc
                res = classifier.classify(_bc.crop_image(photo, d.box,
                                                         args.bioclip_margin))
                d.label = res["species"]
                d.category = res["category"]
                d.species_conf = res["confidence"]

        if not found:
            print(f"  {f.name:28s} nothing found")
            gmap.frames += 1
            continue

        base = np.array([d.base_px for d in found], np.float64)
        xy = proj.to_garden(base, pose, frame_mount)

        # Footprint: project the whole mask outline onto the ground and
        # measure it. Only honest from a steep camera - see geometry.py.
        areas = []
        for d in found:
            if getattr(d, "from_box", False):
                a, _w = geometry.footprint_from_base(proj, d.box, pose,
                                                     frame_mount)
            else:
                a = geometry.footprint_area_cm2(proj, d.polygon, pose,
                                                frame_mount)
            areas.append(a)

        before = len(gmap.plants)
        landed = gmap.add_frame(xy, found, areas)
        added = len(gmap.plants) - before

        # Keep the clearest sighting of each plant - that is the one worth
        # cropping and sending to the classifier.
        for plant, d in zip(landed, found):
            if plant is None:
                continue
            cat = getattr(d, "category", "")
            if cat and not plant.category:
                plant.category = cat
            prev = best_shot.get(plant.id)
            if prev is None or d.confidence > prev[1].confidence:
                best_shot[plant.id] = (plant, d, str(f))

        missed = int(np.isnan(xy[:, 0]).sum())
        note = f"  ({missed} off-ground)" if missed else ""
        print(f"  {f.name:28s} {len(found):3d} detected -> "
              f"{added} new, {len(found) - added - missed} merged{note}")

    # -- classify, report, write ------------------------------------------
    if args.rows:
        c = gmap.classify_by_rows(args.rows, args.row_offset,
                                  args.row_tolerance, args.row_axis)
        print(f"\nRow spacing {args.rows:.0f} cm, "
              f"tolerance +/-{args.row_tolerance:.1f} cm "
              f"-> {c['crop']} crop, {c['weed']} weed")

    # Species names arrive before anything is drawn, so the maps carry them.
    # Plants outside the plot are not in your garden. The projector allows
    # some slack past the corners so a plant right on the edge is not lost
    # to a small error, but that slack is not an invitation - anything
    # properly outside gets dropped here.
    if args.bed:
        bx0, by0, bx1, by1 = args.bed
        m = args.edge_margin
        inside = [q for q in gmap.plants
                  if bx0 - m <= q.x_cm <= bx1 + m and by0 - m <= q.y_cm <= by1 + m]
        outside = len(gmap.plants) - len(inside)
        if outside:
            gone = [q.id for q in gmap.plants if q not in inside]
            gmap.plants = inside
            print(f"\nOff the plot    : dropped {outside} plant(s) {gone} "
                  f"further than {m:.0f} cm outside it")

    if args.species and not pathlib.Path(args.species).exists():
        print(f"\nSpecies         : {args.species} not there yet - "
              f"nothing to label.")
        print(f"                  Send the handoff folder, get it back "
              f"filled in, run again.")
        args.species = None

    if args.species:
        by_id, by_file = crops_mod.read_species(args.species)
        named = crops_mod.apply_species(gmap, by_id, by_file)
        print(f"\nSpecies         : named {named} of {len(gmap.plants)} "
              f"plant(s) from {args.species}")
        unnamed = [p.id for p in gmap.plants if p.id not in by_id]
        if unnamed:
            print(f"                  no species given for plant(s) {unnamed}")
        print(f"                  categories: {gmap.categories()}")

    if args.only and not any(q.category for q in gmap.plants):
        print(f"\nFilter          : --only {args.only} ignored - no plant has "
              f"a category yet.")
        print(f"                  Showing everything. The filter starts "
              f"working once the")
        print(f"                  category column comes back filled in.")

    elif args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        kept, wrong, blank = gmap.keep_categories(
            wanted, unknown_is_crop=not args.strict_crop)
        print("")
        print(f"Filter          : showing category {wanted} only")
        print(f"                  kept {kept}, hid {wrong} in other "
              f"categories")
        if blank:
            print(f"                  hid {blank} with NO category - those "
                  f"are unclassified, not excluded")
        if not args.species:
            print("                  (no --species given, so nothing has a "
                  "category yet)")

    print("\n" + gmap.summary())

    # Everything lands in one folder, named after the run. Keeps the code
    # folder clean and makes a run easy to find, keep or delete as a unit.
    out_dir = pathlib.Path(args.out_dir) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / args.out

    mode = "dark" if args.dark else "light"
    written = [gmap.save(f"{stem}.json"),
               gmap.write_csv(f"{stem}.csv")]

    formats = [f.strip().lstrip(".").lower()
               for f in args.formats.split(",") if f.strip()]
    allowed = {"png", "svg", "pdf"}
    bad = [f for f in formats if f not in allowed]
    if bad:
        sys.exit(f"--formats: don't know {bad}. Pick from png, svg, pdf.")

    for fmt in formats:
        p = gmap.render(f"{stem}.{fmt}", mode=mode, bed=args.bed)
        if p:
            written.append(p)

    if args.grid and args.grid.lower() != "none":
        try:
            cols, rows = (int(v) for v in args.grid.lower().split("x"))
        except ValueError:
            sys.exit(f"--grid wants COLSxROWS, e.g. 6x4 (got {args.grid!r})")

        print("\n" + gmap.grid_summary(cols, rows, args.bed))
        written.append(gmap.write_grid_csv(f"{stem}_grid.csv",
                                           cols, rows, args.bed))
        for fmt in formats:
            gp = gmap.render_grid(f"{stem}_grid.{fmt}", cols, rows, args.bed,
                                  mode=mode)
            if gp:
                written.append(gp)

    # Crops come after the grid exists, so each one can be named by its cell.
    if args.crops:
        if args.grid and args.grid.lower() != "none":
            cols_c, rows_c = (int(v) for v in args.grid.lower().split("x"))
        else:
            cols_c, rows_c = 6, 4
        cells = gmap.grid(cols_c, rows_c, args.bed)

        handoff = pathlib.Path(args.crops_dir) / args.out
        crop_dir, manifest, n = crops_mod.write_crops(
            handoff, best_shot, cells, margin=args.crop_margin, flat=True)

        if manifest:
            # Deliberately not added to `written` - that list belongs to the
            # maps folder, and the handoff bundle lives somewhere else.
            print(f"\nCrops           : {n} image(s) in {crop_dir}")
            print(f"                  {manifest.name} - species column is "
                  f"blank, ready to fill in")

    # On the four-corner route, always save the photo with the corners drawn
    # on it. The maths cannot tell you the corners were wrong - it will
    # produce a confident, nonsense map from nonsense corners. Looking at
    # this picture is the only check, and it takes a second.
    if args.corners:
        import pick_corners
        cp = pick_corners.draw_quad(files[0], pts, f"{stem}_corners.png")
        if cp:
            written.append(cp)

    print("\nWrote everything to: " + str(out_dir))
    for w in written:
        print("   " + pathlib.Path(w).name)

    if args.corners:
        print("\nOpen " + pathlib.Path(written[-1]).name + " first and check")
        print("the green outline really sits on your plot. If it does not,")
        print("every centimetre above is wrong.")

    if intr is not None and intr.approximate:
        print("\nNOTE: these positions came from an uncalibrated lens model.")
        print("Run calibrate.py before quoting any of these numbers.")

    return gmap


if __name__ == "__main__":
    with safestop.guard("the scan"):
        main()
