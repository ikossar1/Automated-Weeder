"""
The garden map.

Holds every plant the robot has found, in garden coordinates, and merges
repeat sightings so one plant stays one entry no matter how many frames it
appears in.

WHY MERGING IS THE WHOLE JOB
    As the robot drives, the same lettuce appears in frame 4, frame 5 and
    frame 6. Three detections, one plant. Without merging your map grows a
    cluster of phantom duplicates everywhere the frames overlapped, and the
    plant count - the number you would actually report - is meaningless.

    The rule used here: a new detection within merge_radius_cm of an existing
    plant of the same kind IS that plant. Position becomes a
    confidence-weighted average, so repeated looks sharpen the estimate
    rather than just overwriting it.

SETTING merge_radius_cm - set it from your ERROR, not your crop spacing
    Repeat sightings of one plant scatter by roughly your position error. So
    the radius has to cover that scatter and nothing more:

        merge_radius_cm  ~  3 x your position error

    Run validate.py to get the error, then multiply. With 1.5 cm error, about
    4-5 cm is right. Measured on synthetic data with 1.5 cm scatter and 29
    plants, radius 5 cm recovered 26 and radius 10 cm recovered only 19.

    Do NOT size it off crop spacing. That was the obvious-looking rule and it
    is wrong: it makes the radius far larger than your error, and everything
    inside it fuses.

THE LIMIT THIS CREATES - worth a paragraph in the report
    Two genuinely different plants closer together than the merge radius get
    fused into one entry, and no amount of tuning fixes it. Shrinking the
    radius to separate them splits single plants into duplicates instead.

    So the system cannot resolve two plants closer than roughly 3x its
    position error. At 1.5 cm error that is about 5 cm. A weed growing within
    5 cm of a crop stem is absorbed into the crop and never gets weeded.

    That is not a bug in this file, it is the in-row weeding problem - the
    same wall the commercial machines hit. Tertill cannot weed in-row at all;
    FarmDroid claims only about half. The honest framing is that position
    accuracy IS in-row capability, which makes validate.py the most important
    script here.

CROP VS WEED
    This file does not ask the detector which is which, because the detector
    is bad at that - species names scored far worse than generic words in
    testing. Instead: detect vegetation confidently, then let geometry
    decide. Anything sitting on a seeded row is a crop; anything between the
    rows is a weed. That is classify_by_rows() below, and it is the core of
    the open-set argument - you never have to enumerate the weed species you
    might meet.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

# Categorical slots 1-3 from the validated reference palette. These three
# clear the all-pairs colour-vision gates in both light and dark modes, which
# matters here because a scatter puts every pair on screen at once. Colour is
# tied to the kind of plant, never to how many of each we happen to have.
COLOURS = {
    "crop":    {"light": "#2a78d6", "dark": "#3987e5"},   # slot 1, blue
    "weed":    {"light": "#eb6834", "dark": "#d95926"},   # slot 2, orange
    "unknown": {"light": "#1baf7a", "dark": "#199e70"},   # slot 3, aqua
}
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
INK_SOFT = {"light": "#52514e", "dark": "#c3c2b7"}


class Plant:
    """One plant in the garden. Possibly seen many times."""

    __slots__ = ("id", "x_cm", "y_cm", "kind", "label", "category",
                 "confidence", "sightings", "size_cm2", "_weight")

    def __init__(self, pid, x_cm, y_cm, kind="unknown", label="plant",
                 confidence=0.0, size_cm2=0.0):
        self.id = pid
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)
        self.kind = kind
        self.label = label
        self.category = ""          # filled from the classifier's CSV
        self.confidence = float(confidence)
        self.sightings = 1
        self.size_cm2 = float(size_cm2)
        self._weight = max(float(confidence), 1e-6)

    def absorb(self, x_cm, y_cm, confidence, size_cm2=0.0):
        """Fold in another sighting of this same plant."""
        w = max(float(confidence), 1e-6)
        total = self._weight + w

        self.x_cm = (self.x_cm * self._weight + x_cm * w) / total
        self.y_cm = (self.y_cm * self._weight + y_cm * w) / total
        self.size_cm2 = max(self.size_cm2, float(size_cm2))
        self.confidence = max(self.confidence, float(confidence))
        self.sightings += 1
        self._weight = total

    def as_dict(self):
        return {"id": self.id, "x_cm": round(self.x_cm, 2),
                "y_cm": round(self.y_cm, 2), "kind": self.kind,
                "label": self.label, "category": self.category,
                "confidence": round(self.confidence, 4),
                "sightings": self.sightings, "size_cm2": round(self.size_cm2, 1)}

    def __repr__(self):
        return (f"Plant#{self.id} {self.kind:7s} "
                f"({self.x_cm:6.1f}, {self.y_cm:6.1f})cm  "
                f"{self.confidence:5.1%}  x{self.sightings}")


class GardenMap:
    """
        gmap = GardenMap(merge_radius_cm=5)
        gmap.add(x, y, confidence=0.4)          # once per detection
        gmap.classify_by_rows(row_spacing_cm=30)
        gmap.save("garden_map.json")
        gmap.render("garden_map.png")
    """

    def __init__(self, merge_radius_cm=5.0):
        self.merge_radius_cm = float(merge_radius_cm)
        self.plants = []
        self.rows = None            # set by classify_by_rows, for the plot
        self._next_id = 1
        self.frames = 0
        self.rejected = 0           # detections whose ray missed the soil

    # -- building -----------------------------------------------------------

    def add(self, x_cm, y_cm, confidence=0.0, kind="unknown",
            label="plant", size_cm2=0.0):
        """Add one detection. Merges into a nearby plant if there is one."""
        if not (math.isfinite(x_cm) and math.isfinite(y_cm)):
            self.rejected += 1
            return None

        match, best = None, self.merge_radius_cm
        for p in self.plants:
            if p.kind != kind and "unknown" not in (p.kind, kind):
                continue
            d = math.hypot(p.x_cm - x_cm, p.y_cm - y_cm)
            if d < best:
                match, best = p, d

        if match is not None:
            match.absorb(x_cm, y_cm, confidence, size_cm2)
            if match.kind == "unknown" and kind != "unknown":
                match.kind = kind
            return match

        p = Plant(self._next_id, x_cm, y_cm, kind, label, confidence, size_cm2)
        self.plants.append(p)
        self._next_id += 1
        return p

    def add_frame(self, xy_garden, detections, areas_cm2=None):
        """
        Add a whole frame at once.

        xy_garden   (N, 2) array from GroundProjector.to_garden - NaN allowed
        detections  the matching list of detect.Detection
        areas_cm2   optional footprint area per detection, same order.
                    None entries just mean "not measured".
        Returns the Plant each detection went into, in the same order, with
        None where the detection was discarded. That mapping is what lets
        the caller tie a cropped image back to the plant it shows.
        """
        self.frames += 1
        xy = np.asarray(xy_garden, np.float64).reshape(-1, 2)
        if areas_cm2 is None:
            areas_cm2 = [None] * len(detections)

        landed = []
        for (x, y), d, a in zip(xy, detections, areas_cm2):
            landed.append(self.add(x, y, confidence=d.confidence,
                                   label=d.label, size_cm2=(a or 0.0)))
        return landed

    # -- classifying --------------------------------------------------------

    def classify_by_rows(self, row_spacing_cm, row_offset_cm=0.0,
                         tolerance_cm=7.5, axis="x"):
        """
        Decide crop vs weed from position alone.

        Rows are assumed straight, evenly spaced, and running along the
        garden's `axis`. A plant within tolerance_cm of a row centreline is a
        crop; everything else is a weed.

        tolerance_cm is the honest knob here. It has to cover your real
        position error plus how far a crop may have come up off-line. Too
        tight and you hoe your lettuces; too loose and in-row weeds survive.
        Measure your error with validate.py first, then set this.
        """
        across = "y" if axis == "x" else "x"
        self.rows = {"spacing_cm": float(row_spacing_cm),
                     "offset_cm": float(row_offset_cm),
                     "tolerance_cm": float(tolerance_cm),
                     "axis": axis}

        for p in self.plants:
            pos = getattr(p, f"{across}_cm")
            # Distance to the nearest row centreline.
            off = (pos - row_offset_cm) % row_spacing_cm
            gap = min(off, row_spacing_cm - off)
            p.kind = "crop" if gap <= tolerance_cm else "weed"

        return self.counts()

    # Delaney's BioCLIP pipeline returns one of these. Anything it is not
    # confident about is treated as a crop, because the two mistakes are
    # not equal: leaving a weed costs a weed, mulching a basil costs the
    # plant. When unsure, protect it.
    CROP_LIKE = ("crop", "possible crop", "uncertain", "unknown")

    def keep_categories(self, wanted, unknown_is_crop=True):
        """
        Drop every plant whose category is not in `wanted`.

        This is the A/B sort: the classifier says what each plant is, a rule
        turns that into a category, and the map shows only the categories
        you asked for.

        Plants with NO category are dropped too, and counted separately -
        an unclassified plant is not the same as one you decided to exclude,
        and quietly treating it as excluded would hide a broken handoff.

        Returns (kept, dropped_wrong_category, dropped_no_category).
        """
        wanted = {str(w).strip() for w in wanted if str(w).strip()}

        # Asking for "crop" means every category that might be one.
        if unknown_is_crop and "crop" in wanted:
            wanted |= set(self.CROP_LIKE)

        keep, wrong, blank = [], 0, 0

        for p in self.plants:
            if not p.category:
                # No category at all is different from an uncertain one -
                # it means the classifier never saw this plant. Keep it if
                # unknowns count as crops, but count it separately so a
                # half-filled CSV is still visible.
                blank += 1
                if unknown_is_crop:
                    keep.append(p)
            elif p.category in wanted:
                keep.append(p)
            else:
                wrong += 1

        self.plants = keep
        return len(keep), wrong, blank

    def categories(self):
        out = {}
        for p in self.plants:
            out[p.category or "(none)"] = out.get(p.category or "(none)", 0) + 1
        return out

    def counts(self):
        out = {"crop": 0, "weed": 0, "unknown": 0}
        for p in self.plants:
            out[p.kind] = out.get(p.kind, 0) + 1
        return out

    def bounds(self):
        if not self.plants:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p.x_cm for p in self.plants]
        ys = [p.y_cm for p in self.plants]
        return (min(xs), min(ys), max(xs), max(ys))

    def weeds(self):
        """Weeding targets, nearest the garden origin first."""
        w = [p for p in self.plants if p.kind == "weed"]
        w.sort(key=lambda p: math.hypot(p.x_cm, p.y_cm))
        return w

    # -- the grid -----------------------------------------------------------

    def grid(self, cols=6, rows=4, bed=None, snap_cm=4.0):
        """
        Divide the bed into cols x rows cells and say what is in each.

        Returns a list of cells, each a dict:
            row, col      0-based. row 0 is the BOTTOM of the map (low y),
                          col 0 is the LEFT (low x)
            name          "C3R2" - the human label used on the plot
            x_cm, y_cm    centre of the cell. This is the waypoint your robot
                          drives to
            x0,y0,x1,y1   the cell's edges
            kind          "weed" if any weed is inside, else "crop" if any
                          crop is, else "empty"
            count         plants inside
            crops, weeds  counts by kind
            plant_ids     which plants, so you can get back to exact positions

        `bed` is (x0, y0, x1, y1) in centimetres. Give it if you know the real
        extent of your bed - the grid should describe the BED, not just the
        part you happened to photograph. Left out, it is inferred from the
        plants found, padded slightly, which shifts if you scan a different
        area. For anything you report, pass the real numbers.
        """
        if bed is None:
            x0, y0, x1, y1 = self.bounds()
            px = max(5.0, 0.06 * max(x1 - x0, 1.0))
            py = max(5.0, 0.06 * max(y1 - y0, 1.0))
            x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
        else:
            x0, y0, x1, y1 = (float(v) for v in bed)

        w = (x1 - x0) / cols
        h = (y1 - y0) / rows

        cells = []
        for r in range(rows):
            for c in range(cols):
                cx0, cx1 = x0 + c * w, x0 + (c + 1) * w
                cy0, cy1 = y0 + r * h, y0 + (r + 1) * h

                # Edge cells reach a little past the plot. A plant 1 cm
                # outside a measured bed is a corner-click that landed a
                # few pixels high, not a plant in the next bed - and
                # dropping it silently loses a real crop from the map.
                lo_x = cx0 - snap_cm if c == 0 else cx0
                hi_x = cx1 + snap_cm if c == cols - 1 else cx1
                lo_y = cy0 - snap_cm if r == 0 else cy0
                hi_y = cy1 + snap_cm if r == rows - 1 else cy1

                inside = [p for p in self.plants
                          if lo_x <= p.x_cm < hi_x and lo_y <= p.y_cm < hi_y]
                # The far edges are closed, so plants exactly on the boundary
                # are not silently dropped off the grid.
                if c == cols - 1:
                    inside += [p for p in self.plants
                               if p.x_cm == cx1 and cy0 <= p.y_cm < cy1]
                if r == rows - 1:
                    inside += [p for p in self.plants
                               if cx0 <= p.x_cm <= cx1 and p.y_cm == cy1]
                inside = list({id(p): p for p in inside}.values())

                nw = sum(1 for p in inside if p.kind == "weed")
                nc = sum(1 for p in inside if p.kind == "crop")

                # A cell with ANY weed is a weeding target, even if a crop is
                # in there too - it still has to be visited. A cell holding
                # only unclassified plants is "unknown", not empty: something
                # is growing there, we just have not decided what it is.
                # (Nothing is classified until classify_by_rows runs, so
                # without --rows every occupied cell lands here.)
                if nw:
                    kind = "weed"
                elif nc:
                    kind = "crop"
                elif inside:
                    kind = "unknown"
                else:
                    kind = "empty"

                cells.append({
                    "row": r, "col": c, "name": f"C{c+1}R{r+1}",
                    "x_cm": (cx0 + cx1) / 2, "y_cm": (cy0 + cy1) / 2,
                    "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1,
                    "kind": kind,
                    "count": len(inside), "crops": nc, "weeds": nw,
                    "plant_ids": sorted(p.id for p in inside),
                })

        return cells

    def grid_summary(self, cols=6, rows=4, bed=None):
        cells = self.grid(cols, rows, bed)
        occupied = [c for c in cells if c["kind"] != "empty"]
        targets = [c for c in cells if c["kind"] == "weed"]
        unknown = [c for c in cells if c["kind"] == "unknown"]
        placed = sum(c["count"] for c in cells)

        lines = [
            f"Grid            : {cols} x {rows} = {cols * rows} cells",
            f"Cell size       : {cells[0]['x1'] - cells[0]['x0']:.1f} x "
            f"{cells[0]['y1'] - cells[0]['y0']:.1f} cm",
            f"Occupied        : {len(occupied)} of {len(cells)}",
            f"Weeding targets : {len(targets)} cell(s)"
            + (f"  -> {', '.join(c['name'] for c in targets)}" if targets else ""),
        ]
        if unknown:
            lines.append(f"Unclassified    : {len(unknown)} cell(s) - run with "
                         f"--rows to split crop from weed")
        if placed < len(self.plants):
            lines.append(f"WARNING         : {len(self.plants) - placed} plant(s) "
                         f"fell outside the grid")
            lines.append(f"                  Either the plot really ends "
                         f"before them, or the near")
            lines.append(f"                  corners were clicked a little "
                         f"high. Check the corner")
            lines.append(f"                  overlay - their bases must be "
                         f"inside the green outline.")
        return "\n".join(lines)

    def render_grid(self, path="garden_grid.png", cols=6, rows=4, bed=None,
                    mode="light", show_plants=True, title="Garden grid",
                    dpi=200):
        """
        The bed as a block of cells, coloured in where plants are.

        A cell holding any weed is a weeding target and takes the weed
        colour, because that is the actionable state - a cell with a weed
        tucked beside a lettuce still has to be visited.

        Each cell carries its plant count as a direct label, so the map never
        relies on colour alone to be read.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
        except ImportError:
            print("matplotlib not installed - skipping the grid plot.")
            print("  python -m pip install matplotlib")
            return None

        surface, ink, soft = SURFACE[mode], INK[mode], INK_SOFT[mode]
        empty_fill = "#f0efec" if mode == "light" else "#383835"
        cells = self.grid(cols, rows, bed)

        gx0, gy0 = cells[0]["x0"], cells[0]["y0"]
        gx1, gy1 = cells[-1]["x1"], cells[-1]["y1"]
        span_x, span_y = gx1 - gx0, gy1 - gy0
        scale = 7.5 / max(span_x, span_y, 1.0)

        fig, ax = plt.subplots(figsize=(max(span_x * scale, 4.0),
                                        max(span_y * scale, 3.0)))
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)

        # A 2px surface gap between fills keeps neighbouring cells distinct.
        gap_x, gap_y = span_x / cols * 0.012, span_y / rows * 0.012

        for cell in cells:
            filled = cell["kind"] != "empty"
            face = COLOURS[cell["kind"]][mode] if filled else empty_fill

            ax.add_patch(Rectangle(
                (cell["x0"] + gap_x, cell["y0"] + gap_y),
                (cell["x1"] - cell["x0"]) - 2 * gap_x,
                (cell["y1"] - cell["y0"]) - 2 * gap_y,
                facecolor=face, edgecolor=surface, linewidth=2.0,
                zorder=2, alpha=1.0 if filled else 0.55))

            if filled:
                ax.text(cell["x_cm"], cell["y_cm"],
                        str(cell["count"]), ha="center", va="center",
                        fontsize=15, color="#ffffff", zorder=5)

            ax.text(cell["x0"] + gap_x * 2.5, cell["y1"] - gap_y * 2.5,
                    cell["name"], ha="left", va="top", fontsize=6.5,
                    color="#ffffff" if filled else soft,
                    alpha=0.85 if filled else 0.6, zorder=5)

        # Actual plant positions on top, so the grid can be checked against
        # the thing it is summarising.
        if show_plants and self.plants:
            for kind in ("crop", "weed", "unknown"):
                pts = [p for p in self.plants if p.kind == kind]
                if not pts:
                    continue
                ax.scatter([p.x_cm for p in pts], [p.y_cm for p in pts],
                           s=34, c=surface, edgecolors=soft, linewidths=1.0,
                           marker="o" if kind != "weed" else "X",
                           zorder=6, alpha=0.9)

        ax.set_xlim(gx0, gx1)
        ax.set_ylim(gy0, gy1)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("x from corner stake  (cm)", fontsize=9, color=soft)
        ax.set_ylabel("y from corner stake  (cm)", fontsize=9, color=soft)

        n_occ = sum(1 for c in cells if c["kind"] != "empty")
        n_weed = sum(1 for c in cells if c["kind"] == "weed")

        ax.set_title(title, fontsize=12, color=ink, loc="left", pad=14)
        ax.text(0.0, 1.015,
                f"{n_occ} of {len(cells)} cells occupied   -   "
                f"{n_weed} weeding target(s)   -   "
                f"cell {cells[0]['x1'] - cells[0]['x0']:.0f} x "
                f"{cells[0]['y1'] - cells[0]['y0']:.0f} cm",
                transform=ax.transAxes, fontsize=8.5, color=soft)

        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(soft)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=soft, labelsize=8)

        present = [k for k in ("crop", "weed", "unknown")
                   if any(c["kind"] == k for c in cells)]
        handles = [Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=COLOURS[k][mode],
                          markeredgecolor=surface, markersize=11,
                          label="Weed present" if k == "weed" else k.capitalize())
                   for k in present]
        handles.append(Line2D([0], [0], marker="s", color="none",
                              markerfacecolor=empty_fill,
                              markeredgecolor=surface, markersize=11,
                              label="Empty"))
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=8.5, labelcolor=soft, handletextpad=0.5)

        fig.tight_layout()
        fig.savefig(path, dpi=dpi, facecolor=surface, bbox_inches="tight")
        plt.close(fig)
        return path

    def write_grid_csv(self, path="garden_grid.csv", cols=6, rows=4, bed=None):
        """
        The grid as a table - and as your robot's waypoint list.

        x_cm and y_cm are cell centres, which is where the robot drives.
        Filter to kind == weed to get the visit list.
        """
        out = ["cell,col,row,x_cm,y_cm,kind,count,crops,weeds,plant_ids"]
        for c in self.grid(cols, rows, bed):
            ids = " ".join(str(i) for i in c["plant_ids"])
            out.append(f"{c['name']},{c['col']+1},{c['row']+1},"
                       f"{c['x_cm']:.2f},{c['y_cm']:.2f},{c['kind']},"
                       f"{c['count']},{c['crops']},{c['weeds']},{ids}")
        pathlib.Path(path).write_text("\n".join(out) + "\n")
        return path

    # -- persistence --------------------------------------------------------

    def save(self, path="garden_map.json"):
        pathlib.Path(path).write_text(json.dumps({
            "merge_radius_cm": self.merge_radius_cm,
            "frames": self.frames,
            "rejected": self.rejected,
            "rows": self.rows,
            "counts": self.counts(),
            "plants": [p.as_dict() for p in self.plants],
        }, indent=2))
        return path

    @classmethod
    def load(cls, path="garden_map.json"):
        d = json.loads(pathlib.Path(path).read_text())
        g = cls(d.get("merge_radius_cm", 5.0))
        g.frames = d.get("frames", 0)
        g.rejected = d.get("rejected", 0)
        g.rows = d.get("rows")

        for r in d["plants"]:
            p = Plant(r["id"], r["x_cm"], r["y_cm"], r["kind"],
                      r.get("label", "plant"), r.get("confidence", 0.0),
                      r.get("size_cm2", 0.0))
            p.category = r.get("category", "")
            p.sightings = r.get("sightings", 1)
            g.plants.append(p)
            g._next_id = max(g._next_id, r["id"] + 1)
        return g

    def summary(self):
        c = self.counts()
        b = self.bounds()
        once = sum(1 for p in self.plants if p.sightings == 1)

        lines = [
            f"Frames processed : {self.frames}",
            f"Plants mapped    : {len(self.plants)}"
            f"   (crop {c['crop']}, weed {c['weed']}, unclassified {c['unknown']})",
            f"Extent           : x {b[0]:.0f} to {b[2]:.0f} cm, "
            f"y {b[1]:.0f} to {b[3]:.0f} cm",
        ]
        if self.rejected:
            lines.append(f"Discarded        : {self.rejected} detection(s) "
                         f"whose ray missed the soil")
        if self.plants and self.frames > 1:
            lines.append(f"Seen once only   : {once} of {len(self.plants)}"
                         f"  (high here means merging or poses need a look)")

        sized = [p for p in self.plants if p.size_cm2 > 0]
        if sized:
            lines.append("")
            lines.append("Footprint on the ground:")
            for p in sorted(sized, key=lambda q: -q.size_cm2):
                across = 2.0 * math.sqrt(p.size_cm2 / math.pi)
                lines.append(f"  #{p.id:<3d} {p.label:10s} "
                             f"{p.size_cm2:8.1f} cm2 "
                             f"  {across:5.1f} cm across")
        return "\n".join(lines)

    # -- the picture --------------------------------------------------------

    def render(self, path="garden_map.png", mode="light", label_weeds=True,
               title="Garden map", dpi=200, bed=None):
        """
        Top-down map of the bed.

        bed is (x0, y0, x1, y1) in cm - the real extent of the plot. Pass it
        and the picture comes out the shape of the plot, so a 94 x 30 cm bed
        draws wide and shallow like the bed itself. Leave it out and the
        picture is cropped to whatever plants were found, which changes shape
        from run to run.

        Equal aspect either way, because this is a map - a stretched axis
        would make a square bed look rectangular and quietly misreport the
        geometry.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("matplotlib not installed - skipping the plot.")
            print("  python -m pip install matplotlib")
            return None

        surface, ink, soft = SURFACE[mode], INK[mode], INK_SOFT[mode]

        if bed is not None:
            x0, y0, x1, y1 = (float(v) for v in bed)
            pad = 0.0
        else:
            x0, y0, x1, y1 = self.bounds()
            pad = max(10.0, 0.08 * max(x1 - x0, y1 - y0, 1.0))

        span_x, span_y = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
        scale = 7.5 / max(span_x, span_y, 1.0)

        # Keep the figure in the plot's own proportions. Clamping each side
        # to a minimum would quietly square up a long thin bed.
        fig, ax = plt.subplots(figsize=(max(span_x * scale, 2.2),
                                        max(span_y * scale, 1.6)))
        fig.patch.set_facecolor(surface)
        ax.set_facecolor(surface)

        # Seeded rows, drawn first so plants sit on top of them.
        if self.rows:
            r = self.rows
            lo, hi = (y0 - pad, y1 + pad) if r["axis"] == "x" else (x0 - pad, x1 + pad)
            n0 = math.floor((lo - r["offset_cm"]) / r["spacing_cm"])
            n1 = math.ceil((hi - r["offset_cm"]) / r["spacing_cm"])

            for n in range(n0, n1 + 1):
                c = r["offset_cm"] + n * r["spacing_cm"]
                if not (lo <= c <= hi):
                    continue
                band = (ax.axhspan if r["axis"] == "x" else ax.axvspan)
                band(c - r["tolerance_cm"], c + r["tolerance_cm"],
                     color=COLOURS["crop"][mode], alpha=0.10, lw=0, zorder=1)
                line = (ax.axhline if r["axis"] == "x" else ax.axvline)
                line(c, color=COLOURS["crop"][mode], lw=1.0,
                     alpha=0.45, zorder=2)

        # Plants. A surface-coloured ring keeps overlapping marks readable.
        for kind in ("crop", "weed", "unknown"):
            pts = [p for p in self.plants if p.kind == kind]
            if not pts:
                continue
            ax.scatter([p.x_cm for p in pts], [p.y_cm for p in pts],
                       s=[90 + 160 * min(p.confidence, 1.0) for p in pts],
                       c=COLOURS[kind][mode], edgecolors=surface,
                       linewidths=2.0, zorder=4 if kind == "weed" else 3,
                       marker="o" if kind != "weed" else "X")

        # Label weeds only - they are the actionable set, and a number on
        # every mark would bury the map.
        if label_weeds:
            for p in self.weeds():
                ax.annotate(f"{p.id}", (p.x_cm, p.y_cm),
                            textcoords="offset points", xytext=(9, 5),
                            fontsize=7.5, color=soft, zorder=5)

        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("x from corner stake  (cm)", fontsize=9, color=soft)
        ax.set_ylabel("y from corner stake  (cm)", fontsize=9, color=soft)

        c = self.counts()
        ax.set_title(title, fontsize=12, color=ink, loc="left", pad=14)
        ax.text(0.0, 1.015,
                f"{len(self.plants)} plants from {self.frames} frame(s)   -   "
                f"{c['weed']} weeding target(s)",
                transform=ax.transAxes, fontsize=8.5, color=soft)

        ax.grid(True, lw=0.5, alpha=0.18, color=soft)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(soft)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=soft, labelsize=8)

        present = [k for k in ("crop", "weed", "unknown")
                   if any(p.kind == k for p in self.plants)]
        if len(present) >= 2:
            ax.legend(handles=[
                Line2D([0], [0], marker="X" if k == "weed" else "o",
                       color="none", markerfacecolor=COLOURS[k][mode],
                       markeredgecolor=surface, markeredgewidth=1.5,
                       markersize=9, label=k.capitalize())
                for k in present],
                loc="upper right", frameon=False, fontsize=8.5,
                labelcolor=soft, handletextpad=0.4)

        fig.tight_layout()
        fig.savefig(path, dpi=dpi, facecolor=surface, bbox_inches="tight")
        plt.close(fig)
        return path

    def write_csv(self, path="garden_map.csv"):
        """Flat table - for the report, or for pasting into a spreadsheet."""
        rows = ["id,kind,category,label,x_cm,y_cm,confidence,sightings,size_cm2"]
        for p in sorted(self.plants, key=lambda q: (q.kind, q.id)):
            rows.append(f"{p.id},{p.kind},{p.category},{p.label},"
                        f"{p.x_cm:.2f},{p.y_cm:.2f},"
                        f"{p.confidence:.4f},{p.sightings},{p.size_cm2:.1f}")
        pathlib.Path(path).write_text("\n".join(rows) + "\n")
        return path
