# Garden mapping — pixels to a map of the bed

Turns YOLOE detections into a top-down map of the garden floor, in
centimetres from a stake you put in one corner.

No depth camera, no second camera, no 3D model. One assumption does the work:
**the soil is flat, and every plant grows out of it.**

---

## Why there is no neural network in here

A pixel is a *direction*, not a place. That is why one photo looks like it has
lost a dimension, and it is why people reach for depth models.

But you already know where the plant is in that missing dimension: on the
ground. So follow the direction out from the lens until it crosses the soil,
and stop. One ray, one surface, one answer.

Models that recover 3D from multiple photos do exist and they are good
— [VGGT](https://huggingface.co/papers/2503.11651) is the current best. They
are the wrong tool here for four reasons:

- They need a GPU. Your Pi 5 cannot run them near real time.
- They return shape, not **metres** — you would still need the ground plane
  to pin down scale.
- Their main value is recovering *unknown* camera poses. Yours are known,
  because your own motors set them.
- Independent 2025 evaluation still puts their pose accuracy below classical
  photogrammetry.

You would add a GPU-class dependency to get a less accurate answer than
trigonometry gives you.

---

## The files

| File | What it does |
|---|---|
| `geometry.py` | The core. Pixel → position on the soil. |
| `detect.py` | Wraps YOLOE, returns the pixel where each stem meets the ground. |
| `garden_map.py` | Accumulates detections across frames, merges repeats, draws the map. |
| `scan.py` | The driver — point it at a folder of images. |
| `calibrate.py` | Measures your lens, once. Run before quoting numbers. |
| `validate.py` | **Proves the positions are right.** The one that matters for the report. |

---

## Quick start

```bash
python -m pip install ultralytics opencv-python matplotlib
```

Run it on one photo with a rough lens guess, just to see it work:

```bash
python scan.py photos/ --hfov 70 --height 35 --tilt 60 --prompts plant
```

You get `garden_map.json`, `garden_map.csv` and `garden_map.png`.

Then do it properly:

```bash
python calibrate.py calib_photos/
python validate.py truth.csv --height 35 --tilt 60
python scan.py photos/ --height 35 --tilt 60 --poses poses.csv --rows 30
```

---

## The three things you must measure

Everything else is code. These are physical and nobody can supply them for you.

**1. Camera height.** Lens to soil. A direct multiplier on every distance
— 5% off here is 5% off everywhere.

**2. Tilt angle.** 90° is straight down. Steeper is more forgiving of both
angle error and uneven soil, at the cost of seeing less per frame.

**3. Robot pose per frame.** Which image was taken from where. Comes off your
own motor commands. Log it as you capture — reconstructing it later is
miserable.

---

## Two results worth putting in the report

### Generic prompts beat species names

Measured on your own test images:

| Prompt | Confidence |
|---|---|
| `"plant"` | **0.408** |
| `"lettuce"` | 0.135 |
| `"flower"` | **0.24** |
| `"poppy"` | 0.00 |

Three times better from the vaguer word. This is the evidence for the
crop-centric architecture: **detect vegetation confidently, then let row
geometry decide crop from weed.** You never have to enumerate weed species,
which is what makes it an open-set solution.

`classify_by_rows()` is that step.

### Position accuracy *is* in-row weeding capability

Repeat sightings of one plant scatter by your position error, so the merge
radius has to cover that scatter — about **3× your measured error**. Set it
from `validate.py` output, never from crop spacing.

That creates a hard floor: **two plants closer together than the merge radius
fuse into one, and tuning cannot fix it.** Shrink the radius and single plants
split into duplicates instead.

At 1.5 cm error that floor is about 5 cm. A weed within 5 cm of a crop stem is
absorbed into the crop and never weeded.

That is not a bug — it is the in-row weeding problem, the same wall the
commercial machines hit. Tertill cannot weed in-row at all; FarmDroid claims
roughly half. Framing it as *position accuracy sets in-row capability* turns
`validate.py` from a test script into your central performance argument.

---

## The grid view

The bed as a block of cells, coloured in wherever a plant sits. Default is
**6 columns x 4 rows = 24 cells**.

```bash
python scan.py photos/ --height 35 --tilt 60 --rows 30        --grid 6x4 --bed 0 -10 130 70
```

Writes `garden_map_grid.png` and `garden_map_grid.csv`.

Cells are named `C<col>R<row>`, counting from the bottom-left — `C1R1` is the
near-left corner, `C6R4` the far-right.

A cell holding **any** weed takes the weed colour, even if a crop is in there
too. That is the actionable state: the cell still has to be visited. The plant
count is printed in each cell, so the map never depends on colour alone, and
the real plant positions are dotted on top so you can check the grid against
what it is summarising.

### Pass `--bed` if you know your real dimensions

Left out, the grid is fitted to whatever plants were found — so it shifts and
resizes every time you scan a different area, and `C3R2` stops meaning the same
patch of soil between runs. Give the real extent and the cells stay put.

### The grid is your waypoint list

`garden_map_grid.csv` has a centre coordinate per cell. Filter to
`kind == weed` and you have the robot's visit list:

```
C5R1  drive to (97.5,  0.0) cm
C2R2  drive to (32.5, 20.0) cm
C6R4  drive to (119.2, 60.0) cm
```

Which connects this straight to your navigation prototype — the thing it has
to do is drive to an x/y coordinate, and these are the coordinates.

### Pick the cell size deliberately

A cell is a "go here and weed" instruction, so its size is really a **tool
width** decision. 6x4 over a 130 x 80 cm bed gives 22 x 20 cm cells. If your
weeding tool clears 10 cm, a cell holding one weed and one lettuce cannot be
cleared without a finer move, and you are back to needing the exact position
from `garden_map.csv`.

Reasonable framing for the report: the **grid** is the coarse plan of where to
go, and the **exact positions** are what the tool uses once it arrives.

---

## Known limits

- **Flat soil only.** A bump of height *h* reads as a position error of about
  `h / tan(tilt)`. At 45° that is 1:1. Steep tilt shrinks it.
- **Uncalibrated intrinsics are rough.** `--hfov` ignores lens distortion,
  which is worst at the frame edges — exactly where your far plants are.
- **Straight, evenly spaced rows.** `classify_by_rows` assumes it. Curved or
  hand-sown beds need a different classifier.
- **No tracking.** Merging is by position alone. If two plants are closer than
  the merge radius they fuse, regardless of how they look.

---

## Coordinate frames

```
image    pixels. u across, v down, origin top-left.
camera   OpenCV. x right, y down, z out of the lens.
robot    origin on the soil under the robot's centre.
         x forward, y left, z up. Soil is z = 0.
garden   origin at your corner stake. x, y along the ground.
         This is the frame every reported number is in.
```

`tilt` is measured down from horizontal, `pan` is positive to the left, and
`heading` is counter-clockwise from the garden's +x axis.
