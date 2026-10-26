# Automated Weeder

Senior design, Tufts University. A photo of a garden bed goes in; instructions
for a mulch sprayer come out.

Two halves, joined by one file:

```
mapping/    photo  ->  map of where the plants are      (maps/<name>/<name>.json)
                              |
spraying/                     v  ->  motor and valve commands  (dc_commands.csv)
```

Nothing is imported across that line, so the two can be worked on
independently.

---

## Setup, once

```bash
python -m pip install -r requirements.txt
```

Needs Python 3.10+. `ultralytics` pulls in torch, which is a large download.

---

## The guided way

```bash
python run.py
```

Asks for everything it needs, stops twice to let you check the work, and
writes the commands at the end. Nothing moves — it only produces a file.

```
photo  ->  map  ->  [you approve]  ->  route  ->  [you approve]  ->  commands
```

**Approve nothing you have not looked at.** Both halves can produce a
confident, wrong answer with no error:

- A badly clicked corner puts every plant in the wrong place, and the maths
  cannot tell. Check `<name>_corners.png`.
- A wrong plot size, rail length or throw plans a route the machine cannot
  drive. Check `spray_route.png`.

---

## The manual way, step by step

### 1. New photo

```bash
cd mapping
python new_bed.py
```

Lists the photos it can see, asks which one, asks the plot size, opens the
corner picker, then runs the scan.

### 2. Corners, if you are doing it by hand

```bash
python pick_corners.py photos/mybed.jpg 30.5 35
```

A window opens. Click the four corners of the **soil**, in this order:

```
1. near-left    2. near-right    3. far-right    4. far-left
```

Near means closest to the camera, at the bottom of the picture. It prints
eight numbers.

### 3. Map the bed

Running our own detector:

```bash
python scan.py photos/mybed.jpg \
    --corners <eight numbers> \
    --plot 30.5 35 \
    --prompts "plant,leaf" --conf 0.25 --imgsz 1600 \
    --merge 4 --grid 5x6 --crops \
    --out mybed
```

Or using Delaney's BioCLIP output instead, which carries her species names
and crop/not-crop call:

```bash
python scan.py photos/mybed.jpg \
    --results delaney_output/results.csv \
    --corners <eight numbers> --plot 30.5 35 \
    --merge 4 --grid 5x6 --crops --out mybed
```

**Open `maps/mybed/mybed_corners.png` first.** If the green outline is not
sitting on your plot, every centimetre below it is wrong.

Optional, once species come back labelled:

```bash
python scan.py ... --species handoff/mybed/crops.csv --only crop
```

### 4. Plan the spray route

```bash
cd ../spraying
python simulate.py ../mapping/maps/mybed/mybed.json \
    --bed 0 0 30.5 35 \
    --side left --rail-length 35 \
    --throw 30.5 --height 14 \
    --tilt-min 18 --tilt-max 80 \
    --offset-across 3.6 \
    --clearance 2 \
    --continuous --cell 1.5 \
    --grid 5x6 --grid-only \
    --out plans/mybed
```

**Open `plans/mybed/spray_route.png`.** One unbroken blue line that never
enters an orange box is what you want.

### 5. Write the commands

```bash
python motors.py plans/mybed/plan.json --dc
python motor_sim.py dc_commands.csv --plan plans/mybed/plan.json
```

The second one reports how long the run takes, which axis is the bottleneck,
and whether valve timing can put mulch on a plant.

### 6. Check the aiming maths after any change to aim.py

```bash
python verify_aim.py
```

27 checks, no hardware needed.

---

## The numbers you must measure

Every one of these is a physical measurement. Wrong values produce a clean
map and a clean route that are both fiction.

| Setting | What it is | Ours |
|---|---|---|
| `--plot W D` | plot size in cm, near edge then depth | `30.5 35` |
| `--rail-length` | how far the carriage actually travels, cm | `35` |
| `--throw` | how far the sprayer reaches, cm | `30.5` |
| `--height` | nozzle height above the soil, cm | `14` |
| `--offset-across` | camera lens to nozzle, cm | `3.6` |
| `--tilt-min` / `--tilt-max` | tilt travel, degrees | `18` / `80` |
| `--clearance` | keep-off margin round each plant, cm | `2` |

Two that are easy to get wrong:

**Reach is `height / tan(tilt_min)`, not the throw.** A nozzle that cannot aim
shallow enough will not reach the far side however hard it blows.
`simulate.py` prints both and says which one binds.

**`--clearance` should be at least half your spray width.** With the valve
held open the whole run, this standoff is the only thing keeping the stream
off a plant.

---

## Stopping a run

### Stages 1&ndash;3 &mdash; calibration, mapping, spraying

**`Ctrl + C`** in the terminal. Or press **`q`** to close the corner-picking
or box-picking window.

Nothing has moved. These stages only write files, so the worst case is a
part-written output you can delete and redo. The scripts catch the interrupt
and say what was finished:

```
STOPPED  -  spray planning was interrupted

Finished before the stop:
  plans/bed1/plan.json

Nothing has moved - these stages only produce files.
```

The command file is written **atomically** &mdash; to a temporary name, then
renamed over the target. A rename either happens or it does not, so a `Ctrl + C`
mid-write cannot leave a truncated CSV that still looks like a valid plan. A
machine handed one of those would run half a plan and stop somewhere
arbitrary, possibly with the valve open, and nothing in the file would say it
was incomplete.

### Stage 4 &mdash; the machine is moving

**Shut the valve first. Then stop the motors.**

That order is not negotiable. Stop the motors first and the valve keeps
running onto one spot &mdash; a second of that is a heap of mulch on whatever
is underneath. Shut it first and the worst case is a short line of mulch
where the head was already travelling.

This belongs in the Raspberry Pi code; nothing on the laptop can reach the
hardware once the file has been handed over.

### Build a physical stop

`Ctrl + C` needs a laptop, a cable and a Python process all still working. A
switch in the air line or the motor supply does not.

Treat the software abort as the convenience and the switch as the actual
safety case.

---

## For the Raspberry Pi code

**You do not need any of the above.** The vision half runs on a laptop and
produces one file. That file is the entire interface.

### `dc_commands.csv`

One row per move. Columns:

| Column | Meaning |
|---|---|
| `i` | step number |
| `rail_cm`, `pan_deg`, `tilt_deg` | the same move in real units, for debugging |
| `rail_counts`, `pan_counts`, `tilt_counts` | **encoder targets** — drive each axis here |
| `rail_dir`, `pan_dir`, `tilt_dir` | `+1`, `-1` or `0` — H-bridge direction |
| `rail_secs`, `pan_secs`, `tilt_secs` | open-loop run time, if there are no encoders |
| `spray` | `0` or `1` — valve state for this move |

**With encoders** (recommended): run a PID loop per axis driving the encoder
count toward `*_counts`. `*_dir` says which way to start.

**Without encoders**: drive `*_dir` for `*_secs`. Treat the durations as
rough — the same time travels further on a warm motor or a fresh battery.

### Three axes

```
rail   linear, along x       carries the head
pan    rotates about z (up)  swings the aim left and right
tilt   rotates about x       raises and lowers the aim, which sets range
```

The maths assumes **the tilt stage is mounted on the pan stage**, so tilt
pivots in the nozzle's own frame after pan has turned it. That is how most
pan-tilt heads are built, and it is what keeps the two axes independent.
Check it against the actual bracket.

### The valve

`spray` changes **once** in a well-planned run. The route is laid out so the
aim never crosses a plant, which means there is nothing to gate — the valve
opens at the start of a patch of soil and closes at the end of it.

If `spray` flips often, something is wrong with the plan, not the hardware.

A valve has lag. Command it **early** by its own open and close time so the
lag is absorbed rather than accumulated. `motor_sim.py --plan` reports how
far the aim point travels during that lag.

### Gear ratios and limits live in one file

`spraying/motors.py`, four numbers per axis:

```python
rail = DCAxis("rail", counts_per_unit=120.0, lo=0.0,  hi=300.0, units_per_second=8.0)
pan  = DCAxis("pan",  counts_per_unit=12.0,  lo=-90.0, hi=90.0, units_per_second=45.0)
tilt = DCAxis("tilt", counts_per_unit=12.0,  lo=5.0,   hi=85.0, units_per_second=45.0)
```

These absorb gearing, encoder resolution, and a motor wired backwards
(`invert=True`). Out-of-travel targets are **refused, not clamped** — a
clamped command would park the head at the end of the rail and keep going.

**Every number there is a placeholder.** Measure yours.

---

## Pi 5 note

`RPi.GPIO` does not work on the Pi 5 — GPIO moved behind the RP1 chip. Use
`lgpio` or `gpiozero`. Many stepper and H-bridge tutorials still assume
`RPi.GPIO` and will fail in confusing ways.

---

## What is proven and what is not

**Proven.** The aiming maths: `aim()` and `spray_point()` are exact inverses,
checked over 4,000 random targets to 3e-13 cm. The pipeline reproduces
exactly — same photo in, same numbers out. The planner never routes over a
plant.

**Not proven.** That any centimetre matches the real world. Nobody has yet
put a marker at a measured spot and checked the code agrees. `mapping/validate.py`
exists to do exactly that and has not been run on real data.

Those are separate claims. Keep them separate when writing this up.

---

## Licence note

Ultralytics is **AGPL-3.0**. Fine for coursework and a public project repo;
if any of this were distributed commercially those terms would reach the code
using it.
