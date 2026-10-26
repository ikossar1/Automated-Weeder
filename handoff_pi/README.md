# Running the mulch depositor on the Pi

Two files. `run_plan.py` drives the motors, `dc_commands.csv` tells it
where to go. Nothing else is needed - the driver uses only Python's
standard library plus `RPi.GPIO`.

Planning happens on a laptop (OpenCV, YOLOE and BioCLIP all live there).
The Pi only executes. When the plan changes, only `dc_commands.csv` is
resent.

## Wiring

Matches the original control script, BCM numbering:

| Part  | Pins   | Direction                                  |
|-------|--------|--------------------------------------------|
| Rail  | 23, 24 | pin 23 high drives toward +x (right)       |
| Pan   | 27, 22 | pin 27 high rotates clockwise from above   |
| Tilt  | 5, 6   | pin 5 high raises the nozzle               |
| Valve | 13     | high opens                                 |

## First time

Check the GPIO library is there:

    python3 -c "import RPi.GPIO; print('GPIO ok')"

If it errors:

    sudo apt install -y python3-rpi-lgpio

## Every run

1. Rehearse first. This touches no pins and takes a second:

       python3 run_plan.py dc_commands.csv --dry-run

   Read the summary it prints. If it says NOTHING WILL BE SPRAYED, the
   plan is bad - do not run it, ask for a new one.

   Add `--real-time` to rehearse at true speed and see how long the real
   run takes.

2. Measure the rail speed once, with the rail clear to move end to end:

       python3 run_plan.py --calibrate-rail

   Put the number it prints into `SPEED["rail"]` near the top of
   `run_plan.py`. Every distance in the run depends on this one value,
   and until it is measured it is a guess.

3. Run it:

       python3 run_plan.py dc_commands.csv

Ctrl+C aborts at any point. The valve closes before the motors stop, so
it cannot keep spraying while coasting.

## How it stays accurate without encoders

The motors have no position feedback, so every move is open loop: a
direction and a duration. Error would pile up across a run, so the rail
drives into its hard stop before each box and starts from a known place.
That is the 12 seconds of homing in the summary - it is not wasted time,
it is what keeps the aim honest.

Pan and tilt have no hard stop. They are set to a neutral datum by hand
before the run and limited in software to +/-90 and 80 degrees.

Moves shorter than 0.15 s are skipped rather than attempted, because a
DC motor cannot do a pulse that short with any accuracy. A skipped move
leaves the aim at most 1 cm off, against a 5.1 cm spray patch, so the
ground is still covered.
