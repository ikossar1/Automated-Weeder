"""
Hand-jog the machine, from the mobility folder.

The script itself lives in camera/move_camera.py, because framing a
photo is what it was written for. This is a shim so it can be run from
here too without changing folders - one copy, so the pin numbers
cannot drift apart between two files that both claim to drive the same
hardware.

    python3 move.py
"""
import pathlib
import runpy
import sys

TARGET = pathlib.Path(__file__).resolve().parent.parent / "camera" / "move_camera.py"

if not TARGET.exists():
    sys.exit(f"Cannot find {TARGET}. Has camera/ been pulled?")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
