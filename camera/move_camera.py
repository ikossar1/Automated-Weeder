"""
Hand-jog the machine, from the camera folder.

The script itself lives in mobility/move_camera.py, since driving
motors is a mobility job. This is a shim so framing a photo does not
mean changing folders - one copy, so the pin numbers cannot drift
apart between two files that both claim to drive the same hardware.

    python3 move_camera.py
"""
import pathlib
import runpy
import sys

TARGET = (pathlib.Path(__file__).resolve().parent.parent
          / "mobility" / "move_camera.py")

if not TARGET.exists():
    sys.exit(f"Cannot find {TARGET}. Has mobility/ been pulled?")

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
