"""
Stopping cleanly, at every stage.

    with safestop.guard("mapping"):
        main()

Ctrl+C anywhere in stages 1 to 3 and you get a short message saying what was
and was not written, instead of thirty lines of traceback that make it look
as though something broke.

WHAT ABORT MEANS DEPENDS ON THE STAGE
    Stages 1-3 only ever write files. Stopping them is free: nothing has
    moved, nothing is half-done that matters, and the worst case is a
    partly written output you can delete and redo.

    Stage 4 is different. Motors are turning and a valve may be open. There
    the order matters and it is not negotiable:

        1. SHUT THE VALVE
        2. then stop the motors

    Stop the motors first and the valve keeps running onto one spot. A
    second of that is a heap of mulch on whatever happens to be underneath.
    Shut it first and the worst case is a short line of mulch where the
    head was already travelling.

    That sequence belongs in the Pi code, not here - this file cannot reach
    the hardware. It is written down because it is the kind of thing that
    gets decided wrong at two in the morning.

AND A PHYSICAL STOP BEATS A SOFTWARE ONE
    Ctrl+C depends on a laptop, a USB cable and a Python process all still
    working. A switch in the air line or the motor supply does not. Build
    the switch; this is the convenience, not the safety case.
"""
from __future__ import annotations

import contextlib
import signal
import sys


@contextlib.contextmanager
def guard(stage="this stage", wrote=None, on_abort=None):
    """
    Turn Ctrl+C into a clean stop.

    stage     named in the message, so it is obvious what was interrupted
    wrote     a callable returning a list of paths finished so far
    on_abort  a callable run before exiting, for anything that must be
              undone or released
    """
    try:
        yield
    except KeyboardInterrupt:
        print("\n\n" + "=" * 62)
        print(f"STOPPED  -  {stage} was interrupted")
        print("=" * 62)

        if on_abort:
            try:
                on_abort()
            except Exception as e:
                print(f"  (cleanup raised {type(e).__name__}: {e})")

        done = []
        if wrote:
            try:
                done = list(wrote() or [])
            except Exception:
                done = []

        if done:
            print("\nFinished before the stop:")
            for d in done:
                print(f"  {d}")
            print("\nAnything not listed was not written. Delete the folder")
            print("and run again if you want a clean start.")
        else:
            print("\nNothing was written.")

        print("\nNothing has moved - these stages only produce files.")
        print("=" * 62)
        sys.exit(130)       # the conventional code for "killed by Ctrl+C"


def ignore_sigint_during(fn, *args, **kwargs):
    """
    Run something that must not be cut in half - a file write, mainly.

    Ctrl+C pressed during it is held and raised the moment it finishes, so
    the interrupt is honoured but the file is whole.
    """
    pending = {}

    def hold(sig, frame):
        pending["hit"] = True
        print("\n  (finishing the current write first...)")

    try:
        old = signal.signal(signal.SIGINT, hold)
    except (ValueError, AttributeError):
        return fn(*args, **kwargs)      # not on the main thread

    try:
        return fn(*args, **kwargs)
    finally:
        signal.signal(signal.SIGINT, old)
        if pending:
            raise KeyboardInterrupt


ABORT_NOTE = """
ABORTING A RUN

  Stages 1-3  (calibration, mapping, spraying)
      Ctrl+C in the terminal. Nothing has moved; these stages only write
      files. Close the corner-picking or box-picking window with q and the
      run stops the same way.

  Stage 4  (mobility - the machine is moving)
      Shut the valve FIRST, then stop the motors. Stopping the motors with
      the valve open leaves it running onto one spot.

      Best is a physical switch in the air line or the motor supply, which
      does not depend on a laptop, a cable and a process all still working.
"""
