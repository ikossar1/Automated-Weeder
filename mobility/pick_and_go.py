"""
Pick a box, watch it spray, pick the next one. All in one terminal.

This is the whole machine driven from the Pi over SSH, with no window to
click and no file to carry across from a laptop. The map is drawn as text
because a graphical window cannot reach you through an SSH session, and
because a 4x2 grid of boxes does not need pixels to be legible.

Nothing is queued. One box is planned, driven and sprayed while you
watch, and only then are you asked for the next. That is the difference
from spray.py and spray_box.py: those decide everything up front and then
commit, which is right for a finished bed and wrong for a bench test
where you want to stop after the first box that misbehaves.

    python3 pick_and_go.py                 rehearse, touching no pins
    python3 pick_and_go.py --live          actually drive the motors
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spraying"))
sys.path.insert(0, str(HERE.parent))

import aim as A
import motors as M
import simulate as S
import spray_box as SB
import run_plan as RP

NL_ = chr(10)

MAPS = HERE.parent / "mapping" / "maps"
BOXES_FILE = HERE / "boxes.txt"
EXAMPLE_FILE = HERE / "boxes.example.txt"


def newest_map():
    """The map most recently made, since that is nearly always the one."""
    if not MAPS.exists():
        sys.exit(f"No maps folder at {MAPS}")
    beds = [d for d in MAPS.iterdir() if d.is_dir() and (d / f"{d.name}.json").exists()]
    if not beds:
        sys.exit(f"No maps in {MAPS}. Run the mapping pathway first.")
    return sorted(beds, key=lambda d: -d.stat().st_mtime)[0]


# The four boxes keep the numbers they already had, so nothing taught
# so far has to be renumbered. 3 and 5 are no longer boxes - they are
# the two x translations, which are taught and fired the same way but
# move the rail instead of aiming at anything.
BOX_SLOTS = [0, 1, 2, 4]
X_SLOTS = {3: "X_FWD", 5: "X_BACK"}

RAIL_KEYS = "abcdefgh"


def rail_stops(step_mm, count):
    """Where along the rail the head may be parked, in cm from home."""
    return [(RAIL_KEYS[i], i * step_mm / 10.0) for i in range(count)]


def draw(cells, cols, rows, done, hw=None, stops=(), taught=None):
    """
    The boxes and the x translations, in one picture.

    Both are things you press a number for and both are taught the same
    way, so they belong on one screen - with what each one actually does
    written next to it, because a bare number says nothing about whether
    it has been taught or what it would do if pressed.
    """
    taught = taught or {}
    by_slot = {}
    for n, cell in enumerate(cells):
        if n < len(BOX_SLOTS):
            by_slot[BOX_SLOTS[n]] = cell

    print("")
    print("        " + "".join(f"{'C' + str(c + 1):^9}" for c in range(cols))
          + "      x translation")
    for r in range(rows):
        line = f"   R{r + 1}  "
        for c in range(cols):
            n = r * cols + c
            cell = cells[n] if n < len(cells) else None
            if cell is None:
                line += f"{'':^9}"
            elif cell["count"]:
                line += f"{'[crop]':^9}"
            elif cell["name"] in done:
                line += f"{'done':^9}"
            else:
                line += f"{str(cell['index']):^9}"
        slots = sorted(X_SLOTS)
        if r < len(slots):
            slot = slots[r]
            name = X_SLOTS[slot]
            cases = taught.get(name, [])
            what = (f"{cases[0]['rail_mm']:+.0f} mm" if cases
                    else "not taught")
            line += f"      {slot}   {name.lower():<7} {what}"
        print(line)

    print("")
    print("   What each one does:")
    for slot in sorted(list(by_slot) + list(X_SLOTS)):
        if slot in by_slot:
            cell = by_slot[slot]
            cases = taught.get(cell["name"].upper(), [])
            label = f"box {cell['name']}"
            if cell["count"]:
                detail = "holds a crop, will not be sprayed"
            elif not cases:
                detail = "not taught yet"
            else:
                detail = ", ".join(c["case"] for c in cases)
        else:
            name = X_SLOTS[slot]
            cases = taught.get(name, [])
            label = f"x move {name.lower()}"
            detail = (", ".join(f"{c['case']} {c['rail_mm']:+.0f}mm"
                                for c in cases) if cases else "not taught yet")
        print(f"     {slot}   {label:<14} {detail}")

    if hw is not None:
        print("")
        print(f"   Believed now: pan {hw.angle['pan']:+.0f}, "
              f"tilt {hw.angle['tilt']:+.0f} deg, rail "
              f"{hw.rail_cm * 10:.0f} mm")


def load_taught(path=BOXES_FILE):
    """Cases from boxes.txt, as {box: [case, case, ...]} in file order."""
    out = {}
    if not path.exists():
        # First run here. Start from the template rather than nothing,
        # so there is something to edit and to read the columns off.
        if EXAMPLE_FILE.exists():
            path.write_text(EXAMPLE_FILE.read_text(encoding="utf-8"),
                            encoding="utf-8")
            print(f"  Started {path.name} from the template.")
        else:
            return out
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        bits = line.split()
        if len(bits) != 7:
            print(f"  boxes.txt line {n}: want 7 values, got {len(bits)}")
            continue
        try:
            case = {
                "case": bits[1],
                "rail_mm": float(bits[2]), "pan_s": float(bits[3]),
                "tilt_s": float(bits[4]), "spray_s": float(bits[5]),
                "crosses": [] if bits[6] == "-" else
                           [b.strip().upper() for b in bits[6].split(",")],
            }
        except ValueError:
            print(f"  boxes.txt line {n}: not all numbers, ignored")
            continue
        out.setdefault(bits[0].upper(), []).append(case)
    return out



def choose_case(name, cases, occupied):
    """
    The first case whose spray does not pass over a planted box.

    This is the map earning its place. A taught motion cannot see a
    plant, but it can be labelled with the ground it flies over, and
    the map knows what is growing there. So the choice between "head
    on" and "from further along the rail" is made from the bed as it
    is today, not from how it looked when the motions were taught.
    """
    for case in cases:
        blocked = [b for b in case["crosses"] if b in occupied]
        if not blocked:
            return case, None
        last = blocked
    return None, last



def run_taught(hw, name, m, use_rail, cap=5.0, spray=True):
    """
    Replay a taught motion, then undo it.

    Undoing matters more than it looks. Nothing measures where pan and
    tilt are, so the only way the next box can start from a known place
    is to put this one back where it began - same durations, opposite
    directions. Drift still creeps in, which is what z is for.
    """
    def drive(axis, secs):
        if abs(secs) < RP.MIN_MOVE_S:
            return
        if abs(secs) > cap:
            print(f"    {axis} {abs(secs):.1f}s is over the {cap:.0f}s "
                  f"limit - check boxes.txt")
            return
        hw.drive(axis, 1 if secs > 0 else -1)
        RP.wait(hw, abs(secs))
        hw.stop(axis)
        RP.wait(hw, RP.SETTLE_S)

    hw.valve(False)
    if use_rail:
        hw.rail_to(m["rail_mm"] / 10.0)

    print(f"    pan {m['pan_s']:+.2f}s, tilt {m['tilt_s']:+.2f}s")
    drive("pan", m["pan_s"])
    drive("tilt", m["tilt_s"])

    if spray:
        print(f"    spraying {m['spray_s']:.1f}s")
        hw.valve(True)
        RP.wait(hw, m["spray_s"])
        hw.valve(False)

    print("    back to the start")
    drive("tilt", -m["tilt_s"])
    drive("pan", -m["pan_s"])



def save_case(box, case, rail_mm, pan_s, tilt_s, spray_s, crosses,
              path=BOXES_FILE):
    """
    Write one case back to boxes.txt, keeping the comments.

    Rewrites the matching box+case line if there is one, appends if not.
    The notes at the top of that file are the only explanation of what
    the columns mean, so they are preserved rather than regenerated.
    """
    line = (f"{box:<8} {case:<11} {rail_mm:6.0f} {pan_s:6.2f} "
            f"{tilt_s:7.2f} {spray_s:8.1f}   "
            f"{','.join(crosses) if crosses else '-'}")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    for i, old in enumerate(lines):
        bare = old.split("#")[0].split()
        if len(bare) >= 2 and bare[0].upper() == box and bare[1] == case:
            lines[i] = line
            path.write_text(NL_.join(lines) + NL_, encoding="utf-8")
            return "updated"

    lines.append(line)
    path.write_text(NL_.join(lines) + NL_, encoding="utf-8")
    return "added"


ARROWS = {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}

# Which way each key drives. The size of the step is set separately, so
# the same four keys do fine work or coarse without changing meaning.
def delete_case(box, case, path=BOXES_FILE):
    """Drop one case from boxes.txt, leaving the comments alone."""
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        bare = line.split("#")[0].split()
        if len(bare) >= 2 and bare[0].upper() == box and bare[1] == case:
            del lines[i]
            path.write_text(NL_.join(lines) + NL_, encoding="utf-8")
            return True
    return False


JOG = {"LEFT": ("pan", -1), "RIGHT": ("pan", +1),
       "UP": ("tilt", +1), "DOWN": ("tilt", -1)}
# Which way s and j move the rail. How FAR is rail_jog_cm in
# settings.txt, read at run time so it can be changed without code.
RAIL_KEY = {"s": +1, "j": -1}
STEPS = (0.1, 0.4, 1.0)


def read_key(prompt="    > "):
    """
    One keypress, arrows included, without waiting for Enter.

    The arrow keys send an escape sequence rather than a character, and
    a normal input() never sees them - the shell's line editor takes
    them first. So the terminal is put in raw mode for exactly as long
    as it takes to read one key, and put back afterwards.

    Falls back to reading a line where there is no terminal, so the
    thing can still be driven from a script or a pipe.
    """
    print(prompt, end="", flush=True)
    try:
        import termios
        import tty
    except ImportError:
        return input().strip()
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.strip()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == chr(27):            # escape, so an arrow key follows
            return ARROWS.get(sys.stdin.read(2), "ESC")
        if ch == chr(3):             # ctrl-C in raw mode
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def teach_x(hw, name, taught, use_rail):
    """
    Teach one x translation: drive the rail, keep the distance.

    Separate from the boxes on purpose. Moving along the bed and aiming
    at something are different jobs, and tying them together was what
    made a box press drag the rail back to where the aiming geometry
    thought it should be. Here the rail is the whole of it - no pan, no
    tilt, no valve.
    """
    if not use_rail:
        print("  The rail is off, so there is nothing to teach.")
        return None

    print("")
    print(f"  Teaching {name.lower()}. Existing: "
          + (", ".join(c["case"] for c in taught.get(name, [])) or "none"))
    case = input("  Name for this case [standard]: ").strip() or "standard"

    begin = hw.rail_cm
    print("")
    print(f"    s / j   rail right / left {RP.RAIL_JOG_CM:.0f} cm")
    print("    0       back to where this started")
    print("    k       keep the distance      c  cancel")

    while True:
        moved = (hw.rail_cm - begin) * 10.0
        print(f"{NL_}    moved {moved:+.0f} mm from where you began")
        try:
            said = read_key(f"    teach {name.lower()} > ")
        except EOFError:
            return None
        if said in RAIL_KEY:
            hw.rail_to(hw.rail_cm + RAIL_KEY[said] * RP.RAIL_JOG_CM)
        elif said == "0":
            hw.rail_to(begin)
        elif said == "c":
            print("    cancelled, nothing saved")
            return None
        elif said == "k":
            if abs(moved) < 0.5:
                print("    It has not moved. Drive it somewhere first.")
                continue
            what = save_case(name, case, moved, 0.0, 0.0, 0.0, [])
            print(f"    {what} {name.lower()} '{case}': {moved:+.0f} mm")
            return load_taught()
        elif said:
            print("    s, j, 0, k or c")


def teach(hw, cell, taught, use_rail, stops, history=None):
    """
    Jog and spray in the same loop, and save what was last sprayed from.

    Aiming and timing are not really separable on this machine: where
    the mulch lands is the only readout either of them has, and you get
    it by firing. So the valve is available throughout, and the position
    recorded is the one it was last fired from - jogging on afterwards
    is how you look for something better, not how you withdraw what
    already worked.
    """
    box = cell["name"].upper()
    print("")
    print(f"  Teaching {box}. Existing cases: "
          + (", ".join(c["case"] for c in taught.get(box, [])) or "none"))
    name = input("  Name for this case [direct]: ").strip() or "direct"

    start = dict(hw.angle)
    totals = {"pan": 0.0, "tilt": 0.0}
    rail_mm = hw.rail_cm * 10.0
    spray_s = 2.0
    step = STEPS[0]
    approved = None
    opened_at = None

    def help_():
        print("")
        print(f"    TEACHING {box} - one keypress, no Enter needed")
        print("      arrows  left / right pan, up / down tilt")
        if use_rail:
            print(f"      s / j   rail right / left {RP.RAIL_JOG_CM:.0f} cm")
        print("      f       change the step size")
        print("      t       valve OPEN, t again to SHUT")
        print("      + / -   nudge the spray time, p to type it")
        print("      0       undo the jogging, back to the start")
        print("      ?       show this again")
        print("      k       SAVE and leave      c  leave without saving")

    help_()

    while True:
        mark = ""
        if approved is not None:
            mark = (f"   (sprayed from pan {approved['pan']:+.2f}"
                    f" tilt {approved['tilt']:+.2f})")
        print(f"{NL_}    pan {totals['pan']:+.2f}s  tilt {totals['tilt']:+.2f}s"
              f"  rail {rail_mm:.0f}mm  step {step:.1f}s"
              f"  spray {spray_s:.1f}s{mark}")
        try:
            said = read_key(f"    teach {box} > ")
        except EOFError:
            return None
        if not said:
            continue

        if said == "?":
            help_()
        elif said in JOG:
            axis, way = JOG[said]
            hw.drive(axis, way)
            RP.wait(hw, step)
            hw.stop(axis)
            RP.wait(hw, RP.SETTLE_S)
            totals[axis] += way * step
        elif said in RAIL_KEY:
            if not use_rail:
                print("    rail is off")
                continue
            rail_mm += RAIL_KEY[said] * RP.RAIL_JOG_CM * 10.0
            hw.rail_to(rail_mm / 10.0)
        elif said == "f":
            step = STEPS[(STEPS.index(step) + 1) % len(STEPS)]
        elif said == "t":
            # Open, and open it stays until you press t again. The time
            # between the two presses IS the spray time - you judge it
            # by watching the mulch land, which is the only way it can
            # be judged, and the clock just records what you decided.
            if opened_at is None:
                hw.valve(True)
                opened_at = time.monotonic()
                print("    valve OPEN - press t again to shut it")
            else:
                hw.valve(False)
                spray_s = round(max(0.1, time.monotonic() - opened_at), 1)
                opened_at = None
                approved = {"pan": totals["pan"], "tilt": totals["tilt"],
                            "rail_mm": rail_mm, "spray_s": spray_s}
                print(f"    valve SHUT after {spray_s:.1f}s - that is what "
                      f"k will save")
        elif said in ("+", "-"):
            spray_s = max(0.1, spray_s + (0.2 if said == "+" else -0.2))
        elif said == "p":
            try:
                spray_s = max(0.1, float(input("    seconds: ").strip()))
            except ValueError:
                print("    that is not a number")
        elif said == "0":
            for axis in ("tilt", "pan"):
                if abs(totals[axis]) >= RP.MIN_MOVE_S:
                    hw.drive(axis, -1 if totals[axis] > 0 else 1)
                    RP.wait(hw, abs(totals[axis]))
                    hw.stop(axis)
            totals.update({"pan": 0.0, "tilt": 0.0})
            hw.angle.update(start)
        elif said == "c":
            if opened_at is not None:
                hw.valve(False)
            print("    cancelled, nothing saved")
            return None
        elif said == "k":
            if opened_at is not None:
                hw.valve(False)
                opened_at = None
                print("    valve shut first")
            keep = approved
            if keep is None:
                print("    Not sprayed yet, so nothing has been seen to")
                print("    work. Press t first, or k again to save anyway.")
                if read_key("    > ") != "k":
                    continue
                keep = {"pan": totals["pan"], "tilt": totals["tilt"],
                        "rail_mm": rail_mm, "spray_s": spray_s}
            elif (abs(keep["pan"] - totals["pan"]) > 1e-9
                    or abs(keep["tilt"] - totals["tilt"]) > 1e-9):
                print(f"    Saving where it sprayed from "
                      f"(pan {keep['pan']:+.2f}s, tilt {keep['tilt']:+.2f}s),")
                print("    not where the jogging left it.")
            raw = input("    Boxes the spray crosses, comma "
                        "separated [-]: ").strip()
            crosses = [] if raw in ("", "-") else [
                b.strip().upper() for b in raw.split(",")]
            what = save_case(box, name, keep["rail_mm"], keep["pan"],
                             keep["tilt"], keep["spray_s"], crosses)
            # A taught case is a motion the machine has just performed,
            # so it belongs in the history like any other - otherwise
            # retrace has nothing to walk back through after a session
            # spent teaching rather than spraying.
            if history is not None:
                history.append((box, dict(keep, case=name,
                                          crosses=crosses), 0.0, 0.0))
            print(f"    {what} {box} '{name}': pan {keep['pan']:+.2f}s, "
                  f"tilt {keep['tilt']:+.2f}s, spray {keep['spray_s']:.1f}s")
            return load_taught()
        else:
            print("    not a command")


def centre_shot(cell, rail, params):
    """
    One shot at the middle of the box, and nothing else.

    A full box is covered by a serpentine of eight or so points, and from
    a head that cannot move they sit at quite different angles - so the
    nozzle sweeps back and forth across the box and the tilt rises and
    falls. That is correct, and it is useless for telling whether the
    machine went to the right place. Aiming once at the centre makes the
    movement legible: one pan, one tilt, one spray.
    """
    pos = rail["lo"]
    noz = S.nozzle_at(rail, pos, params["height"], params["pan_limit"],
                      params["tilt_min"], params["tilt_max"],
                      params["throw"], 0.0, params["offset_across"])
    shot = A.aim(noz, (cell["x_cm"], cell["y_cm"]))
    row = [0, round(pos, 1), round(shot.pan_deg, 1), round(shot.tilt_deg, 1),
           round(cell["x_cm"], 1), round(cell["y_cm"], 1), 0, 0]
    on = list(row); on[6] = 1
    off = list(row); off[6] = 0
    return [row, on, off], shot


def plan_one(cell, cells, plants, bed, rail, params, one_shot=False):
    """Work out the moves for a single box, as motor commands."""
    if one_shot:
        steps, shot = centre_shot(cell, rail, params)
        if not shot.ok:
            print(f"  {cell['name']}: centre is out of reach "
                  f"({shot.why if hasattr(shot, 'why') else 'no solution'})")
            return []
        print(f"  {cell['name']}: pan {shot.pan_deg:+.0f}, "
              f"tilt {shot.tilt_deg:.0f}")
    else:
        steps = SB.box_plan(cells, [cell["name"]], plants, bed, rail, params)
    if not steps:
        return []

    plan = {
        "bed": list(bed),
        "plants": [{"id": p.id, "x": round(p.x_cm, 1), "y": round(p.y_cm, 1),
                    "r": round(A.footprint_radius(p), 1)} for p in plants],
        "rail": {k: (round(v, 1) if isinstance(v, float) else v)
                 for k, v in rail.items()},
        "nozzle": {"height": params["height"], "throw": params["throw"]},
        "stops": sorted({s[1] for s in steps}),
        "boxes": [cell["name"]],
        "fields": ["box", "rail_pos", "pan", "tilt", "land_x", "land_y",
                   "spraying", "blocked_by"],
        "steps": steps,
    }
    out = HERE / "_last_box.csv"
    M.export_dc(plan, M.DEFAULT_DC_HEAD, out_csv=str(out))
    return RP.load_rows(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick a box, spray it, repeat.")
    ap.add_argument("--live", action="store_true",
                    help="drive the motors for real (default is a rehearsal)")
    ap.add_argument("--bed", help="map name; default is the newest")
    ap.add_argument("--width", type=float, default=30.5,
                    help="bed width cm, across the rail (default 30.5)")
    ap.add_argument("--depth", type=float, default=35.0,
                    help="bed depth cm, along the rail (default 35)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--max-turn", type=float, default=RP.MAX_DRIVE_S["pan"],
                    help="longest single pan drive, seconds")
    ap.add_argument("--rail-x", type=float, default=None, metavar="CM",
                    help="where the head sits across the bed. Omit for the "
                         "edge; give a value inside the bed if it is "
                         "mounted over it (e.g. 15.25 for mid-bed)")
    ap.add_argument("--pan-limit", type=float, default=None,
                    help="how far pan may swing either side of its datum")
    ap.add_argument("--head-x", type=float, default=None, metavar="CM",
                    help="where the head sits across the bed "
                         "(default: the middle)")
    ap.add_argument("--head-y", type=float, default=0.0, metavar="CM",
                    help="how far the head sits from the near edge "
                         "(default 0, level with it)")
    ap.add_argument("--rail-step", type=float, default=80.0, metavar="MM",
                    help="spacing of the rail stops in mm (default 80)")
    ap.add_argument("--rail-stops", type=int, default=3,
                    help="how many rail stops to offer (default 3)")
    ap.add_argument("--max-drive", type=float, default=RP.MAX_DRIVE_S_HARD,
                    metavar="SEC", help="longest one drive may run")
    ap.add_argument("--drive-steps", type=int, default=RP.MAX_DRIVE_STEPS,
                    help="how many drives one move may be split into")
    ap.add_argument("--tilt-down", type=float, default=RP.TILT_DOWN,
                    metavar="DEG",
                    help="tip every shot this many degrees further down, "
                         "to correct a mount whose zero is not level")
    ap.add_argument("--computed", action="store_true",
                    help="ignore boxes.txt and work the angles out from "
                         "the map instead")
    ap.add_argument("--one-shot", action="store_true",
                    help="aim once at the centre of the box instead of "
                         "covering it, so the movement is easy to follow")
    ap.add_argument("--offset", type=float, default=0.0, metavar="CM",
                    help="camera-to-nozzle offset across the head "
                         "(default 0 when the rail is off)")
    ap.add_argument("--all-boxes", action="store_true",
                    help="allow picking boxes that hold a crop, for "
                         "testing movement")
    ap.add_argument("--no-rail", action="store_true",
                    help="leave the rail stepper alone and move only pan "
                         "and tilt")
    ap.add_argument("--duty", type=float, default=RP.DUTY,
                    help=f"motor power percent, lower is slower "
                         f"(default {RP.DUTY:.0f})")
    ap.add_argument("--standoff", type=float,
                    default=0.0)
    ap.add_argument("--height", type=float, default=9.4)
    args = ap.parse_args(argv)

    bed_dir = (MAPS / args.bed) if args.bed else newest_map()
    map_json = bed_dir / f"{bed_dir.name}.json"
    if not map_json.exists():
        sys.exit(f"No map at {map_json}")

    plants, _doc = S.load_plants(map_json)
    bed = (0.0, 0.0, args.width, args.depth)
    params = dict(SB.DEFAULTS)
    params.update({"height": args.height, "standoff": args.standoff,
                   "pan_limit": 70.0, "cell": SB.DEFAULTS["cell"],
                   "tilt_min": SB.DEFAULTS["tilt_min"],
                   "tilt_max": SB.DEFAULTS["tilt_max"]})
    # The head looks INTO the bed from the near edge, whether or not the
    # rail is working. Pan swings across the columns, tilt sets how far
    # out the shot lands, and the rail - when it is on - carries the head
    # up the bed so the far rows come within tilt range at all.
    #
    # Keeping one geometry for both cases matters: the old code swapped
    # to an edge-mounted rail pointing sideways as soon as the rail was
    # enabled, which is a different machine from the one in the room.
    head_x = args.head_x if args.head_x is not None else (bed[2] + bed[0]) / 2
    # Aiming is worked out from the head standing still, always. The
    # rail is a separate axis the operator drives with the letter keys,
    # and turning it on does not change a single pan or tilt angle -
    # which is the point: the aiming was checked box by box on the
    # machine, and enabling another motor should not quietly redo it.
    rail = {"axis": "y", "fixed": head_x, "lo": args.head_y,
            "hi": args.head_y, "heading": 90.0}
    params["pan_limit"] = args.pan_limit or 70.0
    params["offset_across"] = args.offset

    if args.rail_x is not None and not args.no_rail:
        lo, hi = ((bed[0], bed[2]) if rail["axis"] == "y"
                  else (bed[1], bed[3]))
        rail["fixed"] = args.rail_x
        if lo < args.rail_x < hi:
            # Mounted over the bed, so boxes sit on BOTH sides of the
            # head. Pointing across would put half of them behind it and
            # ask pan for 180 degrees it does not have. Pointing along
            # the rail instead makes pan a left/right swing, and each
            # side is then a quarter turn away at most.
            rail["heading"] = 90.0 if rail["axis"] == "y" else 0.0
            params["pan_limit"] = args.pan_limit or 90.0
            print(f"  Head sits {args.rail_x:.2f} cm across the bed, "
                  f"inside it -")
            print(f"  aiming along the rail so both sides are in range.")
    if args.pan_limit:
        params["pan_limit"] = args.pan_limit

    print("=" * 62)
    print(f"PICK AND GO   {'LIVE' if args.live else '(rehearsal)'}")
    print("=" * 62)
    print(f"  Bed    : {bed_dir.name},  {bed[2] - bed[0]:.0f} x "
          f"{bed[3] - bed[1]:.0f} cm,  {len(plants)} plants")
    S.report_reach(bed, rail, params["height"], params["throw"],
                   params["tilt_min"], params["tilt_max"])

    if args.duty != RP.DUTY:
        # Changing the power changes how far a given duration turns the
        # axis, so the speeds and the caps have to move with it.
        RP.DUTY = args.duty
        RP.SPEED_DEG = {a: v * args.duty / 100.0
                        for a, v in RP.SPEED_DEG_FULL.items()}
        RP.SPEED = {"rail": 4.0 * args.duty / 100.0,
                    "pan": RP.SPEED_DEG["pan"]}
        RP.MAX_DRIVE_S = {a: RP.MAX_DRIVE_DEG[a] / RP.SPEED_DEG[a] * RP.GRACE
                          for a in RP.MAX_DRIVE_DEG}
        print(f"  Motor power: {args.duty:.0f}%  -> pan "
              f"{RP.SPEED_DEG['pan']:.1f} deg/s")

    if args.tilt_down:
        print(f"  Every shot tipped {args.tilt_down:.0f} deg further down.")

    if args.no_rail:
        print(f"  Rail OFF - head fixed at x={rail['fixed']:.1f}, "
              f"y={args.head_y:.1f}, looking into the bed.")
        print("  Pan covers the columns, tilt covers the rows.")

    RP.MAX_DRIVE_STEPS = args.drive_steps
    RP.MAX_DRIVE_S = {a: min(args.max_drive, RP.MAX_DRIVE_S[a])
                      for a in RP.MAX_DRIVE_S}
    print(f"  Drives: at most {args.drive_steps} per move, "
          f"{min(args.max_drive, RP.MAX_DRIVE_S_HARD):.2f}s each "
          f"= {args.drive_steps * min(args.max_drive, RP.MAX_DRIVE_S_HARD) * RP.SPEED_DEG['pan']:.0f} deg max")

    hw = RP.Hardware(dry=not args.live)
    if args.live:
        RP.ask_datum()

    # A movement test should not be stopped by what is growing in the
    # bed - it is about watching the axes reach each box.
    aim_plants = [] if args.all_boxes else plants
    if args.all_boxes:
        print("  All boxes selectable, plants ignored - movement test only.")

    cells = SB.build_cells(bed, plants, args.cols, args.rows)
    for i, c in enumerate(cells):
        c["index"] = BOX_SLOTS[i] if i < len(BOX_SLOTS) else i
        if args.all_boxes:
            # For a movement test the plants do not matter - what is
            # wanted is every box reachable, so the head can be sent
            # round all of them and watched.
            c["count"] = 0

    # Which boxes hold a plant, by name. This is the only thing the
    # case picker needs from the map, and it is the thing only the map
    # knows.
    occupied = {c["name"].upper() for c in cells if c["count"]}
    taught = {} if args.computed else load_taught()
    if taught:
        print(f"  {len(taught)} taught motion(s) in boxes.txt - these are")
        print("  replayed as written. --computed works the angles out instead.")
    stops = rail_stops(args.rail_step, args.rail_stops)
    done = set()
    # Where each box was aimed, in the order they were done, so the run
    # can be walked back through.
    history = []
    try:
        while True:
            draw(cells, args.cols, args.rows, done, hw, stops, taught)
            try:
                said = input("  > ").strip().lower()
            except EOFError:
                break
            if said in ("q", "quit", "exit"):
                break

            picked_stop = [s for s in stops if s[0] == said]
            if picked_stop:
                key, cm = picked_stop[0]
                if args.no_rail:
                    print(f"  Rail is off (--no-rail), so {key} cannot "
                          f"move it. Drop that flag to use the stops.")
                    continue
                # Travelling, so the valve is shut first - the same rule
                # the box runs follow.
                hw.valve(False)
                print(f"  Rail to {cm * 10:.0f} mm ...")
                was = RP.MAX_DRIVE_STEPS
                RP.MAX_DRIVE_STEPS = 40
                try:
                    hw.rail_to(cm)
                finally:
                    RP.MAX_DRIVE_STEPS = was
                print(f"  Rail now at {hw.rail_cm * 10:.0f} mm.")
                continue

            if said == "z":
                # The datum has drifted - from a force-stop, from being
                # nudged by hand, or from the run starting somewhere
                # other than neutral. Declaring the current position as
                # zero costs nothing and puts every later angle back on
                # a footing that matches the machine.
                hw.angle["pan"] = 0.0
                hw.angle["tilt"] = 0.0
                print("  Noted - this position is now neutral.")
                continue

            if said == "cases":
                if not taught:
                    print("  Nothing taught yet.")
                    continue
                print("")
                for box in sorted(taught):
                    free = box not in occupied
                    print(f"    {box}{'' if free else '   [crop - skipped]'}")
                    for c in taught[box]:
                        print(f"       {c['case']:<12} rail {c['rail_mm']:>4.0f}mm"
                              f"  pan {c['pan_s']:+.2f}s  tilt {c['tilt_s']:+.2f}s"
                              f"  spray {c['spray_s']:.1f}s"
                              f"  crosses {','.join(c['crosses']) or '-'}")
                print("")
                print("    Cases are tried top to bottom, so the first one")
                print("    whose crossings are clear is the one that runs.")
                continue

            if said == "del":
                box = input("  Which box (e.g. C1R1): ").strip().upper()
                if box not in taught:
                    print(f"  Nothing taught for {box}.")
                    continue
                print("    " + ", ".join(c["case"] for c in taught[box]))
                case = input("  Which case: ").strip()
                if delete_case(box, case):
                    print(f"  Removed {box} '{case}'.")
                    taught = load_taught()
                else:
                    print(f"  No case '{case}' for {box}.")
                continue

            if said == "jog":
                # Moving without recording. Inside teach every nudge has
                # to count, or the saved numbers would not reproduce the
                # position they describe - so repositioning to look at
                # something, or to free a jammed axis, belongs out here
                # where nothing is being written down.
                def jog_help():
                    print("")
                    print("    JOGGING - one keypress, no Enter needed")
                    print("      arrows  left / right pan, up / down tilt")
                    if not args.no_rail:
                        print(f"      s / j   rail right / left "
                              f"{RP.RAIL_JOG_CM:.0f} cm")
                    print("      ?       show this again")
                    print("      x       back to the main menu")
                    print("    Nothing here is saved. Press z afterwards")
                    print("    if you have moved what you call neutral.")

                jog_help()
                while True:
                    try:
                        key = read_key("    jog > ")
                    except EOFError:
                        break
                    if key in ("x", "q"):
                        break
                    if key == "?":
                        jog_help()
                    elif key in JOG:
                        axis, way = JOG[key]
                        hw.jog(axis, way * STEPS[0])
                    elif key in RAIL_KEY and not args.no_rail:
                        hw.rail_to(hw.rail_cm
                                   + RAIL_KEY[key] * RP.RAIL_JOG_CM)
                    elif key:
                        print("    arrows, s / j, or x")
                continue

            if said.startswith("t") and said[1:].strip().isdigit():
                idx = int(said[1:].strip())
                if idx in X_SLOTS:
                    got = teach_x(hw, X_SLOTS[idx], taught,
                                  not args.no_rail)
                    if got is not None:
                        taught = got
                    continue
                if idx not in BOX_SLOTS:
                    print(f"  There is no {idx}.")
                    continue
                got = teach(hw, cells[BOX_SLOTS.index(idx)], taught,
                            not args.no_rail, stops, history)
                if got is not None:
                    taught = got
                continue

            if said == "all":
                # The map decides the order, not the operator. This is
                # what the mapping stage is for: the boxes with crops in
                # them were already ruled out when the grid was built,
                # so going through what is left needs no further
                # judgement - only confirmation before it starts.
                todo = [c for c in cells
                        if not c["count"] and c["name"] not in done]
                if not todo:
                    print("  Nothing left to spray.")
                    continue
                print(f"  {len(todo)} free box(es): "
                      f"{', '.join(c['name'] for c in todo)}")
                if input("  Spray all of them? (y/N): ").strip().lower()                         not in ("y", "yes"):
                    continue
                for cell in todo:
                    print(f"{NL_}  {cell['name']}...")
                    if cell["name"].upper() in taught:
                        case, blocked = choose_case(
                            cell["name"], taught[cell["name"].upper()],
                            occupied)
                        if case is None:
                            print(f"    every case crosses "
                                  f"{', '.join(blocked)} - skipped")
                            continue
                        print(f"    case '{case['case']}'")
                        run_taught(hw, cell["name"], case,
                                   use_rail=not args.no_rail)
                        done.add(cell["name"])
                        continue
                    rows = plan_one(cell, cells, aim_plants, bed, rail,
                                    params, one_shot=args.one_shot
                                    or args.all_boxes)
                    if not rows or not any(
                            str(r["spray"]).strip() == "1" for r in rows):
                        print("    nothing reachable, skipped")
                        continue
                    RP.run(hw, rows, rehome_every_box=True,
                           max_turn=args.max_turn, use_rail=False,
                           tilt_bias=args.tilt_down)
                    history.append((cell["name"],
                                    float(rows[-1]["pan_deg"]),
                                    float(rows[-1]["tilt_deg"])))
                    done.add(cell["name"])
                print("  All done.")
                continue

            if said == "r":
                if not history:
                    print("  Nothing done yet to retrace.")
                    continue
                # Back through the same aim points in reverse, valve shut
                # throughout. Useful for checking that the angles the
                # machine reached on the way out are repeatable, without
                # laying any more mulch while doing it.
                print(f"  Retracing {len(history)} box(es) backwards, "
                      f"valve stays shut.")
                hw.valve(False)
                was = RP.MAX_DRIVE_STEPS
                RP.MAX_DRIVE_STEPS = 40
                try:
                    for name, case, pan, tilt in reversed(history):
                        if case is not None:
                            # Replay the taught motion itself, out and
                            # back, rather than an angle - a taught case
                            # is a pair of durations, and it is those
                            # that have to be repeated to land in the
                            # same place.
                            print(f"    back through {name} "
                                  f"('{case['case']}')")
                            run_taught(hw, name, case,
                                       use_rail=not args.no_rail,
                                       spray=False)
                        else:
                            print(f"    back to {name}  "
                                  f"pan {pan:+.0f}, tilt {tilt:.0f}")
                            hw.turn_to("pan", pan)
                            hw.turn_to("tilt", tilt)
                    print("    and back to neutral")
                    hw.turn_to("tilt", 0.0)
                    hw.turn_to("pan", 0.0)
                finally:
                    RP.MAX_DRIVE_STEPS = was
                    hw.valve(False)
                history.clear()
                done.clear()
                continue

            if said == "n":
                # Allowed to run to completion rather than one short
                # drive at a time. Aiming is kept incremental because a
                # wrong speed should only be able to be wrong by a few
                # degrees - but going back to a known datum is a thing
                # you ask for deliberately and want finished.
                print("  Returning to neutral...")
                was = RP.MAX_DRIVE_STEPS
                RP.MAX_DRIVE_STEPS = 40
                try:
                    hw.turn_to("tilt", 0.0)
                    hw.turn_to("pan", 0.0)
                finally:
                    RP.MAX_DRIVE_STEPS = was
                continue
            if not said:
                continue

            if said.isdigit() and int(said) in X_SLOTS:
                name = X_SLOTS[int(said)]
                cases = taught.get(name, [])
                if not cases:
                    print(f"  {name.lower()} is not taught yet - "
                          f"use t{said}.")
                    continue
                if args.no_rail:
                    print("  rail is off")
                    continue
                move = cases[0]["rail_mm"]
                print(f"  {name.lower()}: rail {move:+.0f} mm")
                hw.valve(False)
                was = RP.MAX_DRIVE_STEPS
                RP.MAX_DRIVE_STEPS = 40
                try:
                    hw.rail_to(hw.rail_cm + move / 10.0)
                finally:
                    RP.MAX_DRIVE_STEPS = was
                print(f"  rail now at {hw.rail_cm * 10:.0f} mm")
                continue

            try:
                idx = int(said)
            except ValueError:
                # Typing a shell command here is the easy mistake: this
                # prompt looks enough like a shell that a second command
                # gets typed into the still-running first one, which then
                # eats it as a box number and appears to do nothing.
                if any(said.startswith(w) for w in
                       ("python", "cd ", "ls", "git", "sudo", "./")):
                    print("")
                    print("  That is a shell command, and this is not a shell -")
                    print("  pick_and_go is still running and just read it as")
                    print("  your box choice.")
                    print("")
                    print("  Type  q  to leave, then run your command.")
                else:
                    print(f"  '{said}' is not a number.")
                continue
            if not 0 <= idx < len(cells):
                print(f"  There is no box {idx}.")
                continue

            if idx not in BOX_SLOTS:
                print(f"  There is no {idx}.")
                continue
            cell = cells[BOX_SLOTS.index(idx)]
            if cell["count"]:
                print(f"  Box {idx} holds a plant. Pick another.")
                continue

            if cell["name"].upper() in taught:
                case, blocked = choose_case(
                    cell["name"], taught[cell["name"].upper()], occupied)
                if case is None:
                    print(f"  {cell['name']}: every taught case shoots over "
                          f"{', '.join(blocked)}, which is planted.")
                    print("  Teach another approach, or use --computed.")
                    continue
                print(f"  {cell['name']} - taught motion '{case['case']}'"
                      + (f", crossing {', '.join(case['crosses'])}"
                         if case["crosses"] else ""))
                run_taught(hw, cell["name"], case,
                           use_rail=not args.no_rail)
                history.append((cell["name"], case, hw.angle["pan"],
                                hw.angle["tilt"]))
                done.add(cell["name"])
                continue

            print(f"\n  Planning {cell['name']}...")
            # use_rail=False below is not about --no-rail. Aiming at a
            # box only ever moves pan and tilt: the plan carries a rail
            # position of 0, because that is where the aiming geometry
            # stands the head, and letting the box run act on it would
            # drag the head back to 0 every time a box was pressed after
            # the rail had been moved by hand. The rail belongs to the
            # letter keys and to nothing else.
            rows = plan_one(cell, cells, aim_plants, bed, rail, params,
                            one_shot=args.one_shot or args.all_boxes)
            if not rows:
                print("  Nothing in that box is within reach of the sprayer.")
                print("  The reach check above says how far it can go.")
                continue

            on = sum(1 for r in rows if str(r["spray"]).strip() == "1")
            print(f"  {len(rows)} moves, spraying on {on}.")
            if on == 0:
                print("  Nothing would be sprayed, so not moving.")
                continue

            RP.run(hw, rows, rehome_every_box=True,
                   max_turn=args.max_turn, use_rail=False,
                   tilt_bias=args.tilt_down)
            history.append((cell["name"], None, float(rows[-1]["pan_deg"]),
                            float(rows[-1]["tilt_deg"])))
            done.add(cell["name"])
            print(f"  {cell['name']} done.")
    except KeyboardInterrupt:
        print("\n\n  Stopping - pulsing the valve shut, then the motors.")
    finally:
        hw.cleanup()
        print("\n  All pins low. Valve shut.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
