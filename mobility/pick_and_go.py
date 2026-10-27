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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spraying"))
sys.path.insert(0, str(HERE.parent))

import aim as A
import motors as M
import simulate as S
import spray_box as SB
import run_plan as RP

MAPS = HERE.parent / "mapping" / "maps"


def newest_map():
    """The map most recently made, since that is nearly always the one."""
    if not MAPS.exists():
        sys.exit(f"No maps folder at {MAPS}")
    beds = [d for d in MAPS.iterdir() if d.is_dir() and (d / f"{d.name}.json").exists()]
    if not beds:
        sys.exit(f"No maps in {MAPS}. Run the mapping pathway first.")
    return sorted(beds, key=lambda d: -d.stat().st_mtime)[0]


def draw(cells, cols, rows, done):
    """
    The grid, as text.

    Columns run left to right as x grows, rows top to bottom as y grows,
    which is the same orientation as the map image - so what you see here
    and what you saw on the laptop are the same way up.
    """
    by_name = {c["name"]: c for c in cells}
    width = 7

    print("")
    print("    " + "".join(f"{'C' + str(c + 1):^{width}}" for c in range(cols)))
    for r in range(rows):
        line = f" R{r + 1} "
        for c in range(cols):
            cell = by_name.get(f"C{c + 1}R{r + 1}")
            if cell is None:
                line += f"{'':^{width}}"
            elif cell["count"]:
                line += f"{'[crop]':^{width}}"
            elif cell["name"] in done:
                line += f"{'done':^{width}}"
            else:
                line += f"{str(cell['index']):^{width}}"
        print(line)
    print("")
    print("  [crop] holds a plant and cannot be sprayed.")
    print("  Type a number to spray that box, or q to finish.")


def plan_one(cell, cells, plants, bed, rail, params):
    """Work out the moves for a single box, as motor commands."""
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
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--max-turn", type=float, default=RP.MAX_DRIVE_S["pan"],
                    help="longest single pan drive, seconds")
    ap.add_argument("--rail-x", type=float, default=None, metavar="CM",
                    help="where the head sits across the bed. Omit for the "
                         "edge; give a value inside the bed if it is "
                         "mounted over it (e.g. 15.25 for mid-bed)")
    ap.add_argument("--pan-limit", type=float, default=None,
                    help="how far pan may swing either side of its datum")
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
    rail = S.limit_rail(S.build_rail(bed, "left", args.standoff),
                        params["rail_length"])

    if args.rail_x is not None:
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

    cells = SB.build_cells(bed, plants, args.cols, args.rows)
    for i, c in enumerate(cells):
        c["index"] = i

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

    if args.no_rail:
        print("  Rail is OFF - only pan and tilt will move.")
        print("  The head stays put, so nothing lands where the map says.")

    hw = RP.Hardware(dry=not args.live)
    if args.live:
        RP.ask_datum()

    done = set()
    try:
        while True:
            draw(cells, args.cols, args.rows, done)
            try:
                said = input("  > ").strip().lower()
            except EOFError:
                break
            if said in ("q", "quit", "exit"):
                break
            if not said:
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

            cell = cells[idx]
            if cell["count"]:
                print(f"  Box {idx} holds a plant. Pick another.")
                continue

            print(f"\n  Planning {cell['name']}...")
            rows = plan_one(cell, cells, plants, bed, rail, params)
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
                   max_turn=args.max_turn, use_rail=not args.no_rail)
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
