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


RAIL_KEYS = "abcdefgh"


def rail_stops(step_mm, count):
    """Where along the rail the head may be parked, in cm from home."""
    return [(RAIL_KEYS[i], i * step_mm / 10.0) for i in range(count)]


def draw(cells, cols, rows, done, hw=None, stops=()):
    """
    The grid, as text.

    Columns run left to right as x grows, rows top to bottom as y grows,
    which is the same orientation as the map image - so what you see here
    and what you saw on the laptop are the same way up.
    """
    by_name = {c["name"]: c for c in cells}
    width = 7

    # The rail stops print down the right of the grid, so the two things
    # the operator can choose - which box to aim at, and where along the
    # rail to stand - are read in one glance rather than two menus.
    gap = "     "
    head = "    " + "".join(f"{'C' + str(c + 1):^{width}}" for c in range(cols))
    if stops:
        head += gap + "rail"
    print("")
    print(head)

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
        if r < len(stops):
            key, cm = stops[r]
            here = ""
            if hw is not None and abs(hw.rail_cm - cm) < 0.2:
                here = "  <- here"
            line += f"{gap} {key}  {cm * 10:3.0f} mm{here}"
        print(line)

    # Any stops past the last grid row carry on underneath it.
    for key, cm in stops[rows:]:
        pad = "    " + " " * (width * cols)
        here = ""
        if hw is not None and abs(hw.rail_cm - cm) < 0.2:
            here = "  <- here"
        print(f"{pad}{gap} {key}  {cm * 10:3.0f} mm{here}")
    if hw is not None:
        # Where the code thinks the axes are. This is a running total of
        # what it has commanded, not a measurement - nothing on this
        # machine reports an angle back. If it disagrees with what you
        # can see, the belief is what is wrong, and z fixes it.
        print(f"  Believed now: pan {hw.angle['pan']:+.0f}, "
              f"tilt {hw.angle['tilt']:+.0f} deg")
    print("")
    print("  [crop] holds a plant and cannot be sprayed.")
    print("  number  aim at that box      letter  move the rail there")
    print("  n       back to neutral")
    print("  r       retrace backwards, valve shut")
    print("  z       call this position neutral")
    print("  q       finish")


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

    cells = SB.build_cells(bed, plants, args.cols, args.rows)
    for i, c in enumerate(cells):
        c["index"] = i
        if args.all_boxes:
            # For a movement test the plants do not matter - what is
            # wanted is every box reachable, so the head can be sent
            # round all of them and watched.
            c["count"] = 0

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

    stops = rail_stops(args.rail_step, args.rail_stops)
    done = set()
    # Where each box was aimed, in the order they were done, so the run
    # can be walked back through.
    history = []
    try:
        while True:
            draw(cells, args.cols, args.rows, done, hw, stops)
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
                    for name, pan, tilt in reversed(history):
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
                   max_turn=args.max_turn, use_rail=False)
            history.append((cell["name"], float(rows[-1]["pan_deg"]),
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
