"""Fold the driver and the current plan into one standalone .py file."""
import csv, io, pathlib

root = pathlib.Path(r"C:\Users\isabe\OneDrive\Documents\GitHub\Automated-Weeder")
src = (root / "mobility" / "run_plan.py").read_text(encoding="utf-8")
plan = (root / "handoff_pi" / "dc_commands.csv").read_text(encoding="utf-8-sig")

# --- read the plan from inside the file unless a path is given ----------
old = '''def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no commands in it.")'''
new = '''def load_rows(path=None):
    """
    The plan, from a file if one is named and from inside this file if not.

    Carrying the plan inside the script is what makes this version a
    single file: there is no second file to copy, no folder to be in and
    nothing to go stale against it. Pass a .csv path to override it once
    a newer plan arrives.
    """
    if path is None:
        rows = list(csv.DictReader(io.StringIO(EMBEDDED_PLAN.strip())))
        path = "(built in)"
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no commands in it.")'''
assert old in src
src = src.replace(old, new, 1)

old = '''        path = pathlib.Path(args.commands)
        if not path.exists():
            sys.exit(f"No command file at {path}")

        rows = load_rows(path)'''
new = '''        if args.commands:
            path = pathlib.Path(args.commands)
            if not path.exists():
                sys.exit(f"No command file at {path}")
            rows = load_rows(path)
        else:
            path = "(built in)"
            rows = load_rows()'''
assert old in src
src = src.replace(old, new, 1)

old = '''    ap.add_argument("commands", nargs="?",
                    default="../spraying/dc_commands.csv")'''
new = '''    ap.add_argument("commands", nargs="?", default=None,
                    help="a .csv plan to use instead of the built-in one")'''
assert old in src
src = src.replace(old, new, 1)

src = src.replace("import csv\nimport math", "import csv\nimport io\nimport math", 1)

# --- append the plan, then the main guard, in that order ----------------
guard = '''if __name__ == "__main__":
    main()
'''
had_guard = guard in src
if had_guard:
    src = src.replace(guard, "", 1)
else:
    import re
    m = re.search(r'\nif __name__ == "__main__":\n(?:    .*\n|\n)*', src)
    guard = m.group(0).lstrip("\n")
    src = src[:m.start()] + src[m.end():]

src = src.rstrip() + '\n\n\n# ' + '-' * 68 + '''
# The plan. Rewrite this block to change where the machine goes - or pass
# a .csv on the command line to use a different one without touching it.
# ''' + '-' * 68 + '\nEMBEDDED_PLAN = """\\n' + plan.strip() + '\n"""\n\n\n' + guard

out = root / "handoff_pi" / "weeder.py"
out.write_text(src, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
