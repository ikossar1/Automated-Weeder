"""
Saved answers, so a demo does not mean typing fifteen numbers.

    python mapping/new_bed.py  --list-presets
    python mapping/new_bed.py  --preset demo
    python mapping/new_bed.py  --save-preset demo
    python mapping/new_bed.py  --delete-preset demo

A preset is just the answers you would have typed. Call one and every
prompt it covers is filled in and skipped; anything it does not cover is
still asked. Saving one records whatever you just answered.

WHY THIS EXISTS
    Standing in front of a room is the worst possible time to be typing
    "30.5" into a prompt. One wrong digit and the map is quietly wrong, and
    you will not know until the picture looks odd.

    It is also how a result becomes repeatable. "We measured 62% coverage"
    is worth more when the exact settings that produced it are on disk
    rather than in somebody's memory.

WHERE THEY LIVE
    presets.json at the top of the repo. Plain JSON, so you can read it,
    edit it by hand, and commit it - the settings behind a result belong
    with the code that produced it.
"""
from __future__ import annotations

import json
import pathlib

FILE = pathlib.Path(__file__).parent / "presets.json"


def load_all():
    if not FILE.exists():
        return {}
    try:
        return json.loads(FILE.read_text())
    except Exception:
        return {}


def get(name):
    """One preset, or None. Returns a copy, so callers can edit freely."""
    return dict(load_all().get(name, {})) or None


def save(name, values):
    """Record a preset. Overwrites one of the same name."""
    allp = load_all()
    allp[name] = {k: v for k, v in values.items() if v is not None}
    FILE.write_text(json.dumps(allp, indent=2, sort_keys=True))
    return FILE


def delete(name):
    allp = load_all()
    if name not in allp:
        return False
    del allp[name]
    FILE.write_text(json.dumps(allp, indent=2, sort_keys=True))
    return True


def reset():
    """Remove every preset. Asked for by name, never done quietly."""
    if FILE.exists():
        FILE.unlink()
        return True
    return False


def describe(name=None):
    """Print the presets, or the detail of one."""
    allp = load_all()
    if not allp:
        print("\nNo presets saved yet.")
        print("Run something the normal way, then add --save-preset <name>.")
        return

    if name:
        p = allp.get(name)
        if not p:
            print(f"\nNo preset called '{name}'. There is: "
                  f"{', '.join(sorted(allp))}")
            return
        print(f"\n{name}")
        for k in sorted(p):
            print(f"    {k:20s} {p[k]}")
        return

    print("\nSaved presets:\n")
    for n in sorted(allp):
        keys = sorted(allp[n])
        head = ", ".join(keys[:5])
        more = f", +{len(keys) - 5} more" if len(keys) > 5 else ""
        print(f"  {n:16s} {head}{more}")
    print(f"\n  {FILE}")


class Answers:
    """
    Asks, unless a preset already answered.

    Wrap the normal ask() with this and the script works the same whether
    a preset is in play or not - the only difference is which prompts
    appear. Everything it ends up with is collected, so --save-preset can
    record the whole run.
    """

    def __init__(self, preset=None, asker=input):
        self.preset = preset or {}
        self.used = {}
        self._asker = asker

    def ask(self, key, prompt, default=None, cast=str):
        if key in self.preset:
            value = self.preset[key]
            try:
                value = cast(value)
            except (TypeError, ValueError):
                pass
            print(f"{prompt}: {value}   (preset)")
            self.used[key] = value
            return value

        suffix = f" [{default}]" if default is not None else ""
        while True:
            raw = self._asker(f"{prompt}{suffix}: ").strip()
            if not raw and default is not None:
                self.used[key] = default
                return default
            if not raw:
                continue
            try:
                value = cast(raw)
            except ValueError:
                print("  not a number, try again")
                continue
            self.used[key] = value
            return value

    def confirm(self, key, prompt, default=False):
        """A yes/no that a preset can answer with true or false."""
        if key in self.preset:
            value = bool(self.preset[key])
            print(f"{prompt}: {'yes' if value else 'no'}   (preset)")
            self.used[key] = value
            return value
        raw = self._asker(prompt).strip().lower()
        value = raw.startswith("y") if raw else default
        self.used[key] = value
        return value

    def record(self, key, value):
        """Stash something the user chose by other means, like clicking."""
        self.used[key] = value
        return value


def add_arguments(ap):
    """The four preset flags, for any script's argument parser."""
    g = ap.add_argument_group("presets")
    g.add_argument("--preset", metavar="NAME",
                   help="use saved answers and skip those prompts")
    g.add_argument("--save-preset", metavar="NAME",
                   help="save this run's answers under NAME")
    g.add_argument("--list-presets", action="store_true",
                   help="show what is saved, then exit")
    g.add_argument("--delete-preset", metavar="NAME",
                   help="remove one preset, then exit")
    g.add_argument("--reset-presets", action="store_true",
                   help="remove ALL presets, then exit")
    return ap


def handle_flags(args):
    """
    Deal with the listing and removing flags. Returns True if the script
    should stop now.
    """
    if getattr(args, "list_presets", False):
        describe(getattr(args, "preset", None))
        return True

    if getattr(args, "delete_preset", None):
        name = args.delete_preset
        print(f"\n{'Deleted' if delete(name) else 'No preset called'} "
              f"'{name}'")
        return True

    if getattr(args, "reset_presets", False):
        print("\nAll presets removed." if reset() else "\nThere were none.")
        return True

    return False
