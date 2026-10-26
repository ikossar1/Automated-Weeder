# Photo naming

Every photo the project has ever used lives in this one folder, named:

```
<bed>_<number>.<ext>
```

| Part | What it is | Example |
|---|---|---|
| `bed` | which plot, one word, no spaces | `basil` |
| `number` | which shot of that bed, counting from 1 | `1`, `2`, `3` |

```
basil_1.jpg      first photo of the basil bed
basil_2.jpg      second
table_1.jpeg     first photo of the table
field_1.webp
```

## How to use it

**Count up within a bed.** The next photo of the basil bed is `basil_4` if
there are already three. `new_bed.py` shows the folder before you pick, so
the next free number is right there.

**A new plot gets a new bed name**, starting again at 1.

## Why it matters here

The whole project is measurements tied to a particular photo. A corner click,
a plot size, a map and a spray route all belong to **one image**, and all of
them are silently wrong if applied to a different one.

So every photo needs a name nothing else has, and the map folder carries that
same name — `maps/basil_2/` came from `basil_2.jpg`, no guessing.

## The trade, worth knowing

**The number says nothing about when.** `basil_7` could be from any day, and
the order only tells you which came first, not how far apart.

If the date matters for the report, note it in the lab book or let the file's
own timestamp carry it. The name is for telling photos apart, not for dating
them.

## Rules

- **Never overwrite.** A new photo of the same bed takes the next number.
- **Never reuse a number**, even after deleting a photo — an old map folder
  still refers to it.
- **Never rename after a scan.** The map records its source image; renaming
  breaks the link.
- Lower case, underscores, no spaces.

## Run names follow the photo

A scan of `basil_2.jpg` should be `--out basil_2`, so `maps/basil_2/` sits
beside the photo it came from. `new_bed.py` offers that by default.
