# camera

Point the machine, take a photo, and file it where mapping looks for it.

## Taking a photo

    python3 take_photo.py

It asks what the bed is called, runs the camera, and saves the picture
into `mapping/photos/` already named. Or name it up front:

    python3 take_photo.py basil

## What the names mean

    <bed>_<number>.jpg

    basil_1.jpg      first photo of the basil bed
    basil_2.jpg      second photo of the same bed
    bed_7.jpg        seventh photo of a bed just called "bed"

The **bed** part is yours to choose and should say where the photo was
taken - `basil`, `northrow`, `demo`. The **number** is the next free one
for that bed, picked automatically, so taking another photo never
overwrites an earlier one.

This matters more than it looks. The camera writes the same filename
every time it runs, so without renaming, each capture either destroys
the last one or turns into photo1, photo2, photo_final. Named on the
way in, a map made months later can still be traced back to the exact
frame it came from.

To see what has been taken:

    python3 take_photo.py --list

## Choosing one to map

    cd ../mapping
    python3 new_bed.py

That lists the photos newest first and asks which to use, so the name
is what you pick from. Pick the newest `basil_*` and you are mapping
the basil bed; the number tells you which attempt.

## Framing the shot

    python3 move_camera.py

Drives the machine by hand so the bed fills the frame before you shoot.
Arrow keys pan and tilt, J and K move the rail, A pulses the valve, q
quits. Framing is a camera job - there is nothing to map until the bed
is actually in the picture.

The valve on this machine latches: a pulse opens it and a pulse the
other way shuts it, so quitting pulses it shut rather than just
dropping the pins.
