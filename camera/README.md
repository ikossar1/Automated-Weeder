# camera

## Taking a photo

    ./photo.sh

Asks for a bed name, takes the photo, saves it to `mapping/photos/` as
`<bed>_<number>.jpg` with the next free number.

The whole thing is one command plus the naming:

    rpicam-still -n -t 2000 -o ../mapping/photos/basil_1.jpg

The naming is the only reason this is a script at all. The camera
writes whatever filename you give it, so without picking a new one each
time, every photo either overwrites the last or becomes photo1,
photo2, photo_final - and a map made weeks later cannot be traced back
to the frame it came from.

To see what has been taken:

    ls ../mapping/photos/

## Choosing one to map

    cd ../mapping
    python3 new_bed.py

Lists the photos newest first and asks which to use. Pick the newest
`basil_*` and you are mapping the basil bed.

## Framing the shot

    python3 ../mobility/move_camera.py

Drives the machine by hand so the bed fills the frame. Arrow keys pan
and tilt, J and K move the rail, A pulses the valve, q quits.

The valve latches - a pulse opens it and a pulse the other way shuts
it - so quitting pulses it shut rather than just dropping the pins.
