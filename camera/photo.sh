#!/bin/bash
# Take a photo and put it in mapping/photos as <bed>_<number>.jpg
read -p "Bed name: " bed
n=1
while [ -e "../mapping/photos/${bed}_$n.jpg" ]; do n=$((n + 1)); done
rpicam-still -n -t 2000 -o "../mapping/photos/${bed}_$n.jpg"
echo "Saved ${bed}_$n.jpg"
