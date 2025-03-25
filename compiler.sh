#!/bin/bash

# Clean first
make clean

# Time the compilation and execution
make COMPILER=$1

time (
    ./graphics/fire_animation_data ./data/2000_8 | python3 ./graphics/fire_animation.py 2000_8_fire_animation.mp4
)
