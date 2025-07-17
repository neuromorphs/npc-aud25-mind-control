#!/bin/bash

#muselsl stream --address 00:55:DA:BB:A8:44 &
# or
muselsl stream --address 00:55:DA:BB:A6:65 &
muselsl view & 

python experiment5.py

# then run python dog-race.py <threshold>

