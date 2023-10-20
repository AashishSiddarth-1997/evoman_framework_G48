#!/bin/bash

# This script is used to run the experiment in a loop until the experiment is finished.

EXPERIMENT_NAME="Custom_generalist"
PLAYER_MODE="ai"
# ENEMIES="1,3,6,7"
# ENEMIES="1,2,3,4,5,6,7,8"
ENEMIES="1,4,6"
# 1 3 4 6 or 1 3 4 6 7 or 3 4 6 7 8 or 3 4 6 8
CONTROLLER="generalist"
MULTI_MODE="yes"
SPEED="fastest"
ENEMY_MODE="static"
LEVEL=2
VISUALS=0
GENS=50
ITER=10
PARALLEL=1

conda init

python train.py $EXPERIMENT_NAME $PLAYER_MODE $ENEMIES $CONTROLLER $MULTI_MODE $SPEED $ENEMY_MODE $LEVEL $VISUALS $GENS $ITER $PARALLEL

exit 0
	