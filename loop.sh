#!/bin/bash

# This script is used to run the experiment in a loop until the experiment is finished.

END=1

EXPERIMENT_NAME="Custom_generalist"
PLAYER_MODE="ai"
ENEMIES="1,2"
CONTROLLER="generalist"
MULTI_MODE="yes"
SPEED="fastest"
ENEMY_MODE="static"
LEVEL=2
VISUALS="True"
GENS=30
ITER=1
PARALLEL="False"

conda init

if [ ! -d "$EXPERIMENT_NAME" ]; then
	mkdir $EXPERIMENT_NAME
fi

while [ $ITER -le $END ]
do 
	python train.py $EXPERIMENT_NAME $PLAYER_MODE $ENEMIES $CONTROLLER $MULTI_MODE $SPEED $ENEMY_MODE $LEVEL $VISUALS $GENS $ITER $PARALLEL
	ITER=$((ITER+1))
done

exit 0
	