#!/bin/bash          
LOG=$1 #log.simpleFoam-06.02.2020-21.42.17
DIR=$2
PID=$3
cat $LOG | grep -w 'Time' | cut -d' ' -f3 | tr -d ',' > ./$DIR/time_$PID.txt
cat $LOG | grep 'Solving for Ux' | cut -d' ' -f9 | tr -d ',' > ./$DIR/resx_$PID.txt
cat $LOG | grep 'Solving for Uy' | cut -d' ' -f9 | tr -d ',' > ./$DIR/resy_$PID.txt
cat $LOG | grep 'Solving for p' | cut -d' ' -f9 | tr -d ',' > ./$DIR/resp_$PID.txt
cat $LOG | grep 'Solving for nuTilda' | cut -d' ' -f9 | tr -d ',' > ./$DIR/resnutilda_$PID.txt
sed '/^$/d' ./$DIR/time_$PID.txt > ./$DIR/time1_$PID.txt
paste ./$DIR/time1_$PID.txt ./$DIR/resx_$PID.txt ./$DIR/resy_$PID.txt ./$DIR/resp_$PID.txt ./$DIR/resnutilda_$PID.txt | column -s $'\t' -t > ./$DIR/residual_$PID.txt
