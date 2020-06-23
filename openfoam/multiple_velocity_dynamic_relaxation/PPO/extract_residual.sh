#!/bin/bash          
LOG=$1 #log.simpleFoam-06.02.2020-21.42.17
cat $LOG | grep -w 'Time' | cut -d' ' -f3 | tr -d ',' > ./logs/time.txt
cat $LOG | grep 'Solving for Ux' | cut -d' ' -f9 | tr -d ',' > ./logs/resx.txt
cat $LOG | grep 'Solving for Uy' | cut -d' ' -f9 | tr -d ',' > ./logs/resy.txt
cat $LOG | grep 'Solving for p' | cut -d' ' -f9 | tr -d ',' > ./logs/resp.txt
cat $LOG | grep 'Solving for epsilon' | cut -d' ' -f9 | tr -d ',' > ./logs/reseps.txt
cat $LOG | grep 'Solving for k' | cut -d' ' -f9 | tr -d ',' > ./logs/resk.txt
sed '/^$/d' ./logs/time.txt > ./logs/time1.txt
paste ./logs/time1.txt ./logs/resx.txt ./logs/resy.txt ./logs/resp.txt ./logs/reseps.txt ./logs/resk.txt | column -s $'\t' -t > ./logs/residual.txt
