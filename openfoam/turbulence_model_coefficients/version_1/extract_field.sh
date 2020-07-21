#!/bin/bash          
DIR=$1
FOLD=$2
LOG=$3 
#echo $LOG
rm ./$DIR/$LOG\_*.txt
cat $DIR/$FOLD/$LOG | grep '(' |  tr -d '()' > ./$DIR/$LOG\_$FOLD.txt
head -n -3 ./$DIR/$LOG\_$FOLD.txt > ./$DIR/$LOG\_$FOLD\_t.txt
