#!/bin/bash          
DIR=$1
FOLD=$2
#echo $LOG
sed '23,15022!d' ./$DIR/$FOLD/mag\(\U\) > ./$DIR/$FOLD/mag\_u.txt
