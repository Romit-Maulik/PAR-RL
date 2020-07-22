#!/bin/sh
rm -r [1-9]*
simpleFoam > log.simpleFoam
foamToVTK
