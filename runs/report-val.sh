#!/usr/bin/env bash
exp=$1
metric=$2

if [[ "$#" -ne 2 ]]; then
    echo "Usage: <exp/dir> <Accuracy|MacroF1|MicroF1>"
    exit 2;
fi 

files=$(ls $exp/models/validation-* |
        while read i; do
            echo $(echo $i | grep -o '[0-9]*000.txt' | sed 's/\.txt//') $i
        done |
        sort -n |
	cut -f2 -d' '
     )
#echo $files
grep $metric $files | sed 's:/models/validation-:\t:;s/.txt:/\t/'
