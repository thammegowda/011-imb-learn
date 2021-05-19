#!/usr/bin/env bash

echo "Experiment,Step,MacroF1,Accuracy,MicroF1"
for i in  */result.step*val.txt ; do
    n=$(dirname $i);
    step=$(basename $i | grep -o 'step[0-9]*' | sed 's/step//')
    macf1=$(grep 'MacroF1' $i | cut -f2 -d,);
    acc=$(grep 'Accuracy' $i | cut -f2 -d,);
    micf1=$(grep 'MicroF1' $i | cut -f2 -d,);    
    echo $n $step $macf1 $acc $micf1 | sed 's/ %//g' | tr ' ' ',';
done
