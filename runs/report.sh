#!/usr/bin/env bash

echo "Experiment MacroF1 Accuracy"
for i in  */result*val.txt ; do
    n=$(dirname $i);
    macf1=$(grep 'MacroF1' $i | awk '{print $2}');
    acc=$(grep 'Accuracy' $i | awk '{print $2}');
    echo $n $macf1 $acc;
done
