#!/usr/bin/env bash
# Author= Thamme Gowda
# Created = June 2021

DIR="$(dirname "${BASH_SOURCE[0]}")"  # Get the directory name
#DIR="$(realpath "${DIR}")"    # Resolve its full path if need be

set -e # exit on error

log_exit() { echo "$2"; exit 1; }

for cmd in wget cut sed; do
  which $cmd &> /dev/null ||
    log_exit 1 "$cmd not found; please install $cmd and rerun me."
done

function get_trec {
    dest="$1"
    [[ -e $dest/_VALID ]] && return
    [[ -d $dest ]] || mkdir -p $dest
    TRAIN="https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
    TEST="https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
    train=$dest/train_5500.orig
    test=$dest/TREC_10.orig
    [[ -s $train ]] || wget "$TRAIN" -O "$train"
    [[ -s $test ]] || wget "$TEST" -O "$test"
    printf "$dest/train $train\n$dest/test $test\n" |
        while read pref file; do
            cat $file | cut -f1 -d' ' | cut -f1 -d: > $pref.coarse
            cat $file | cut -f1 -d' ' | cut -f2 -d: > $pref.fine
            cat $file | sed 's/^[^ ]* //' > $pref.text
        done
    echo "TREC datset is ready at $dest"
    touch $dest/_VALID
   
}

get_trec $DIR/trec
