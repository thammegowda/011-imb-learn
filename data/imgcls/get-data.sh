#!/usr/bin/env bash
# Author= Thamme Gowda
# Created = April 2021

set -e # exit on error

log_exit() { echo "$2"; exit 1; }

for cmd in mtdata wget unzip tqdm; do
  which $cmd &> /dev/null ||
    log_exit 1 "$cmd not found; please install $cmd and rerun me."
done

function get_msl {
  tgt="$1"
  [[ -e $tgt/_VALID ]] && return

  src=msl-labeled-data-set-v2.1
  [[ -f $src/_VALID ]] || {
    [[ -s $src.zip ]] || wget https://zenodo.org/record/4033453/files/$src.zip?download=1 -O $src.zip
    unzip -q $src.zip && touch $src/_VALID
    [[ -d __MACOSX ]] && rm -r __MACOSX
  }

  # restructure directory for pytorch API data loaders
  images=$src/images
  clsmap=$src/class_map.csv
  [[ -f $clsmap ]] || log_exit 2 "$clsmap file not found"
  mapfile -t classes < <(awk -F ',' '{gsub(/\s/, "_", $2); print tolower($2)}' $clsmap)

  echo "restructuring : $src -> $tgt"
  for split in train test val; do
    for l in "${classes[@]}"; do
      [[ -d $tgt/$split/$l ]] || mkdir -p $tgt/$split/$l
    done

    table=$src/$split-set-v2.1.txt
    [[ -f $table ]] || log_exit 3 "$table not found."

    cat $table | while read img cls; do
      label=${classes[$cls]}
      dst=$tgt/$split/$label/$img
      [[ -f $dst ]] || cp $images/$img $dst
    done
  done
  touch $tgt/_VALID
}


function get_hirise {
  tgt=$1
  [[ -e $tgt/_VALID ]] && return
  src=hirise-map-proj-v3_2

  [[ -f $src/_VALID ]] || {
    [[ -s $src.zip ]] || wget https://zenodo.org/record/4002935/files/$src.zip?download=1 -O $src.zip
    unzip -q $src.zip && touch $src/_VALID
    [[ -d __MACOSX ]] && rm -r __MACOSX
  }

  # restructure directory for pytorch API data loaders
  clsmap=$src/landmarks_map-proj-v3_2_classmap.csv
  table=$src/labels-map-proj_v3_2_train_val_test.txt
  images=$src/map-proj-v3_2
  [[ -f $clsmap ]] || log_exit 2 "$clsmap file not found"
  [[ -f $table ]] || log_exit 2 "$table file not found"
  mapfile -t classes < <(awk -F ',' '{gsub(/\s/, "_", $2); print tolower($2)}' $clsmap)

  for split in val test train; do
    for cls in "${classes[@]}"; do
      [[ -d $tgt/$split/$cls ]] || mkdir -p $tgt/$split/$cls
    done
  done

  echo "restructuring : $src -> $tgt"
  tqdm --total $(wc -l < ${table}) < $table | while read img cls split; do
    label=${classes[$cls]}
    dst=$tgt/$split/$label/$img
    [[ -f $dst ]] || cp $images/$img $dst
  done
  touch $tgt/_VALID
}

get_msl msl
get_hirise hirise
