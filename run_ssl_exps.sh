#!/bin/bash

set -e

SUBJECTS=1
EXPS=1
GPU=0

declare -a DATASET_NAMES=(nback)
for DATASET_NAME in "${DATASET_NAMES[@]}"
  do
  python3 dataset_generator.py --dataset "$DATASET_NAME"

    if [ "$DATASET_NAME" == "mmi" ]
      then
        declare -a TASKS=(open_closed t1_t2_t3_t4)
      else
        declare -a TASKS=(baseline_n1_n2 baseline_n3 n1_n2 n0_n1_n2_n3)
    fi



    for CLASS_COMBO in "${TASKS[@]}"
      do
        for exp in $(seq 1 $EXPS);
          do
            for subject in $(seq 1 $SUBJECTS);
            do
               python3 train_ssl_lightning.py --GPU "$GPU" --exp "$exp" --train_type 'subject_independent' --class_combo "$CLASS_COMBO" --subject "$subject" --dataset_name "$DATASET_NAME"  --continue &
               #python3 train_ssl_lightning.py --GPU "$GPU" --exp "$exp" --train_type 'subject_dependent' --class_combo "$CLASS_COMBO" --subject "$subject" --dataset_name "$DATASET_NAME"  --continue &
               wait;
            done
          done
      done
done

