#!/usr/bin/env bash

set -e

DATA_DIR="/storage/zolkin/datasets/small_dataset"
TACOTRON2="pretrained_models/tacotron2/nvidia_tacotron2pyt_fp16.pt"
for FILELIST in small/val_without_speakerID.txt \
                small/train_without_speakerID.txt \
; do
    python extract_mels.py \
        --cuda \
        --dataset-path ${DATA_DIR} \
        --wav-text-filelist filelists/${FILELIST} \
        --batch-size 256 \
        --extract-mels \
        --extract-durations \
        --extract-pitch-char \
        --tacotron2-checkpoint ${TACOTRON2}
done
