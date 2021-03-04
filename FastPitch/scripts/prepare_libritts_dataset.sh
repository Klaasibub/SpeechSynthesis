#!/usr/bin/env bash

set -e

DATA_DIR="/home/zolkin/SpeechSynthesis/FastPitch/LibriTTS"
TACOTRON2="pretrained_models/tacotron2/nvidia_tacotron2pyt_fp16.pt"
for FILELIST in libritts/train_fix_paths.txt \
                libritts/val_fix_paths.txt \
                libritts/test_fix_paths.txt \
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
