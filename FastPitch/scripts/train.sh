#!/bin/bash

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=2}
: ${BS:=20}
: ${GRAD_ACCUMULATION:=2}
: ${OUTPUT_DIR:="/home/zolkin/output"}
: ${AMP:=true}
: ${EPOCHS:=2000}

# Adjust env variables to maintain the global batch size
#
#    NGPU x BS x GRAD_ACC = 256.
#
GBS=$(($NUM_GPUS * $BS * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}.\n"

echo -e "\nSetup: ${NUM_GPUS}x${BS}x${GRAD_ACCUMULATION} - global batch size ${GBS}\n"

mkdir -p "$OUTPUT_DIR"
python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} train.py \
    --amp \
    --cuda \
    --checkpoint-path "/home/zolkin/output/FastPitch_checkpoint_100.pt" \
    -o "$OUTPUT_DIR/" \
    --log-file "$OUTPUT_DIR/nvlog.json" \
    --dataset-path /home/zolkin/SpeechSynthesis/FastPitch/LibriTTS \
    --training-files filelists/libritts/train_mel_dur_pitch.txt \
    --validation-files filelists/libritts/val_mel_dur_pitch.txt \
    --pitch-mean-std-file /home/zolkin/SpeechSynthesis/FastPitch/LibriTTS/pitch_char_stats__train_fix_paths.json \
    --epochs ${EPOCHS} \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs ${BS} \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps ${GRAD_ACCUMULATION} \
    --n-speakers 10000
    ${AMP_FLAG}
