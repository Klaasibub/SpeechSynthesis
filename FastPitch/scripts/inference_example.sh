#!/usr/bin/env bash

: ${WAVEGLOW:="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"}
: ${FASTPITCH:="/home/zolkin/output/FastPitch_checkpoint_100.pt"} # "/storage/zolkin/output/FastPitch_checkpoint_900.pt"}
: ${BS:=20}
: ${PHRASES:="phrases/devset10.tsv"}
: ${OUTPUT_DIR:="/home/zolkin/output/audio_$(basename ${PHRASES} .tsv)"}
: ${AMP:=false}

[ "$AMP" = true ] && AMP_FLAG="--amp"

mkdir -p "$OUTPUT_DIR"

python inference.py --cuda \
                    -i ${PHRASES} \
                    -o ${OUTPUT_DIR} \
                    --fastpitch ${FASTPITCH} \
                    --waveglow ${WAVEGLOW} \
		    --wn-channels 256 \
                    --batch-size ${BS} \
                    --n-speakers 10000 \
                    --speaker 19 \
                    ${AMP_FLAG}
