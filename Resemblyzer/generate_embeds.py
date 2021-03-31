from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np


def main():
    encoder = VoiceEncoder()
    ds_dir = '/home/sidenko/my/SV2TTS/synthesizer'
    with open(f'{ds_dir}/filelists.txt', 'r') as f:
        for line in f.readlines():
            audio_fpath = f'{ds_dir}/audio/{line.split("|")[0]}'
            wav = np.load(audio_fpath)
            embed = encoder.embed_utterance(wav)
            embed_fpath = f'{ds_dir}/embeds/{line.split("|")[2]}'
            np.save(embed_fpath, embed)


if __name__ == "__main__":
    main()
            