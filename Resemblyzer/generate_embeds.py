from resemblyzer import VoiceEncoder, preprocess_wav
import librosa
from tqdm import tqdm
import numpy as np
import os


def main():
    encoder = VoiceEncoder()
    ds_dir = 'D:/Рабочий Стол/natasha_dataset'
    os.makedirs(ds_dir+'/embeds/', exist_ok=True)
    out = open(f'{ds_dir}/with_embeds.txt', 'w', encoding='utf-8')
    with open(f'{ds_dir}/marks.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            filename = line.split("|")[0].split('/')[-1].split('.')[0]
            audio_fpath = f'{ds_dir}/{line.split("|")[0]}'
            if audio_fpath.endswith('.npy'):
                wav = np.load(audio_fpath)
            else:
                wav, _ = librosa.load(audio_fpath)
            embed = encoder.embed_utterance(wav)

            audio_fpath = f'{ds_dir}/wavs/{filename}.wav'
            embed_fpath = f'{ds_dir}/embeds/{filename}.npy'
            text = line.split('|')[-1]

            np.save(embed_fpath, embed)
            out.write(f'{audio_fpath}|{embed_fpath}|{text}\n')
    out.close()


if __name__ == "__main__":
    # TODO: Add argparse for ds_dir and meta.
    main()
