
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
from typing import Optional, Union
import struct
from scipy.ndimage.morphology import binary_dilation
import webrtcvad
import audio
from hparams import hparams
import soundfile as sf

out_dir = '.'
skip_existing = True

rescale = True
sample_rate = 16000
trim_silence = True
hop_size = 200
win_size = 800
rescaling_max = 0.9
utterance_min_duration = 1.6
max_mel_frames = 900
clip_mels_length = True

audio_norm_target_dBFS = -30
vad_window_length = 30  # In milliseconds
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6
int16_max = (2 ** 15) - 1


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav):
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sample_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sample_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sample_rate:
        wav = librosa.resample(wav, source_sr, sample_rate)

    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, skip_existing: bool):
    mel_fpath = f'mel-{basename}.npy'
    wav_fpath = f'wav-{basename}.npy'

    if trim_silence:
        wav = preprocess_wav(wav, normalize=False, trim_silence=True)
    
    if len(wav) < utterance_min_duration * sample_rate:
        return None

    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    if mel_frames > max_mel_frames and clip_mels_length:
        return None
    
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    return wav


if __name__ == "__main__":
    ds_dir = '/home/sidenko/my/SV2TTS/synthesizer'
    with open(f'{ds_dir}/filelists.txt', 'r') as f:
        for line in f.readlines():
            audio_fpath = f'{ds_dir}/audio/{line.split("|")[0]}'
            wav = np.load(audio_fpath)
            