import os
import re
import sys
import time
import math
import argparse
from shutil import copyfile

import numpy as np

import torch
from torch.utils.data import DataLoader

from tps import Handler, cleaners

from model import load_model
from utils.data_utils import TextMelLoader
from hparams import create_hparams
from utils import utils as utl

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    return model


def inference(hparams):

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    model = load_checkpoint(hparams.checkpoint, model)

    if hparams.fp16_run:
        from apex import amp
        model = amp.initialize(model, opt_level="O2")

    _ = model.cuda().eval()

    text_handler = Handler.from_config(hparams.text_handler_cfg)
    textLoader = TextMelLoader(text_handler, hparams.inference_files, hparams)

    text = """где хохлатые хохотушки хохотом хохотали и кричали турке, который начерно обкурен трубкой: не кури, турка, трубку, купи лучше кипу пик, лучше пик кипу купи,
              а то придет бомбардир из Бранденбурга — бомбами забомбардирует"""
    sequence = textLoader.get_text(text)
    sequence = np.array(sequence)[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # ================ INFERENCE! ===================
    outputs = model.inference(sequence)
    for idx, mel in enumerate(outputs.mels):
        filename = f'../sova-tts-vocoder/mels/audio_{idx}.pt'
        torch.save(mel, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams_path", type=str, default="./data/hparams.yaml",
                        required=False, help="hparams path")
    args = parser.parse_args()

    hparams = create_hparams(args.hparams_path)
    hparams.path = args.hparams_path

    device = hparams.device.split(":")
    device = device[0] + ":0" if len(device) == 1 else ":".join(device)

    device = torch.device(device)

    if device.type != "cpu":
        assert torch.cuda.is_available()

        torch.cuda.set_device(device)

        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    inference(hparams)
