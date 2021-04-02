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


def inference(hparams, checkpoint_path, output_path):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    if not checkpoint_path:
        model = load_checkpoint(hparams.checkpoint, model)
    else:
        model = load_checkpoint(checkpoint_path, model)

    if hparams.fp16_run:
        from apex import amp
        model = amp.initialize(model, opt_level="O2")

    _ = model.cuda().eval()

    os.makedirs(output_path, exist_ok=True)

    embed_path = "/home/sidenko/my/output/inf/embed-common_voice_ru_18849004.npy"
    out_fname = embed_path.split('-')[-1].split('.')[0]
    embed = torch.from_numpy(np.load(embed_path))

    text_handler = Handler.from_config(hparams.text_handler_cfg)
    textLoader = TextMelLoader(text_handler, hparams.inference_files, hparams)

    text = """где хохлатые хохотушки хохотом хохотали и кричали турке, который начерно обкурен трубкой: не кури, турка, трубку, купи лучше кипу пик, лучше пик кипу купи,
              а то придет бомбардир из Бранденбурга — бомбами забомбардирует"""

    text = textLoader.get_text(text)
    text = np.array(text)[None, :]
    text = torch.from_numpy(text).to(device='cuda', dtype=torch.int64)

    # ================ INFERENCE! ===================
    outputs = model.inference((text, embed))
    for idx, mel in enumerate(outputs.mels):
        filename = f'{output_path}/{out_fname}.pt'
        torch.save(mel, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams_path", type=str, default="./data/hparams.yaml",
                        required=False, help="hparams path")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None,
                        required=False, help="checkpoint path")
    parser.add_argument("-o", "--output_path", type=str, default='inference_output',
                        required=False, help="output path")
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

    inference(hparams, args.checkpoint_path, args.output_path)
