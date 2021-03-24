import os
import re
import time
import math
import argparse
from shutil import copyfile
from itertools import chain

import numpy as np
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tps import Handler

from model import load_model
from utils.data_utils import TextMelLoader, TextMelCollate, CustomSampler
from utils.distributed import apply_gradient_allreduce
from modules.optimizers import build_optimizer, build_scheduler, SchedulerTypes
from modules.loss_function import OverallLoss
from hparams import create_hparams
from utils import gradient_adaptive_factor


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def reduce_loss(loss, distributed_run, n_gpus):
    return reduce_tensor(loss.data, n_gpus).item() if distributed_run else loss.item()


def calc_gaf(model, optimizer, loss1, loss2, max_gaf):
    safe_loss = 0. * sum([x.sum() for x in model.parameters()])

    gaf = gradient_adaptive_factor.calc_grad_adapt_factor(
        loss1 + safe_loss, loss2 + safe_loss, model.parameters(), optimizer)
    gaf = min(gaf, max_gaf)

    return gaf


def init_distributed(hparams, n_gpus, rank, group_name):
    print("Initializing Distributed")

    # Initialize distributed communication
    dist.init_process_group(backend=hparams.dist_backend, init_method=hparams.dist_url,
                            world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams, distributed_run=False):
    # Get data, data loaders and collate function ready
    assert isinstance(hparams.text_handler_cfg, str)
    text_handler = Handler.from_config(hparams.text_handler_cfg)
    text_handler.out_max_length = None
    assert text_handler.charset.value == hparams.charset

    trainset = TextMelLoader(text_handler, hparams.inference_files, hparams)
    valset = TextMelLoader(text_handler, hparams.inference_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if distributed_run:
        train_sampler = DistributedSampler(trainset)
    else:
        train_sampler = CustomSampler(trainset, hparams.batch_size, hparams.shuffle, hparams.optimize, hparams.len_diff)

    train_loader = DataLoader(trainset, num_workers=1, sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if not any(re.search(layer, k) for layer in ignore_layers)}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, criterion, restore_lr=True):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))

    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    if criterion.mmi_criterion is not None:
        criterion.mmi_criterion.load_state_dict(checkpoint_dict["mi_estimator"])

    iteration = checkpoint_dict["iteration"]

    if not restore_lr:
        base_lr = lr_scheduler.get_last_lr()
        for lr, param_group in zip(base_lr, optimizer.param_groups):
            param_group["lr"] = lr
    else:
        lr_scheduler_params = checkpoint_dict.get("lr_scheduler", None)
        if lr_scheduler_params is not None:
            lr_scheduler.load_state_dict(lr_scheduler_params)

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))

    return model, optimizer, lr_scheduler, criterion, iteration


def train(hparams, distributed_run=False, rank=0, n_gpus=None):
    if distributed_run:
        assert n_gpus is not None

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams, distributed_run)
    criterion = OverallLoss(hparams)
    if criterion.mmi_criterion is not None:
        parameters = chain(model.parameters(), criterion.mmi_criterion.parameters())
    else:
        parameters = model.parameters()
    optimizer = build_optimizer(parameters, hparams)
    lr_scheduler = build_scheduler(optimizer, hparams)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2")

    if distributed_run:
        model = apply_gradient_allreduce(model)


    # copyfile(hparams.path, os.path.join(hparams.output_dir, 'hparams.yaml'))
    train_loader, valset, collate_fn = prepare_dataloaders(hparams, distributed_run)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if hparams.checkpoint is not None:
        if hparams.warm_start:
            model = warm_start_model(
                hparams.checkpoint, model, hparams.ignore_layers)
        else:
            model, optimizer, lr_scheduler, mmi_criterion, iteration = load_checkpoint(
                hparams.checkpoint, model, optimizer, lr_scheduler, criterion, hparams.restore_scheduler_state
            )

            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.eval()
    is_overflow = False
    # ================ INFERENCE! ===================
    count_1 = 1
    count_2 = 1
    for i, batch in enumerate(train_loader):
        start = time.perf_counter()
        inputs, alignments, inputs_ctc = model.parse_batch(batch)
        outputs, decoder_outputs = model(inputs)
        duration = time.perf_counter() - start
        print(f'{duration} sec per batch')
        
        for mel in inputs.mels:
            filename = f'/home/sidenko/my/sova-tts-vocoder/mels/original/audio_{count_1}.pt'
            mel = torch.save(mel, filename)
            count_1 += 1

        for mel in outputs.mels:        
            filename = f'/home/sidenko/my/sova-tts-vocoder/mels/audio_{count_2}.pt'
            mel = torch.save(mel, filename)
            count_2 += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams_path", type=str, default="./data/hparams.yaml",
                        required=False, help="hparams path")
    parser.add_argument("-d", "--distributed_run", action="store_true",
                        required=False, help="switch script to distributed work mode")
    parser.add_argument("--gpus_ranks", type=str, default="",
                        required=False, help="gpu's indices for distributed run (separated by commas)")
    parser.add_argument("--gpu_idx", type=int, default=0,
                        required=False, help="device index for the current run")
    parser.add_argument("--group_name", type=str, default="group_name",
                        required=False, help="Distributed group name")
    args = parser.parse_args()

    hparams = create_hparams(args.hparams_path)
    hparams.path = args.hparams_path

    n_gpus = 0
    rank = 0

    if args.distributed_run:
        assert args.gpus_ranks
        gpus_ranks = {elem: i for i, elem in enumerate(int(elem) for elem in args.gpus_ranks.split(","))}
        n_gpus = len(gpus_ranks)
        rank = gpus_ranks[args.gpu_idx]

        device = "cuda:{}".format(args.gpu_idx)
    else:
        device = hparams.device.split(":")
        device = device[0] + ":0" if len(device) == 1 else ":".join(device)

    device = torch.device(device)

    if device.type != "cpu":
        assert torch.cuda.is_available()

        torch.cuda.set_device(device)
        if args.distributed_run:
            init_distributed(hparams, n_gpus, rank, args.group_name)

        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    else:
        assert not args.distributed_run

    hparams.learning_rate = float(hparams.learning_rate)
    hparams.weight_decay = float(hparams.weight_decay)

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", args.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(hparams, distributed_run=args.distributed_run, rank=rank, n_gpus=n_gpus)
