import os
import re
import sys
import time
import math
import argparse
from shutil import copyfile

import matplotlib
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

    text = "В четверг четвертого числа в четыре с четвертью часа лигурийский регулировщик регулировал в Лигурии"
    text = "но тридцать три корабля лавировали, лавировали, да так и не вылавировали, а потом протокол про протокол протоколом запротоколировал"
    text = "как интервьюером интервьюируемый лигурийский регулировщик речисто, да не чисто рапортовал, да не дорапортовал дорапортовывал" 
    text = "да так зарапортовался про размокропогодившуюся погоду что, дабы инцидент не стал претендентом на судебный прецедент"
    text = "лигурийский регулировщик акклиматизировался в неконституционном Константинополе,"
    text = "где хохлатые хохотушки хохотом хохотали и кричали турке, который начерно обкурен трубкой: не кури, турка, трубку, купи лучше кипу пик, лучше пик кипу купи, а то придет бомбардир из Бранденбурга — бомбами забомбардирует за то, что некто чернорылый у него полдвора рыломизрыл, вырыл и подрыл;"
    # text = """ но на самом деле турка не был в деле, да и Клара к крале в то время кралась к ларю, пока Карл у Клары кораллы крал, за что Клара у Карла украла кларнет, а потом на дворе деготниковой вдовы Варвары два этих вора дрова воровали; но грех — не смех — не уложить в орех: о Кларе с Карлом во мраке все раки шумели в драке, — вот и не до бомбардира ворам было, и не до деготниковой вдовы, и не до деготниковых детей; зато рассердившаяся вдова убрала в сарай дрова: раз дрова, два дрова, три дрова — не вместились все дрова, и два дровосека, два дровокола-дроворуба для расчувствовавшейся Варвары выдворили дрова вширь двора обратно на дровяной двор, где цапля чахла, цапля сохла, цапля сдохла; цыпленок же цапли цепко цеплялся за цепь; молодец против овец, а против молодца сам овца, которой носит Сеня сено в сани, потом везет Сеньку Соньку с Санькой на санках: санки скок, Сеньку — в бок, Соньку — в лоб, все — в сугроб, а Сашка только шапкой шишки сшиб, затем по шоссе Саша пошел, Саша на шоссе саше нашел; Сонька же — Сашкина подружка шла по шоссе и сосала сушку, да притом у Соньки-вертушки во рту еще и три ватрушки — аккурат в медовик, но ей не до медовика — Сонька и с ватрушками во рту пономаря перепономарит, — перевыпономарит: жужжит, как жужелица, жужжит, да кружится: была у Фрола — Фролу на Лавра наврала, пойдет к Лавру на Фрола Лавру наврет, что — вахмистр с вахмистршей, ротмистр с ротмистршей, что у ужа — ужата, а у ежа — ежата, а у него высокопоставленный гость унес трость, и вскоре опять пять ребят съели пять опят с полчетвертью четверика чечевицы без червоточины, и тысячу шестьсот шестьдесят шесть пирогов с творогом из сыворотки из-под простокваши, о всем о том около кола колокола звоном раззванивали, да так, что даже Константин — зальцбуржский бесперспективняк из-под бронетранспортера констатировал: как все колокола не переколоколовать, не перевыколоколовать, так и всех скороговорок не перескороговорить, не перевыскороговорить; но попытка — не пытка""" 
    text = "Я еб+ался с этой нейр+онкой больше двадцати лет... И вот, что она мне сказала: АЙ ЩИКИ БУ, КУЙЛА ХУЮК, НАМУКК..."
    sequence = textLoader.get_text(text)
    sequence = np.array(sequence)[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # ================ INFERENCE! ===================
    outputs = model.inference(sequence)
    for idx, mel in enumerate(outputs.mels):
        filename = f'/home/sidenko/my/sova-tts-vocoder/mels/audio_{idx}.pt'
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
