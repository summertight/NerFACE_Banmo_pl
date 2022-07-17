from tracemalloc import start
import torch

import os

from runners.nerface import RunnerNerface
from runners.nerface_trans_calib import RunnerNerface_trans_calib
from runners.nerface_DNerf_pervideo import RunnerNerface_DNerf_pervideo
from runners.nerface_DNerf_perframe import RunnerNerface_DNerf_perframe

from models import get_model_pack
from utils.functions import load_ckpt


def get_runner(cfg):
    runner = None

    ##### Ready for running
    # Set model-pack
    start_epoch = 0 
    model_pack = get_model_pack(cfg)

    # Set checkpoint
    if cfg.resume_path is not None and os.path.exists(cfg.resume_path):
        ckpt_data = torch.load(cfg.resume_path)
        model_pack = load_ckpt(cfg, model_pack, ckpt_data)
        # start_epoch = ckpt_data['epoch']
        print('Load pre-trained checkpoint is done...!')

    if cfg.method == 'nerface':
        runner = RunnerNerface(cfg, model_pack, start_epoch)
    if cfg.method == 'nerface_no_expr':
        runner = RunnerNerface(cfg, model_pack, start_epoch)
    if cfg.method == 'nerface_trans_calib':
        runner = RunnerNerface_trans_calib(cfg, model_pack, start_epoch)
    if cfg.method == 'nerface_DNerf_pervideo':
        runner = RunnerNerface_DNerf_pervideo(cfg, model_pack, start_epoch)
    if cfg.method == 'nerface_DNerf_perframe':
        runner = RunnerNerface_DNerf_perframe(cfg, model_pack, start_epoch)

    

    return runner


def get_runner_class(cfg):
    _class = None
    if cfg.method == 'nerface':
        _class = RunnerNerface
    if cfg.method == 'nerface_no_expr':
        _class = RunnerNerface
    if cfg.method == 'nerface_trans_calib':
        _class = RunnerNerface_trans_calib
    if cfg.method == 'nerface_DNerf_pervideo':
        _class = RunnerNerface_DNerf_pervideo
    if cfg.method == 'nerface_DNerf_perframe':
        _class = RunnerNerface_DNerf_perframe    
   

    return _class
