import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = "citeseer"
opt['weight_decay'] = 5e-2

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

run(opt)
