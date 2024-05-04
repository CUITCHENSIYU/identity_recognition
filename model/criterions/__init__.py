import sys
import os
sys.path.append(os.path.abspath('.'))
from utils.registry import get_module
from utils.registry import load_modules

def build_criterions(config):
    load_modules(__file__, "criterions")
    return get_module("criterions",  config['train']['criterions'])(config)