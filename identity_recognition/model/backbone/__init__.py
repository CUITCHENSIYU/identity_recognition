import sys
import os
sys.path.append(os.path.abspath('.'))
from utils.registry import get_module
from utils.registry import load_modules

def build_backbone(config):
    load_modules(__file__, "backbone")
    return get_module("backbone",  config['model']['backbone_name'])(config)