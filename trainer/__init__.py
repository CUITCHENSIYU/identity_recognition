import sys
import os

sys.path.append(os.path.abspath('.'))
from utils.registry import get_module
from utils.registry import load_modules


def build_trainer(config):
    load_modules(__file__, "trainers")
    return get_module("trainers",  config['train']['type'])(config)