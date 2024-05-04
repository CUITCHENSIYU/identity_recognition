import sys
import os
sys.path.append(os.path.abspath('.'))
from utils.registry import get_module
from utils.registry import load_modules

def build_pipeline(config, **kwargs):
    load_modules(__file__, "pipeline")
    return get_module("pipeline",  config['model']['pipeline_name'])(config, **kwargs)