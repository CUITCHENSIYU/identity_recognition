from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules

def build_backbone(config):
    load_modules(__file__, "backbone")
    return get_module("backbone",  config['model']['backbone_name'])(config)