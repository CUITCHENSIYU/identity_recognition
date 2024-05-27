from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules

def build_criterions(config):
    load_modules(__file__, "criterions")
    return get_module("criterions",  config['train']['criterions'])(config)