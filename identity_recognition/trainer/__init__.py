from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules


def build_trainer(config):
    load_modules(__file__, "trainers")
    return get_module("trainers",  config['train']['type'])(config)