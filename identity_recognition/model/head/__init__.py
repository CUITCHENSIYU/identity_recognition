from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules

def build_head(config, feature_dim):
    load_modules(__file__, "head")
    return get_module("head",  config['model']['head_name'])(config, feature_dim)