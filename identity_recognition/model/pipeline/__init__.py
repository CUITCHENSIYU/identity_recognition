from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules

def build_pipeline(config, **kwargs):
    load_modules(__file__, "pipeline")
    return get_module("pipeline",  config['model']['pipeline_name'])(config, **kwargs)