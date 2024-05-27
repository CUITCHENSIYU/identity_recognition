from identity_recognition.utils.registry import get_module
import yaml
from identity_recognition.utils.registry import load_modules

def build_dataloader(config, split=None):
    load_modules(__file__, "input_pipelines")
    return get_module("input_pipelines",  config['train']['dataset_type'])(config, split=split)
                