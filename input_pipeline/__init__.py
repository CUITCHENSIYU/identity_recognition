import sys
import os
sys.path.append(os.path.abspath('.'))
from utils.registry import get_module
import yaml
from utils.registry import load_modules

def build_dataloader(config, split=None):
    load_modules(__file__, "input_pipelines")
    return get_module("input_pipelines",  config['train']['dataset_type'])(config, split=split)

if __name__ == "__main__":
    config_file = open('/home/csy/workspace/cleaning-activity-recognition/configs/config.yaml', 'r')
    config = yaml.safe_load(config_file)
    data_loader = get_dataloader(config, 'test')
    for i, data in enumerate(data_loader, 0):
        input, label= data
        
        print(label)
                