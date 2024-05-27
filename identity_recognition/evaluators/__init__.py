from identity_recognition.utils.registry import get_module
from identity_recognition.utils.registry import load_modules

def build_evaluator(task_type, config):
    load_modules(__file__, "evaluators")
    return get_module("evaluators", task_type)(config)

import yaml
if __name__ == "__main__":
    config_file = open('/home/admin123/Workspace/cleaning-activity-recognition/configs/config.yaml', 'r')
    config = yaml.safe_load(config_file)
    evaluator = build_evaluator(config['evaluator']['type'], config)
        