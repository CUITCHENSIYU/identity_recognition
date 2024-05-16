import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
from identity_recognition.trainer import build_trainer
import argparse
import yaml
from identity_recognition.test import Test
from identity_recognition.deploy import Deploy

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./config/config.yaml", type=str)
args = parser.parse_args()

def main():
    cfg_file = open(args.config_path, 'r')
    cfg = yaml.safe_load(cfg_file)
    print(cfg)
    if cfg['runner_type']=='train_runner':
        trainer = build_trainer(cfg)
        trainer.train() 
    if cfg['runner_type']=='test_runner':
        test = Test(cfg)
        test.run()
    if cfg['runner_type']=='deploy_runner':
        deploy = Deploy(cfg)
        deploy.run()
main()