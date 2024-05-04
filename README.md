# prepare train/val data
'''mkdir data & cd data'''
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/75b7a729-c114-4d31-881e-23b6d7fbbba7)
## make train.txt
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/cae43883-760f-44aa-8e43-8d420e8742fc)

## create train.jsonl
'''
cd tools
python convert_data.py --config_path=../config/config.yaml
'''
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/74e9100f-e0eb-4532-b422-a6c91f1d9c54)

# train
modify config.yaml for user config
Note: runner_type = train_runner
'''
python run.py --config_path=config/config.yaml
'''

# test
runner_type = test_runner
'''
python run.py --config_path=config/config.yaml
'''

# deploy
suported pt to onnx
runner_type = deploy_runner
'''
python run.py --config_path=config/config.yaml
'''

# infer demo
'''
python infer.py
'''
