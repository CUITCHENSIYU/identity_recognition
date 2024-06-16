# install
```
cd identity_recognition
python3 -m pip install -e .
```
Here `-e` means editable mode, which is optional.

# prepare train/val data

you should modify train.txt/val.txt to your data path, first!
```
remove your data in data/ or data2/
```
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/75b7a729-c114-4d31-881e-23b6d7fbbba7)
## make train.txt
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/cae43883-760f-44aa-8e43-8d420e8742fc)


`Note` if you use mat format data: 
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/768b670f-c44f-46ea-9ae4-84827e5a1ca8)


## create train.jsonl
```
python tools/convert_data.py --config_path=../config/config.yaml --data_file=$(your file path) --save_file=$(save path)
```
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/74e9100f-e0eb-4532-b422-a6c91f1d9c54)


`Note` if you use mat format data:
```
python tools/convert_data_mat.py --config_path=../config/config_mat.yaml --data_file=$(your file path) --save_file=$(save path)
```

# train
modify config.yaml for user config
Note: runner_type = train_runner
```
python run.py --config_path=config/config.yaml
```
`Note` if you use mat format data:
```
--config_path=../config/config_mat.yaml
```

# test
runner_type = test_runner
```
python run.py --config_path=config/config.yaml
```
`Note` if you use mat format data:
```
--config_path=../config/config_mat.yaml
```
# deploy
suported pt to onnx
runner_type = deploy_runner
```
python run.py --config_path=config/config.yaml
```
`Note` if you use mat format data:
```
--config_path=../config/config_mat.yaml
```
# infer demo
```
python infer.py
```
input :EEG data(type = np.array), condition: time length>=1s.
return: identity_map：
| id | user_name | score | count |
| :----: | :----: | :----: | :----: |
| 0 | gjc | - | - |
| 1 | wxc | - | - |
| 2 | yl | - | - |
| 3 | zqy | - | - |
| -1 | unknown | 0 | 0 |

if you want add or delete, please update config.yaml

![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/6aba7815-a4e8-4004-b481-858ac0865719)

`Note` if you use mat format data, please update config_mat.yaml
