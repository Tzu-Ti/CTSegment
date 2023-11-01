# [VGHTC] CT Segmentation

## Training
### Data structure
```
-- Folder
    -- xxx
        -- xxx_CT.nii.gz
        -- xxx_Seg.nii.gz
    -- yyy
        -- yyy_CT.nii.gz
        -- yyy_Seg.nii.gz
    ...
```
### Preprocess
1. Convert 3D .nii.gz to 2D .npy, including CT and mask.
```=shell
$ cd data
$ python3 preprocess.py --folder_path <folder_path> --convert_3d_to_2d --json_path <label_json_path>
# e.g.
# python3 preprocess.py --folder_path /root/VGHTC/No_IV_197 --convert_3d_to_2d --json_path label.json
```
2. Split dataset into training set and validation set.
```=shell
$ cd data
$ python3 preprocess.py --folder_path <folder_path> --split_data
# e.g.
# python3 preprocess.py --folder_path /root/VGHTC/No_IV_197_preprocessed/ --split_data
```

### Training
1. Train the model.
```=shell
$ python3 main.py --train --yaml_path <yaml_path>
# e.g.
# python3 main.py --train --yaml_path configs/231030.yaml
```
2. Open the Tensorboard.
```=shell
$ tensorboard --logdir lightning_logs --bind_all
```

### Testing
```=shell
$ python3 main.py --test --yaml_path <yaml_path> --ckpt_path <ckpt_path>
# e.g.
# python3 main.py --test --yaml_path configs/231030.yaml --ckpt_path checkpoints/epoch\=139-step\=91420.ckpt
```

## Inference
### Download checkpoint and example
Download the checkpoint from [Drive](http://gofile.me/6Ukc0/KCdnFlIYh) and put it in "checkpoints" folder.
And download the example (000608355C) from the same drive and put it in "example" folder.
### Data structure
```
-- CTSegment
    -- example
        -- 000608355C
            -- 000608355C_CT.nii.gz
    -- checkpoints
        -- epoch=139-step=91420.ckpt
    ...
```
### Preprocess
Convert 3D .nii.gz to 2D .npy, including CT and mask.
```=shell
$ cd data
$ python3 preprocess.py --folder_path /root/VGHTC/CTSegment/example --convert_3d_to_2d_ct
```

### Inference
Prediction will save in "outputs" folder.
```=shell
$ python3 main.py \
    --predict \
    --ckpt_path <ckpt_path> \
    --yaml_path <yaml_path> \
    --patient_path <patient_path> \
    --ct_path <ct_path> \
    --saving_folder <saving_folder>
# e.g.
# python3 main.py \
    --predict \
    --ckpt_path checkpoints/epoch\=139-step\=91420.ckpt \
    --yaml_path configs/231030.yaml \
    --patient_path /root/VGHTC/CTSegment/example_preprocessed/000608355C \
    --ct_path /root/VGHTC/CTSegment/example/000608355C/000608355C_CT.nii.gz \
    --saving_folder /root/VGHTC/CTSegment/outputs
```

## Evaluation
| Class| Dice  |
|:----:| :----:|
| 131  | 0.990 |
| 181  | 1.000 |
| 212  | 0.962 |
| 231  | 1.000 |
| 241  | 0.986 |
| 242  | 0.981 |
| 243  | 0.989 |
| 244  | 0.984 |
| 245  | 0.977 |
| 246  | 0.963 |
| 254  | 0.987 |
| 261  | 0.949 |
| 301  | 0.968 |
| 311  | 0.967 |
| 341  | 0.918 |
| 342  | 0.968 |
| 361  | 0.972 |
| 362  | 0.979 |
| 363  | 0.956 |
| 364  | 0.948 |
| 365  | 0.957 |
| 366  | 0.917 |
| 367  | 0.927 |
| 411  | 0.981 |
| 471  | 0.973 |
| 491  | 1.000 |
| 512  | 1.000 |
| 513  | 0.947 |
| 514  | 1.000 |
| 515  | 1.000 |
| 516  | 1.000 |
|1131  | 0.992 |
|1132  | 0.987 |
|1133  | 0.987 |
|1134  | 0.982 |
|1135  | 0.977 |
|1136  | 0.985 |
|1137  | 0.988 |
|1141  | 0.988 |
|1142  | 0.988 |
|2111  | 0.980 |
|2112  | 0.976 |
|2131  | 0.952 |
|2132  | 0.947 |
|2511  | 0.979 |
|2512  | 0.967 |
|2521  | 0.980 |
|2522  | 0.975 |
|2531  | 0.983 |
|3711  | 0.982 |
|3712  | 0.975 |
|3821  | 0.976 |
|3822  | 0.974 |
|4121  | 0.988 |
|4122  | 0.986 |
|4211  | 0.962 |
|4212  | 0.961 |
|4221  | 0.979 |
|4222  | 0.976 |
|4231  | 0.988 |
|4232  | 0.989 |
|4241  | 0.989 |
|4242  | 0.989 |
|4411  | 0.928 |
|4412  | 0.927 |
|4421  | 0.954 |
|4422  | 0.955 |
|5111  | 0.962 |
|5112  | 0.977 |
|6111  | 0.992 |
|6112  | 0.990 |