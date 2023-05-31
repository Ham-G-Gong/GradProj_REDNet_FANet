## Training Sample Solution for LPCVC 2023

The sample solution is based on FANet (Hu, et al. "Real-time semantic segmentation with fast attention", IEEE RA-L, 2021).

### 0. Installation:

#### Environment:
1. Linux
2. Python 3.7 
3. Pytorch 1.8
4. NVIDIA GPU + CUDA 10.2 

#### Build

```bash
pip install -r requirements.txt
```

### 1. Prepare Data
Download and save the training/validation data [G-Drive](https://drive.google.com/file/d/1MZhohaJHxvDbcGMMDn2CPGtaH1uyxyW6/view?usp=sharing) 
(Please send an access request with your team's registriation information.) 

### 2. Modify Codes
Modify `*.yml` files in `./config`
* `data:path`: path to dataset 
* `training:batch_size`: batch_size
* `training:train_augmentations:rcrop`: input size for training

### 3. Train
Run
```bash
python train.py --config configs/*.yml
```

Taining log for the sample solution is provided in `sample_solution/runs/FA_Res18/86059/run_2023_01_18_17_48_22.log`


### 4. Validation

Modify model path `validating:resume` in  `./config/*.yml`

Run
```bash
python val.py --config configs/*.yml
```


