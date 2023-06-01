# GradProj_REDNet_FANet

> writer sangho

- UAV 온디바이스 재난 영역 검출 소프트웨어 개발

- segmentation AI 모델
    - FANet [paper](https://ieeexplore.ieee.org/document/9265219/authors#authors)

    - REDNet [paper](https://ieeexplore.ieee.org/abstract/document/9377916)

## Model
### FANet
![FANet structure](./figure/FANet_arch.png)


### REDNet
![REDNet structure](./figure/REDNet_arch.png)

### env seting and Dependencies
1. [ANACONDA](https://www.anaconda.com/) 를 설치해 주세요.
2. yaml 파일을 통해 가상환경을 세팅합니다. 'conda env create --file environment.yaml'

- 개발 환경 및 라이브러리는 ![rednet.yaml](rednet.yaml) 을 통해 확인 할 수 있습니다. 

### Train
1. config 파일을 셋팅해주세요. (default : RED_Res18.yml)
2. train.py 에서 run_id 변수를 설정해주세요. 변수명은 날짜입니다.
3. terminal 에서 'python train.py --config configs/yml 파일 이름.yml'

runs/yml 파일 이름/run_id/ 경로에 matric 이 가장 높았던 best_model.pth 와 학습이 끝난 후 last_model.pth 가 저장됩니다.

run_시간.log 로 학습 log 를 확인 할 수 있습니다.

## Dataset
- ![LPCV](https://lpcv.ai/2023LPCVC/introduction) 에서 데이터 셋을 다운로드 해주세요.
- 1021 장의 training set
- 100 장의 validation set
- 재난 이미지: 512x512x3
- GroundTruth 이미지: 512x512x3

