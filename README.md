# Keras-Study
케라스 라이브러리에서 자주쓰는 내용 정리


# Paper
- [Self-training with Noisy Student improves ImageNet classification Review](https://hoya012.github.io/blog/Self-training-with-Noisy-Student-improves-ImageNet-classification-Review/?fbclid=IwAR2Z3v3aBDS1Zc-UEG2YCdmrdlqJG3qn4_qubVoLYvJPjXNYZKsLklXTA1s)

# Layer
- Dense Layer
- LSTM Layer
# Text
- 단어 수준 원-핫 인코딩
- 해싱 기법을 사용한 단어 수준 원-핫 인코딩
- 동일한 길이로 문장 처리
# Util
- Model 저장하기
- Epoch 마다 Model 저장하기
- Model 불러오기
- 모델의 요약
- EarlyStopping(조기종료)
- Custom Activation(Gelu)
- model fit class_weight
- Using Multi GPU
- Python Util
  - list 원소 count
  - regex 정규표현식
  - multiprocessing 으로 더 빠르게 연산
- optimizer
  - poly
  - cosine restarts


# Ubuntu Command
#### ubuntu python virtual env 생성
```
# env_name : 원하는 환경 이름
python -m virtualenv <env_name>
cd env_name
source bin/activate            # 가상환경 활성화 
```

#### conda virtual env 생성
```
conda create -n py37_tf python=3.7     # python 3.7 버전을 py37_tf 이름으로 생성한다. python3.7버전이 설치됨.
conda info --envs                      # 설치된 anaconda 가상 환경 리스트를 나열해줌
source activate py37_tf                # py37_tf 가상 환경을 활성화          or (conda activate py37_tf)
source deactivate                      # 기본 환경으로 되돌아가기
```
#### jupyter notebook에 가상환경의 kernel 추가/삭제
- 가상환경 추가
```
# env-name : 추가할 가상환경의 이름
# jupyter-display-name : 주피터에서 보여질 이름
python -m ipykernel install --user --name <env-name> --display-name <jupyter-display-name>
```

- 가상환경 삭제
```
# jupyter-display-name : 주피터에서 보여지는 이름
jupyter kernelspec uninstall <jupyter-display-name>
```
#### alias
```
alias file_cnt='ls -l|grep ^-|wc -l'
source /.bashrc
```
#### 우분투 전체 계정에 적용
```
vi /etc/profile      # 

```

### 우분투 딥러닝 환경 설정
- 인터넷 연결
- ssh 연결
- nvidia 설치 [참고 블로그](https://seonghyuk.tistory.com/m/35?category=755641)
```
sudo add-apt-repository ppa:graphics-drivers/ppa         # ppa저장소 추가
sudo apt update 
apt-cache search nvidia | grep nvidia-driver-450         # 설치 가능한 드라이버 리스트 출력
sudo apt-get install nvidia-driver-450                   # 설치
sudo reboot 
```
- cuda 설치 ( https://developer.nvidia.com/cuda-toolkit-aRCHIVE )
  - 사용하려는 라이브러리의 환경에 따라 버전을 보고 설치
  - 되도록이면 파일을 다운받아 설치하는 것이 nvidia 설치한것이랑 겹치지 않음.
  - 파일 다운 받아 설치시 nvidia는 이미 설치했으므로 체크 해제가 필요(★)
  - cuda path 설정
```
vi .bashrc
export PATH=$PATH:/usr/local/cuda-xx.xx/bin                                # xx 는 버전에 맞게 적절히 수정
export CUDADIR=/usr/local/cuda-xx.xx
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-xx.xx/lib64
source ~/.bashrc
sudo reboot
nvcc -V
```
- cudnn 설치 ( https://developer.nvidia.com/rdp/cudnn-archive )
  - cuda 버전에 맞는 cudnn을 설치
  - 딥러닝을 할 때는 Runtime Libary만 설치해주면 됨.
  




# Python parser example
```{python}
import argparse

parser = argparse.ArgumentParser(description='Argparse test')
parser.add_argument('--epoch', type=int,
                help='an integer for printing repeatable'
                    )

args = parser.parse_args()

for i in range(args.epoch):
    print('print number {}'.format(i+1))
    
## Terminal run
python3 parser_test.py -h
python3 parser_test.py --epoch 5
```


# cpu multiprocessing

```
paths=[]
import multiprocessing as mp
from joblib import Parallel, delayed
import cv2
def img_load(path):
    img = cv2.imread(path)
    return img
imgs = Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(img_load)(path) for path in tqdm(paths))

```
