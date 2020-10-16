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
source bin/activate            # 가상환경
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

