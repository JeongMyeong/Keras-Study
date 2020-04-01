# Util


## Model 저장하기
```{Python}
def model_save(MODEL, name):
    model_json = MODEL.to_json()
    with open("{}.json".format(name), "w") as json_file : 
        json_file.write(model_json)
    
    MODEL.save_weights("{}.h5".format(name))
    print("Saved model")
```
## Epoch 마다 Model 저장하기
- filepath에 epoch, val_accuracy, val_loss 등을 입력해주면 해당 epoch의 값들로 저장할 수 있다.
- save_best_only : True or False
    - True인 경우 모델 가중치만 저장되고 False인 경우 전체 모델이 저장됨.
- period : 몇 Epoch 마다 저장할 것인지.
```
# https://keras.io/ko/callbacks/

cp = ModelCheckpoint("epoch({epoch:02d})_accuracy({val_accuracy:.2f}).hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
model.fit(X_train, y_train, validation_data=[X_valid, y_valid] ,epochs=10, batch_size=64, callbacks=[cp])

```


## Model 불러오기
- 저장된 *.h5 과  *.json 파일을 사용하여 로드한다.
- !중요! 모델은 로드한 후 꼭 compile을 하여야한다.
```
from tensorflow.keras.models import model_from_json 
def model_load(model_path):
    json_file = open("{}.json".format(model_path), "r") # json 파일을 먼저 로드한 뒤
    loaded_model_json = json_file.read() 
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("{}.h5".format(model_path)) # h5파일을 로드하면서 weight값들을 올려준다.
    print("Loaded")
    return loaded_model # 로드 후 compile은 필수

```

## 모델의 요약 출력
- model에 layer들이 어떻게 연결되어있고 shape가 어떻게 변화 되는지 출력.
- layer의 이름을 설정해주지 않으면 layer 이름은 임의로 설정되어 출력된다.
```
model.summary()
```

## EarlyStopping - 학습 조기 종료
- keras model의 fit의 callbacks parameter로 줄 수 있다.
- 학습이 어느정도 되었고 성능에 진전이 없을 때 오버피팅을 방지하기 위해 모든 Epoch을 돌리지 않고 조기 종료를 시킨다.
- parameters
    - monitor : 조기 종료의 기준 데이터
    - mode : 조기 종료 기준 데이터를 최대로 할지 최소로 할지. loss가 최소가 되어야 하는건 min 그 반대는 max
    - patience : 성능이 증가하지 않는다고 곧 바로 멈추는건 효과적이지 않다. patience는 이를 막아주는 역할을 하는데 epoch을 얼만큼 더 허용하고 참을것인지 정의한다. 
    - baseline : monitor하고 있는 값이 특정 값에 도달했을 때 멈춘다.
    - restore_best_weights : 조기종료 후 가장 최선의 weights로 model을 선택
    - verbose : 출력 ON/OFF
```
from tensorflow.keras.callbacks import EarlyStopping                                                  # import
es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=10, restore_best_weights=True)  # ex1
es = EarlyStopping(monitor='val_loss',mode='min', baseline=0.6 restore_best_weights=True)             # ex2
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[es])           # 모델을 훈련시킬 때 callbacks 파라미터에 
```

## class_weight
- keras model 의 fit 의 class_weight 파라미터 값으로 설정할 수 있다.
- binary or multiclass 분류 문제에 적용할 수 있다.
- imbalanced 한 target 값에 대해 좀 더 가중치를 주고 학습시킬 수 있다.
- 다중분류보다는 이진분류에 좀 더 효과적일 것이라 생각된다.
- 정확도는 약간 상승하는 것을 볼  loss값은 떨어지지 않고 오히려 올라가는 경우가 발생할 수도 있다.
- 분류의 정확도 보다 loss를 줄이는데는 크게 효과적이지 않은것으로 생각된다.
```{python}
class_weight = {0: 40,
                1: 60
                }
model.fit(X_train, Y_train, epochs=10, batch_size=32, class_weight=class_weight)
```

 
 
 # Python Util
 
 ## list 원소 count
 ```
from collections import Counter
Counter(li).most_common(20)         # count 상위 2
 ```
## regex 정규표현식 기본 <[참고자료](http://pythonstudy.xyz/python/article/401-%EC%A0%95%EA%B7%9C-%ED%91%9C%ED%98%84%EC%8B%9D-Regex)>
```
import re
text = "안녕하세요!@#$% Hello.123456"
kor = re.sub('[^가-힣]', '', text)        # 한글만.
eng = re.sub('[^A-Za-z]', '', text)        # 영어만
num = re.sub('[^0-9]', '', text)        # 숫자만

print(kor)
print(eng)
print(num)

# 여러 regex를 조합하여 (영어,숫자)만 이런식으로 조합가능
# 원하는 글자만 추출할 때
my = re.sub('[^안하!o36]', '', text)        # 숫자만
print(my)

```
