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
from tensorflow.keras.callbacks import ModelCheckpoint    
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
## Custom Activation(gelu) [gelu activation MNIST Performace](https://data-newbie.tistory.com/376)
- 해결하고자 하는 문제에 따라 성능이 다를 수 있지만 MNIST에서 elu, relu 활성화 함수보다 빠르게 수렴하는 모습을 볼 수 있다.
- 학습 횟수가 많아질수록 같은 수준으로 떨어지기는 한다.
- gelu, elu, relu 모두 사용해보고 좋은 활성화 함수를 쓰면된다.

```
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
class Gelu(Activation):
    def __init__(self, activation, **kwargs):
        super(Gelu, self).__init__(activation, **kwargs)
        self.__name__='gelu'
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': Gelu(gelu)})
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

## Using Multi GPU
- tensorflow 2.x 버전에서는 밑의 방법으로 multi gpu를 사용하면 된다.
- multi gpu로 선언되었을 때는 predict할 때 gpu 갯수 적은 만큼 갯수 만족 시켜야 한다 ? 
```
import tensorflow tf
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])    # 사용하는 gpu 번호를 지정할 수 있음.
with strategy.scope():
    # 모델 작성
    model = Model(input_data, output)
    model.compile(loss=['mae'], optimizer='adam')
```
- 밑의 방법을 사용 했을 때 2020.04.01 부터 위 방법으로 사용하라고 한다.
- [tensorflow 분산 훈련 문서](https://www.tensorflow.org/guide/distributed_training?hl=ko)
- [참고 블로그](https://hwiyong.tistory.com/96)

~- GPU 작업을 분산 시킴~
~- gpus는 2 이상~
```
from tensorflow.keras.utils import multi_gpu_model
model = Model(INPUT, OUTPUT)
parallel_model = multi_gpu_model(model, gpus=N)   # N= gpu 갯수
parallel_model.compile(loss='categorical_crossentropy', optimizer='adam')
```
~- 만약 터미널에서 ```nvidia-smi``` 를 통해 gpu 하나만 학습하고 있는것이 확인되면 밑의 두 줄을 추가하면된다.~
~- tf2.0에서는 eager mode가 default로 되어있는데, multi GPU를 위한 분산 strategy를 위해서는 disable해주어야 된다고한다.[[ref]](https://lv99.tistory.com/12)~
```
import tensorflow
tensorflow.compat.v1.disable_eager_execution()
```

## LearningRate reduce [TF](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau)
- monitor 중인 값이 patience 동안 개선이 없으면 lr * factor 로 leraning rate를 감소시킨다-
- 이는 learning rate를 줄여줌으로써 Local Minima를 빠져나올 수 있는 방법 중 하나.
```
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0, **kwargs
)
model.fit(X, y, callbacks=[reduce_lr])
```
## 훈련시 custom generator 사용
- 메모리 관리면에서 효율적이다.
- batch 만큼의 데이터를 불러오면서 전처리가 가능하다.
- 전처리 하는 과정이 오래 걸린다면 훈련하는데 속도도 더딜수도 있음.
- numpy array 연산 경우에는 전체 데이터를 전처리 한 후의 경우와 비슷한 속도를 경험하였음.
- [참고자료](https://www.soscon.net/content/data/commu/1_Day1_1600.pdf) tf keras에서 Generator를 사용하는  방법들이 정리되어 있음.
```
def generator(x_data, y_data, batch_size):
    size = len(x_data)                                  # 데이터 전체 크기
    while True:
        np.random.seed(42)                              # random seed를 고정
        idx = np.random.permutation(size)               # 입력되는 데이터를 섞음
        x_data = x_data[idx]
        y_data = y_data[idx]
        
        for i in range(size//batch_size):               # 배치사이즈 만큼 데이터를 빼온다.
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            y_batch = y_data[i*batch_size: (i+1)*batch_size]
            
            yield x_batch, y_batch
            
train_generator = generator(X, y, batch_size)

# 그냥 train_generator를 입력해주게 되면 훈련을 할 수 있음.
model.fit(train_generator, epochs=10, steps_per_epoch=len(X)//batch_size)

```
## multi-input or output 훈련시 custom generator 사용
- multi-input 혹은 multi-output 시 generator yield(return?) 부분에서 모델의 입력처럼 맞춰줘야 정상적으로 작동함 
```
def generator(x_data, y_data, batch_size):
    x_data = np.array(x_data)                           
    y_data = np.array(y_data)
    
    
    size = len(x_data)                                 # 데이터 전체 크기
    while True:
        np.random.seed(42)                              # random seed를 고정
        idx = np.random.permutation(size)               # 입력되는 데이터를 섞음
        x_data = x_data[idx]
        y_data = y_data[idx]
        for i in range(size//batch_size):               # 배치사이즈 만큼 데이터를 빼온다.
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            y_batch = y_data[i*batch_size: (i+1)*batch_size]
            yield [x_batch[0], x_batch[1]], np.array(y_batch)
```

# GPU 메모리 할당 조절 [링크](https://inpages.tistory.com/155)
```
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
```

## Optimizer

- #### optimizer poly
```
decay_st = (len(X)//batch_size+1)*epochs
poly_sche = tf.keras.optimizers.schedules.PolynomialDecay(0.001, decay_st, end_learning_rate=1e-6, power=0.9)
opt_poly = tf.keras.optimizers.Adam(poly_sche)
```
- #### optimizer cosR
```
cosine_restarts_decay_step = ((((len(X)//batch_size)+1)*epochs)//7)+1
cosine_restarts = tf.keras.experimental.CosineDecayRestarts(0.001, cosine_restarts_decay_step, t_mul=2.0, m_mul=0.9, alpha=0,name=None)
opt_cosr = tf.keras.optimizers.Adam(cosine_restarts)
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

## multiprocessing 으로 더 빠르게 연산(ref)[https://niceman.tistory.com/145]
- process의 갯수가 늘어남에 따라 빠르게 연산이 될 줄알았는데 그렇지 않음. --> 무슨 문제인지 모르겠음.
- memory를 많이 잡아먹는다 ? --> 무슨 문제인지 모르겠음.

```
from multiprocessing import Pool

# 리스트에 있는 값들을 하나씩 받아 연산을 수행하고 return 하는 함수
def run(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)
   

works = [k for k in range(10,15)]          # 함수에 넣어 수행할 값들의 리스트.
pool = Pool(processes=3) #  processes 에 적은 갯수만큼 동시에 연산됨.
result = pool.map(run, works) # run에 리스트에 있는 일을 하나씩 던져 줌. 이 때 processes에 적힌 파라미터 값의 수만큼 동시에 수행.
# result 의 type은 리스트.

```


