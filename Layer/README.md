# Layer
```
import tensorflow.keras.layers
```
## [Dense](https://keras.io/layers/core/#dense)
- dense layer는 Hidden Layer 혹은 Output Layer로 쓰인다.
#### arguments
- units - 입력데이터의 차원을 넣어주면 된다.
- activation - 활성화 함수로 [softmax, relu, sigmoid] 등이 있다. [자세히](https://keras.io/activations/)
- kernel_initializer - 초기값 설정. 기본값은 'glorot_uniform' (Xavier)로 되어있다.  activation을 'relu'로 사용시 초기값을  'he_uniform' 으로 설정하는 것도 좋은 방법이라고 한다.

|Problem| activation |  loss func|
|--                 |--         |--|
|이진분류| sigmoid | binary_crossentropy |
|다중분류|softmax|categorical_crossentropy or sparse_categorical_crossentropy|
|회귀| linear | mse|


```{python}
from tensorflow.keras.layers import Dense  		# Dense layer를 import
model.add(Dense(64, activation='relu'))          # 흔히 쓰는 Dense layer의 hidden layer
model.add(Dense(1, activation='sigmoid'))       # Dense layer의 output 이진분류
model.add(Dense(20, activation='softmax'))      # Dense layer의 output 다중 분류 
model.add(Dense(1))                             # Dense layer의 output 회귀
```

# [LSTM](https://keras.io/layers/recurrent/#lstm)
- Long Short-Term Memory layer
- RNN 의 vanishing gradient problem을 극복하기 위해 고안된 layer
- RNN의 Hidden State에 Cell State를 추가한 구조.
- 이 Cell State 덕분에 layer에 들어간 데이터가 꽤 오래 경과하더라도 앞의 내용을 잃지 않고 잘 전달이 된다.



```{python}
from tensorflow.keras.layers import LSTM        # Dense layer를 import

model.add(LSTM(128, activation='relu'))         # LSTM 사용 기본 구조
model.add(LSTM(128, activation='relu', return_sequences=True))         
# 파라미터 값중 return_sequences 를 True로 하면 각 시퀀스에서 출력을 할 수가 있다.
# return_sequences가 False이면 마지막 시퀀스에서 한 번만 출력한다.

model.add(LSTM(128, batch_input_shape=(32, 10, 100), activation='relu', stateful=True))  
# 파라미터 값중 stateful을 True로 하면 현재 샘플의 학습 상태가 다음 샘플의 초기상태로 전달 된다는것을 의미한다. 상태유지 모드에서는 batch_input_shape=(배치크기,타임 스텝, 속성)으로 설정해야한다.
```
