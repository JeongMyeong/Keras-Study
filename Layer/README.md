# Layer
```
import tensorflow.keras.layers
```
## [Dense](https://keras.io/layers/core/#dense)
- dense layer는 Hidden Layer 혹은 Output Layer로 쓰인다.
#### arguments
- units - 입력데이터의 차원을 넣어주면 된다.
- activation - 활성화 함수로 [softmax, relu, sigmoid] 등이 있다. [자세히](https://keras.io/activations/)

|Problem| activation |  loss func|
|--                 |--         |--|
|이진분류| sigmoid | binary_crossentropy |
|다중분류|softmax|aa|
|회귀| linear | mse|



```{python}
from tensorflow.keras.layers import Dense  		# Dense layer를 import
model.add(Dense(64, activation='relu'))          # 흔히 쓰는 Dense layer의 hidden layer
model.add(Dense(1, activation='sigmoid'))       # Dense layer의 output 이진분류
model.add(Dense(20, activation='softmax'))      # Dense layer의 output 다중 분류 
model.add(Dense(1))                             # Dense layer의 output 회귀
```

