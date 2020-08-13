from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"                              # CPU 설정
# os.environ["CUDA_VISIBLE_DEVICES"]="0"                               # GPU 설정


def f(x):
    y = x+5
    return y

X = np.array([k for k in range(-1000, 1000)])                      # define x
y = np.array([f(x) for x in X])                                                    # define y

print(X[:10])
print(y[:10])

def gen_model():
    input_ = Input(shape=(1,))
    hidden = Dense(16, activation='relu')(input_)
    output = Dense(1, activation='linear')(hidden)
    
    
    
    model = Model(input_, output)
    model.compile(loss='mae', optimizer='adam')
    
    return model



# 단순 예제로 validation dataset을 구성하지 않음.
model = gen_model()
model.fit(X, y, batch_size=128, epochs=30)

print(model.predict([2000]))