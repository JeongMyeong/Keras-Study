## Preprocessing

### 단어 수준 원-핫 인코딩
```
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=1000)    # 가장 빈도 높은 1,000개 단어만 선택하도록 하는 토크나이저 객체 생성
tokenizer.fit_on_texts(sentences)               # tokenizer 수행
sequences = tokenizer.texts_to_sequences(sentences) # 문자열을 정수 인덱스로 변환.
word_index = tokenizer.word_index                        # 단어 인덱스 구하기
```

### 해싱 기법을 사용한 단어 수준 원-핫 인코딩
- 단어를 크기가 1,000인 벡터로 저장 하는 예 
```
dim = 1000
max_length = 10

for i, sample in enumerate(sentences):
    for j, word in list(enumerate(sentences):
        index = abs(hash(word)) % dim
        results[i, j, index] = 1.

```

### 동일한 크기로 문장 만들기
```
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = pad_sequences(sequences, maxlen=None, padding='pre', value=0.0)   # padding : 'post' or 'pre  , values: 채울 값

```


#### Ref
- https://subinium.github.io/Keras-6-1/
