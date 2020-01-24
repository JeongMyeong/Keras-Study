# Util


## Model Save
```{Python}
model_json = model.to_json()
path='/'                                                                      # 경로 설정
name='keras_model'                                                     # 저장할 이름 설정
with open('{}+{}.json'.format(path, name), "w") as json_file:     # json파일 생성 및 weight 값 저장
    json_file.write(model_json)
    model.save_weights('{}+{}.json'.format(path, name))
    print('Saved')
```

## Model Load
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