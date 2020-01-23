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