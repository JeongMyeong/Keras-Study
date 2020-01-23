# Util


## Model Save
```
model_json = model.to_json()
path='/'
name='keras_model'
with open('{}+{}.json'.format(path, name), "w") as json_file:
    json_file.write(model_json)
    model.save_weights('{}+{}.json'.format(path, name))
    print('Saved')
```