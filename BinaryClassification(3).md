# Binary Classification(cats vs dogs)

## 1. csv파일 사용

img  → csv



## 2. 이미지 파일 사용

### 1) Data Processing

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```python
train_dir = '/content/cat_dog_small/train'
validation_dir = '/content/cat_dog_small/validation'

train_gen = ImageDataGenerator(rescale=1/255)
validation_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_gen.flow_from_directory(train_dir, 
                                                classes=['cats', 'dogs'], 
                                                target_size=(150,150), 
                                                batch_size=20, 
                                                class_mode='binary')
validation_generator = validation_gen.flow_from_directory(validation_dir, 
                                                          classes=['cats', 'dogs'], 
                                                          target_size=(150,150), 
                                                          batch_size=20, 
                                                          class_mode='binary')
```

* ImageDataGenerator() 

  이미지 어떻게 변화시킬 지 지정
  rescale : 각 이미지 픽셀에 rescale 값 곱하기 →  0~1 사이값으로 정규화

* flow_from_directory() 

  특정 폴더에서 이미지 데이터 가져오기
  해당 폴더 안에 labeling 된 하위폴더 있어야 사용 가능한 함수
  여기서는 target directory인 train_dir 안에 cats와 dogs라는 이름의 하위 폴더 두 개 존재
  train_dir : target directory 경로
  classes : 리스트 안에 적힌 순서로 0,1,2, ... label 설정
            classes 속성 생략하면 폴더 순서로 label 설정
  target_size :  이미지 resize, resize 값은 CNN 구조에 따라 다름
  batch_size : 한 번에 가져올 이미지 파일 개수, label값에 상관없이 가져옴
  class_mode : binary classification → 'binary'
  multinomial classification → 'categorical'

```python
for x_data, y_data in train_generator:
    print(x_data.shape) 
    print(y_data.shape)  
    break;
```

```python
(20,150,150,3)
(20,)
```

ImageDataGenerator는 무한히 작동
지정된 batch_size가 전체 데이터를 돌고 난 뒤에도 계속 작동 →  *break 필요*

### 2) Model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

```python
model = Sequential()

# convolution layers
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# FC layers
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

# 모델 구조 확인
print(model.summary())

# 모델 설정 및 학습
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=50, 
                    validation_data=validation_generator, validation_steps=50)
```

* input_shape : batch를 한 번 돌 때 train_generator의 x_data는 (150,150,3) 가 20개 있음
                           input_shape에는 1개 데이터 형태 입력
* epoch : 전체 데이터 반복 학습 횟수
* steps_per_epoch : (batch_size) × (steps_per_epoch) =  전체 학습 데이터 개수 
* validation_steps : (validation_batch_size) × (validation_steps) = 전체 validation 데이터 개수

## 3. 이미지 파일 사용: Data Augmentation

> 데이터양이 부족하면 overfitting 발생, data augmentation 통해 overfitting 부분적으로 해결 가능

### 1) Example

```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
```

```python
gen = ImageDataGenerator(rotation_range=90,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.2,  
                         horizontal_flip=True,
                         vertical_flip=True)  

img = image.load_img('./data/iu.jpg', target_size=(300,300))
data = image.img_to_array(img)
print(data.shape)                            # (300, 300, 3)
data = data.reshape((1,) + data.shape)
print(data.shape)                            # (1, 300, 300, 3)

fig = plt.figure()
fig_ls = list()
for i in range(12):
    fig_ls.append(fig.add_subplot(3,4,i+1))

idx = 0
for batch in gen.flow(data): 
    fig_ls[idx].imshow(image.array_to_img(batch[0])) 
    idx += 1
    if idx == 12:
        break
plt.tight_layout()
plt.show()
```

* ImageGenerator()

  rotation_range : 회전 범위, 여기서는 0~90도 범위에서 임의로 회전
  width_shift_range : 가로 이동 범위, 여기서는 1-0.2 ~ 1+0.2 범위에서 이동
  height_shift_range : 세로 이동 범위, 여기서는 1-0.2 ~ 1+0.2 범위에서 이동
  zoom_range: 확대 및 축소 범위, 여기서는 1-0.2 ~ 1+0.2 범위에서 확대 및 축소
  horizontal_flip = True : 좌우 반전
  vertical_flip = True : 상하 반전

* flow(): 4차원 np.array 들어가야 함
* array_to_img() : 3차원 np.array →  이미지
                              gen.flow(data) 안에 있는 np.array는 4차원, batch[0] 사용하여 3차원으로 만듦

![다운로드](https://user-images.githubusercontent.com/72610879/120614580-59745e80-c492-11eb-911d-5e203a104979.png)


이미지는 ImageGenerator() 에서 설정한 값들을 종합한 결과

### 2) Data Processing

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```python
train_dir = '/content/cat_dog_small/train'
validation_dir = '/content/cat_dog_small/validation'

train_gen = ImageDataGenerator(rescale=1/255, 
                               rotation_range=90, 
                               width_shift_range=0.2, 
                               height_shift_range=0.2, 
                               zoom_range=0.2, 
                               horizontal_flip=True, 
                               vertical_flip=True)
validation_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_gen.flow_from_directory(train_dir, 
                                                classes=['cats', 'dogs'], 
                                                target_size=(150,150), 
                                                batch_size=20, 
                                                class_mode='binary')
validation_generator = validation_gen.flow_from_directory(validation_dir, 
                                                          classes=['cats', 'dogs'], 
                                                          target_size=(150,150), 
                                                          batch_size=20, 
                                                          class_mode='binary')
```

### 3) Model

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

```python
model = Sequential()

# convolution layers
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# FC layers
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

# 모델 구조 확인
print(model.summary())

# 모델 설정 및 학습
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', 
              metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=100, 
                    validation_data=validation_generator, validation_steps=50)
```

data augmentation 사용할 경우 epochs 수를 좀 더 늘려주는 것이 좋음
조금씩 변형된 데이터가 계속 들어가니까 학습에 이득

## 4. 이미지 파일 사용: Transfer Learning

> 이미 학습된 네트워크 사용하는 방법

Pretrained network

* VGG16 
* VGG19
* ResNet
* Inception
* EfficientNet : 가장 효율적
* MobileNet
*  ...

### 1) CV layers - FC layers 분리

pretrained network 사용하여 특징 추출 후  FC layer 반복 학습, 학습 속도 빠름

1. Pretrained Network

```python
from tensorflow.keras.applications import VGG16

model_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
model_base.summary()
```

```python
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
...
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
```

* weights = 'imagenet' : imagenet에 있는 데이터로 학습한 모델 사용
* include_top = False : convolution layers 만 사용, FC layers 사용 X
* input_shape = (150,150,3) : 입력할 데이터 각각의 형태
                                                   model_base.summary()의 input_1 (InputLayer)에서 확인 가능

```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

```python
base_dir = '/content/data/cat_dog_img'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1/255)
batch_size = 20

def extract_feature(directory, sample_count):
  features = np.zeros(shape=(sample_count,4,4,512))
  labels = np.zeros(shape=(sample_count,)) 

  generator = datagen.flow_from_directory(directory, 
                                          target_size=(150,150),
                                          batch_size=batch_size, 
                                          class_mode='binary')
  # target_size : 가져온 이미지는 model_base로 들어감 → model_base(input_shape) 참고
  
  i = 0
  for x_batch, y_batch in generator:
    print(i*batch_size)
    features[i*batch_size:(i+1)*batch_size] = model_base.predict(x_batch)
    labels[i*batch_size:(i+1)*batch_size] = y_batch

    i += 1
    if i*(batch_size) >= sample_count:
      break               
  return features, labels

train_features, train_labels = extract_feature(train_dir, 14000)
validation_features, validation_labels = extract_feature(validation_dir, 6000)
```

* train_dir : '/content/data/cat_dog_img/train'

* validation_dir : '/content/data/cat_dog_img/validation'

* def extract_feature(directory, sample_count)

  directory : 이미지를 가져올 폴더
  sample_count : 가져올 이미지 개수 

* features, labels : 함수가 반환할 값
                                 np.zeros()사용하여 학습 결과를 담을 틀 생성
                                 (sample_count,4,4,512) 은  model_base.의 block5_pool (MaxPooling2D)에서 결정

* flow_from_directory : 지정 폴더 아래에 labeling된 하위폴더 있어야 함
                                         target_size 사용하여 하위 폴더 안에 있는 각각의 이미지 크기 지정

*  ImageDataGenerator는 무한 반복 →  전체 데이터 돌고나면 break필요

2. FC layers(DNN)

```python
train_features = np.reshape(train_features, (14000,4*4*512))
validation_features = np.reshape(validation_features, (6000,4*4*512))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(4*4*512,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(learning_rate=1e-2), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_features, 
                    train_labels, 
                    epochs=30, 
                    batch_size=20, 
                    validation_data=(validation_features, validation_labels))
```

* (sample_count, 4 *4 * 512) : FC layers에는 2차원 데이터 들어가므로 데이터를 2차원으로 변경, 여기서는 pretrained network에서 학습된 데이터가 (4, 4, 512) 형태로 출력 →  4 * 4 * 512

### 2) CV layers - FC layers 결합

pretrained network를 FC layer와 결합한 모델 전체를 반복 학습
학습 데이터가 정해져 있는 경우 이 방식을 쓸 이유 없음, pretrained network와 FC layer 분리하여 학습하는 방법이 더 빠름
데이터 증식(Data Augmentation)이나 Fine Tuning 사용할 때 CV layers, FC layers 전체를 반복 학습

#### (1) Data Augmentation

1. Data Augmentation & Data Processing

```python
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
```

```python
base_dir = '/content/drive/MyDrive/Colab Notebooks/data/cat_dog_img'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_gen = ImageDataGenerator(rescale=1/255,
                               rotation_range=90,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True)
validation_gen = ImageDataGenerator(rescale=1/255)
train_generator = train_gen.flow_from_directory(train_dir, 
                                                classes=['cats', 'dogs'],  
                                                target_size=(150,150), 
                                                batch_size=20, 
                                                class_mode='binary')
validation_generator = validation_gen.flow_from_directory(validation_dir, 
                                                          classes=['cats', 'dogs'],
                                                          target_size=(150,150),
                                                          batch_size=20, 
                                                          class_mode='binary')
```

2. Model

```python
model_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
model_base.trainable = False
model = Sequential()
model.add(model_base)
model.add(Flatten(input_shape=(4*4*512,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=1e-2), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
history = model.fit(train_generator, 
                    steps_per_epoch=100, 
                    epochs=30, 
                    validation_data=validation_generator, 
                    validation_steps=50)
```

* include_top = False : imagenet의 convolution layer 만 model_base로 사용
* model_base : 4-2) CV layers - FC layers 분리 1. Pretrained Network 그대로 사용 
                           →  FC layers의 input_shape=(4 * 4 * 512,)

* model.add(model_base) : model_base가 model 안에 포함되어 epoch 수 만큼 전체 모델이 학습

* model_base.trainable = False : 

  전체 모델이 학습하지만 pretrained network(=model_base)는 학습 X 	
  pretrained network결과인 feature 추출하는 필터는 고정
  epoch 돌 때마다 필터가 새롭게 증식된 데이터 받아들이므로 필터를 통과하여 Flatten layer에 입력되는 데이터는 계속 바뀜

#### (2) Fine Tuning

1. Data Augmentation & Data Processing

```python
import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
```

```python
base_dir = '/content/drive/MyDrive/Colab Notebooks/data/cat_dog_img'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_gen = ImageDataGenerator(rescale=1/255,
                               rotation_range=90, 
                               width_shift_range=0.1, 
                               height_shift_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               vertical_flip=True)
validation_gen = ImageDataGenerator(rescale=1/255)
train_generator = train_gen.flow_from_directory(train_dir,
                                                classes=['cats', 'dogs'], 
                                                target_size=(150,150), 
                                                batch_size=20, 
                                                class_mode='binary')
validation_generator = validation_gen.flow_from_directory(validation_dir, 
                                                          classes=['cats', 'dogs'], 
                                                          target_size=(150,150), 
                                                          batch_size=20, 
                                                          class_mode='binary')
```

2. Model

```python
model_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
model_base.trainable = False
model = Sequential()
model.add(model_base)
model.add(Flatten(input_shape=(4*4*512,)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer=RMSprop(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)

model_base.trainable = True
for layer in model_base.layers:
  if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
    layer.trainable = True
  else:
    layer.trainable = False
model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)
```

* Pretrained Network 고정한 상태에서 FC layers 훈련 → 가지고 있는 데이터 사용하여 Pretrained Network 일부를 훈련 → FC layers 훈련
  
* model_base : 4-2) CV layers - FC layers 분리 1. Pretrained Network 그대로 사용 
                           →  FC layers의 input_shape=(4 * 4 * 512,)

* model.add(model_base) : model_base가 model 안에 포함되어 epoch 수 만큼 전체 모델이 학습

* model_base.trainable = True :

  epoch이 돌 때 pretrained network(=model_base)도 학습
  ['block5_conv1', 'block5_conv2', 'block5_conv3']은 학습, 그 외 layer는 학습 X
  ['block5_conv1', 'block5_conv2', 'block5_conv3']는 model_base 구조의 마지막 layer 중 일부
  pretrained network 사용하되 끝부분 layer는 현재 데이터 학습함으로써 현재 데이터에 보다 적합한 feature extraction filter 생성 가능

* learning_rate : FC layers를 두번째 학습할 때는 learning_rate 크기를 더 작게 한다(미세 조정 가능하도록)

## 5. TF Recrod 사용

