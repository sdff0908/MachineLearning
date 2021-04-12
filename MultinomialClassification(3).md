# Multinomial Classification(Fashion MNIST)

## 1. Single-Layered Perceptron(tf.v1)

### 1) Data Processing

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
```

```python
df = pd.read_csv('./data/fashion/fashion-mnist_train.csv')
img_data = df.drop('label', axis=1, inplace=False).values

# 결측치 확인
df.isnull().sum()

# 이미지 확인
fig = plt.figure()
fig_ls = list()

for i in range(10):
    fig_ls.append(fig.add_subplot(2,5,i+1))
    fig_ls[i].imshow(img_data[i].reshape(28,28), cmap='gray')
plt.tight_layout()
plt.show()

# data split
x_tr,x_tst,y_tr,y_tst = train_test_split(img_data, df['label'], 
                                         test_size=0.3, random_state=3)

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# one-hot encoding
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr, depth=10))
y_tst = sess.run(tf.one_hot(y_tst, depth=10))
```

### 2) Model

```python
# placeholder
X = tf.placeholder(shape=[None,784],dtype=tf.float32)
Y = tf.placeholder(shape=[None,10],dtype=tf.float32)

# weight & bias
W = tf.get_variable('W',shape=[784,10], initializer=tf.contrib.layers.variance_scaling_initializer())
b = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
sess.run(tf.global_variables_initializer())

# mini batch gradient
epoch = 1000
batch_size = 128

def run_train(sess, x, y):
    print('start')
    total_batch = int(x.shape[0]/batch_size)
    for step in range(epoch):
        for i in range(total_batch):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([optimizer,loss], feed_dict={X:batch_x,Y:batch_y})
    print('loss:', loss_val)
    print('end')
```

```python
# train
k = 5
results = []
kf = KFold(n_splits=k,shuffle=True)
for tr_idx, cv_idx in kf.split(x_tr):
    tr_x_data = x_tr[tr_idx]
    tr_y_data = y_tr[tr_idx]
    cv_x_data = x_tr[cv_idx]
    cv_y_data = y_tr[cv_idx]
    
    run_train(sess,tr_x_data,tr_y_data)
```

### 3) Evaluation

```python
# accuracy
predicted_y = sess.run(tf.argmax(H, axis=1), feed_dict={X:x_tst})
true_y = sess.run(tf.argmax(Y, axis=1), feed_dict={Y:y_tst})
accuracy = classification_report(true_y, predicted_y)
print(accuracy)
```

```python
              precision    recall  f1-score   support

				...

    accuracy                           0.87     60000
   macro avg       0.87      0.87      0.87     60000
weighted avg       0.87      0.87      0.87     60000
```

## 2. DNN(tf.v1)

### 1) Data Processing

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

```python
df = pd.read_csv('./data/fashion/fashion-mnist_train.csv')
img_data = df.drop('label', axis=1, inplace=False).values

# data split
x_tr,x_tst,y_tr,y_tst = train_test_split(img_data, df['label'], 
                                         test_size=0.3, random_state=3)

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# one-hot encoding
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr, depth=10))
y_tst = sess.run(tf.one_hot(y_tst, depth=10))
```

### 2) Model

```python
# placeholder
X = tf.placeholder(shape=[None,784],dtype=tf.float32)
Y = tf.placeholder(shape=[None,10],dtype=tf.float32)

# weight & bias
W1 = tf.get_variable('W1',shape=[784,256], initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([256]))
layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.get_variable('W2',shape=[256,128], initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([128]))
layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)

W3 = tf.get_variable('W3',shape=[128,32], initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([64]))
layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)

W4 = tf.get_variable('W4',shape=[32,10], initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
sess.run(tf.global_variables_initializer())

# mini batch gradient
epoch = 1000
batch_size = 128

def run_train(sess, x, y):
    print('start')
    total_batch = int(x.shape[0]/batch_size)
    for step in range(epoch):
        for i in range(total_batch):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([optimizer,loss], feed_dict={X:batch_x,Y:batch_y})
        print('loss:', loss_val)
        print('end')
```

```python
# train
run_train(sess, x_tr, y_tr)
```

### 3) Evaluation

```python
# accuracy
predicted_y = sess.run(tf.argmax(H, axis=1), feed_dict={X:x_tst})
true_y = sess.run(tf.argmax(Y, axis=1), feed_dict={Y:y_tst})
accuracy = classification_report(true_y,predicted_y)
print(accuracy)
```

```python
              precision    recall  f1-score   support

				...

    accuracy                           0.89     18000
   macro avg       0.89      0.89      0.89     18000
weighted avg       0.89      0.89      0.89     18000
```

## 3. CNN(tf.v1)

### 1) Data Processing

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

```python
df = pd.read_csv('./data/fashion/fashion-mnist_train.csv')
img_data = df.drop('label', axis=1, inplace=False).values

# data split
x_tr,x_tst,y_tr,y_tst = train_test_split(img_data, df['label'], 
                                         test_size=0.3, random_state=3)

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# one-hot encoding
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr, depth=10))
y_tst = sess.run(tf.one_hot(y_tst, depth=10))
```

### 2) Model

1. Convolution

```python
# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)

# convolution 입력 데이터 설정: 2차원 --> 4차원
x_img = tf.reshape(X, [-1,28,28,1])

# filter 1
W1 = tf.Variable(tf.random.normal([3,3,1,32]))
C1 = tf.nn.conv2d(x_img,W1,strides=[1,1,1,1], padding='SAME')
R1 = tf.nn.relu(C1) # activation map

# pooling 1
P1 = tf.nn.max_pool(R1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') 

# filter 2
W2 = tf.Variable(tf.random.normal([3,3,32,64])) 
C2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
R2 = tf.nn.relu(C2) # activation map

# pooling 2
P2 = tf.nn.max_pool(R2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
```

* x_img.shape : (-1, 28, 28, 1) = (이미지 개수, 이미지 높이, 이미지 너비, 이미지 채널 수)

* W1.shape : (3, 3, 1, 32) = (필터 높이, 필터 너비, 필터 채널 수, 필터 개수)
                       필터 채널 수 = 이미지 채널 수
                       필터 크기는 일반적으로 3*3사용
* pooling 통해 이미지 크기 감소 
  : (?, 28, 28, 1) → W1 → (?, 28, 28, 32) → pooling → (?, 14, 14, 32) → W2 → (?, 14, 14, 64) → pooling → (?, 7, 7, 64)

* pooling : 일반적으로 ksize = stride

2. DNN

```python
# DNN 입력데이터 설정 : 4차원 --> 2차원
P2 = tf.reshape(P2, [-1,7*7*64])

# weight & bias
W3 = tf.get_variable('W3', shape=[7*7*64,256], 
                     initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([256]))
layer3 = tf.nn.relu(tf.matmul(P2,W3) + b3)

W4 = tf.get_variable('W4', shape=[256,10], 
                     initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(layer3,W4) + b4
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# mini batch gradient
epoch = 1000
batch_size = 128

def run_train(sess, x, y):
    print('start')
    sess.run(tf.global_variables_initializer())
    total_batch = int(x.shape[0]/batch_size)
    for step in range(epoch):
        for i in range(total_batch):
            batch_x = x[i*batch_size:(i+1)*batch_size]
            batch_y = y[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([optimizer, loss], feed_dict={X:batch_x, Y:batch_y})
        print('loss:', loss_val)
        print('end') 
```

* P2.shape : (이미지 개수, feature 개수)
* initialization : Xavier initialization, He initialization은 weight에만 적용
                           bias는 더해주는 역할이므로 initialization 영향 받지 X
* GradientDescentOptimizer, SGD, Adam, ... : optimizer는 back propagation기능 포함

```python
# train
run_train(sess, x_tr, y_tr)
```

### 3) Evaluation

```python
# accuracy
predicted_y = sess.run(tf.argmax(H, axis=1), feed_dict={X:x_tst})
true_y = sess.run(tf.argmax(Y, axis=1), feed_dict={Y:y_tst})
accuracy = classification_report(true_y, predicted_y)
print(accuracy)
```

```python
              precision    recall  f1-score   support

				...

    accuracy                           0.91     18000
   macro avg       0.91      0.91      0.91     18000
weighted avg       0.91      0.91      0.91     18000
```

## 4. CNN(tf.v2)

### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

```python
df = pd.read_csv('./data/fashion/fashion-mnist_train.csv')
img_data = df.drop('label', axis=1, inplace=False).values

# data split
x_tr,x_tst,y_tr,y_tst = train_test_split(img_data, df['label'], 
                                         test_size=0.3, random_state=3)

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr).reshape(-1,28,28,1)
x_tst = scaler.transform(x_tst).reshape(-1,28,28,1)
```

x_tr, x_tst : convolution는 4차원 데이터 사용하므로 데이터 형태를 2차원에서 4차원으로 바꿔준다

### 2) Model

```python
# 모델 생성
model = Sequential()

# convolution layer 1
model.add(Conv2D(filters=32, kernel_size=(3,3), 
                 padding='SAME', activation='relu', input_shape=(28,28,1)))

# pooling layer 1
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='SAME'))

# convolution layer 2
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME', activation='relu'))

# pooling layer 2
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

# DNN
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10, activation='softmax'))

print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model.fit(x_tr, y_tr, batch_size=128, 
                    epochs=100, verbose=0, validation_split=0.3)
```

* Conv2D() : convolution함수
                     (filters = 필터개수, input_shape = 입력 데이터 1개의 형태)
                     input_shape 지정하여 input layer역할까지 포함
                     stride = 1(default)이 좋음, stride 값 크면 특징 잘 추출 X 

* MaxPooling2D() : max pooling 함수, pool_size(tf.v2) = ksize(tf.v1)
* Dense(units=노드개수)
* model.summary() : 모델 구조 파악하기

* loss : one-hot encoding한 y 값은 categorical_crossentropy, one-hot encoding 하지 않은 y값은 sparse_categorical_crossentropy 사용

```python
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 14, 64)        18496     
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_10 (Dense)             (None, 256)               803072    
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                2570      
=================================================================
Total params: 824,458
Trainable params: 824,458
Non-trainable params: 0
_________________________________________________________________
None
```

param # : weight 개수

### 3) Evaluation

```python
563/563 [==============================] - 6s 11ms/step - loss: 0.2588 - sparse_categorical_accuracy: 0.9056
[0.25877121090888977, 0.9055555462837219]
```
