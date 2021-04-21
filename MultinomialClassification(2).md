## Multinomial Regression(MNIST)

## 1. ANN

> Artificial Neural Network(인공신경망), 모든 머신러닝 모델 포괄하는 개념

아래 예시의 예시 1,2,3 특징

* 계산과정 : feed-forward 
* Input layer, output layer로 구성, hidden layer X

* FC(Fully-Connected) network : 모든 노드가 서로 연결되어 있는 network = FC layer로 구성된 network

  FC(Fully-Connected) layer : 특정 레이어의 모든 노드가 이전 레이어와 다음 레이어의 모든 노드와 연결되어 있는 layer

### 1. tensorflow.v1

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import cv2
```

```python
# raw data
df = pd.read_csv('./data/mnist/train.csv')

# data split
x_data = df.drop('label', axis=1)
y_data = df['label']
x_tr,x_tst,y_tr,y_tst = train_test_split(x_data,y_data,test_size=0.3,random_state=11)

# feature scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# onehot encoding
label_n = len(set(y_tr))
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr,depth=label_n))
y_tst = sess.run(tf.one_hot(y_tst,depth=label_n))
```

* one-hot encoding

  depth : y 값 종류 갯수

#### 2) Train & Accuracy

```python
# mini batch gradient
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)

W = tf.Variable(tf.random.normal([784,10]))
b = tf.Variable(tf.random.normal([10]))

logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

num_epoch = 1000
batch_size = 100

def run_train(sess, train_x, train_y):
    print('start')
    sess.run(tf.global_variables_initializer())
    total_batch = int(train_x.shape[0] / batch_size)
    for step in range(num_epoch):
        for i in range(total_batch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([train,loss], feed_dict={X:batch_x,Y:batch_y})
    print('loss:',loss_val)
    print('end')
    
# accuracy
predicted_y = tf.argmax(H, axis=1)
true_y = tf.argmax(Y, axis=1)
```

* loss함수 : multinomial은 tf.nn.softmax_cross_entropy_with_logits_v2 사용

```python
# train
k = 5
results = []
kf = KFold(n_splits=k, shuffle=True)
for tr_idx, cv_idx in kf.split(x_tr):
    tr_x_data = x_tr[tr_idx]
    tr_y_data = y_tr[tr_idx]
    cv_x_data = x_tr[cv_idx]
    cv_y_data = y_tr[cv_idx] 
    
    run_train(sess, tr_x_data, tr_y_data)
    results.append(accuracy_score(sess.run(true_y, feed_dict={Y:cv_y_data}), 
                                  sess.run(predicted_y, feed_dict={X:cv_x_data})))
    
print('results:',results)
print('k-fold accuracy:',np.mean(results))
```

k-fold crossvalidation 사용하여 모델 학습

```python
start
loss: 0.1835725
end
...
start
loss: 0.18064842
end

results: [0.9056122448979592, 0.9057823129251701, 0.9078231292517007, 0.9113945578231293, 0.9096938775510204]
k-fold accuracy: 0.908061224489796
```

```python
# evaluation
final_accuracy = accuracy_score(sess.run(true_y, feed_dict={Y:y_tst}), sess.run(predicted_y, feed_dict={X:x_tst}))
print(final_accuracy)
```

```
0.9083333333333333
```

### 2. tensorflow.v2

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
```

```python
# Raw Data
df = pd.read_csv('./data/mnist/train.csv')
x_data = df.drop('label', axis=1, inplace=False)
y_data = df['label']

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

# Data Split
x_tr, x_tst, y_tr, y_tst = train_test_split(x_data, y_data, test_size=0.3, random_state=3)
```

tensorflow 1.x 사용할 경우 반드시 one-hot encoding필요

tensorflow 2.x 사용할 경우 one-hot encoding하지 않아도 설정을 통해 one-hot encoding 처리 가능

tensorflow 2.x에서 설정 사용하지 않고 1.x에서와 같이 직접 one-hot encoding 가능

#### 2) Train & Accuracy

hidden layer 없음

```python
# 모델 생성
model = Sequential()

# layer 생성
model.add(Flatten(input_shape=(x_tr.shape[1],)))
model.add(Dense(len(set(y_data)), activation='softmax'))

# 모델 설정
model.compile(optimizer=SGD(learning_rate=1e-1), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 모델 학습
history = model.fit(x_tr,y_tr,epochs=100,batch_size=512,verbose=1,validation_split=0.2)
# 처음에 epochs 값 임의로 설정(plot 부분 참고)

# 평가
evaluation = model.evaluate(x_tst,y_tst)
print(history)
print(evaluation)
```

* Flatten()

  입력 데이터를 한 줄로 펼치는 역할(예: 입력 데이터가 2차원이면 Flatten으로 1차원 데이터가 된다)
  FC layer는 1차원 데이터만 처리가능하기에 Flatten() 사용
  input_shape = 입력 데이터 column 수

* Dense()

  FC layer에 사용
  
  len(set(y_data)) : Dense layer 가 가진 노드 갯수, logistic 하나 당 Dense 노드 1개binary classification은 노드 1개, multinomial classification은 노드 n(=y값 종류 갯수)개
  
  *Q: Dense layer의 노드 갯수가 2개일 때와 1개일 때의 차이는??*
  
  *노드 갯수가 1개인 binary classification =  노드 갯수가 2개인 multinomial classification*
  
  activation : motinomial classification은 softmax 함수 사용

* sparse_categorical_crossentropy : one-hot encoding 없이 crossentropy사용

  sparse_가 붙어 있지 않은 loss함수(categorical_crossentropy) 사용할 경우 one-hot encoding 별도 처리 필요

* metrics : cross validation에 사용할 metrics

* history = model.fit() : 모델이 학습한 내용을 history라는 객체에 저장

* validation_split = 0.2 : training data set(x_tr, y_tr)에서 cross-validation data로 사용할 비율

```python
# history
Epoch 1/100 
46/46 [==============================] - 0s 6ms/step - loss: 1.2805 - sparse_categorical_accuracy: 0.6954 - val_loss: 0.8518 - val_sparse_categorical_accuracy: 0.8209
...
Epoch 100/100
46/46 [==============================] - 0s 4ms/step - loss: 0.2637 - sparse_categorical_accuracy: 0.9254 - val_loss: 0.3225 - val_sparse_categorical_accuracy: 0.9121

# evaluation
394/394 [==============================] - 0s 543us/step - loss: 0.3045 - sparse_categorical_accuracy: 0.9178
```

* model.fit()

  verbose=1 : epoch 당 loss, accuracy 출력
  
  loss : training data set으로 계산한 loss
  
  val_loss : cross-validation data set으로 계산한 loss
  
* evaluate() : loss, accuracy 출력

#### 3) 그래프 분석

history 객체 내의 history 속성 사용하여 underfitting, overfitting 확인하기

```python
print(type(history.history))
print(history.history.keys())
```

```python
<class 'dict'>
dict_keys(['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])
```

* sparse_categorical_accuracy : training data set으로 계산한 accuracy
* val_sparse_categorical_accuracy : validation data set으로 계산한 accuracy

```python
plt.plot(history.history['sparse_categorical_accuracy']) 
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.show()
```

![다운로드-1616320645717](https://user-images.githubusercontent.com/72610879/112250371-9a831200-8c9c-11eb-8219-66fac091235e.png)

가로축: epochs

세로축: accuracy

epochs 값 커질 수록 두 그래프 차이 발생(overfitting)

두 그래프 차이 거의 없는 지점(epochs=5)을 epoch 수로 다시 설정하여 모델 재학습(train & accuracy 과정 반복)

### 3. Scikit-Learn

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

경고창 무시하고 싶을 때

```python
import warnings
warnings.filterwarnings(action='ignore')
```

```python
# Raw Data
df = pd.read_csv('./data/mnist/train.csv')
x_data = df.drop('label' ,axis=1, inplace=False)
y_data = df['label']

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

# Data Split
x_tr, x_tst, y_tr, y_tst = train_test_split(x_data, y_data, test_size=0.3, random_state=3)
```

* sklearn으로 머신러닝 구현할 때 사용가능한 feature scaling 방법

  1) x_data: feature scale O, y_data: feature scale O
  2) x_data: feature scale O, y_data: feature scale X
  3) x_data: feature scale X, y_data: feature scale X

  셋 모두 결과 동일

#### 2) Train & Accuracy(classification_report)

```python
# 모델 생성
sklearn_model = LogisticRegression(solver='saga')

# 모델 학습
sklearn_model.fit(x_tr,y_tr)

# 모델 평가
y_predict = sklearn_model.predict(x_tst)
evaluation = classification_report(y_tst, y_predict)
print(evaluation)
```

* solver

lbfgs(default) : 적은 양의 데이터 처리에 적합한 알고리즘

sag(Stochastic Average Gradient Descent) : 많은 양의 데이터 처리에 적합한 알고리즘

saga : sag의 확장판, sag보다 더 나은 성능

```python
     			precision    recall  f1-score   support

           0       0.95      0.97      0.96      1214
           1       0.95      0.97      0.96      1362   
        ... 
           9       0.89      0.90      0.90      1290

    accuracy                           0.92     12600
   macro avg       0.92      0.92      0.92     12600
weighted avg       0.92      0.92      0.92     12600
```

* support: 예측값과 실제값이 일치하는 갯수

## 2. DNN

> Deep Neural Network = Deep Learning(DNN은 가장 일반적인 딥러닝 구조)

Input layer,  hidden layer, output layer로 구성

hidden layer 갯수가 많을수록 deep

output layer 이외의 layer에 있는 각각의 노드는 logistic regression 1개 가리킴

### 1. tensorflow.v1

#### 1) Data Processing

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
```

```python
# Raw Data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/train.csv')

# train, test data split
x_tr, x_tst, y_tr, y_tst = train_test_split(df.drop('label', axis=1, inplace=False), df['label'], test_size=0.3, random_state=2)

# normalization
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# one-hot encoding
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr, depth=10))
y_tst = sess.run(tf.one_hot(y_tst, depth=10))
```

#### 2) Random Initialization

> 실제로는 사용 X, Xavier Initialization & He Initialization 과 비교용

```python
# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)

# weight & bias (hidden layer 3개)
W1 = tf.Variable(tf.random.normal([784,64]))
b1 = tf.Variable(tf.random.normal([64]))
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random.normal([64,32]))
b2 = tf.Variable(tf.random.normal([32]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random.normal([32,16]))
b3 = tf.Variable(tf.random.normal([16]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random.normal([16,10]))
b4 = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
sess.run(tf.global_variables_initializer())

# train
epochs = 1000
for step in range(epochs):
  _, loss_val = sess.run([optimizer,loss], feed_dict={X:x_tr, Y:y_tr})
  if step % 100 ==0:
    print('loss:', loss_val)
```

```python
loss: 4.669587
...
loss: 1.321085
```

```python
# accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32)) 

print('accuracy:', sess.run(accuracy, feed_dict={X:x_tst, Y:y_tst}))
```

```python
accuracy: 0.5996825
```

*random initialization으로 인해 hidden layer를 사용했음에도 hidden layer를 전혀 사용하지 않은 경우보다 accuracy가 낮게 나온다* 

#### 3) Xavier Initialization

> 입력갮수와 출력 갯수 이용하여 초기값 결정

```python
# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)

# weight & bias (hidden layer 3개)
W1 = tf.get_variable('W1', shape=[784,64], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random.normal([64]))
layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.get_variable('W2', shape=[64,32], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random.normal([32]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.get_variable('W3', shape=[32,16], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random.normal([16]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable('W4', shape=[16,10], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
sess.run(tf.global_variables_initializer())

# train
epochs = 1000
for step in range(epochs):
  _, loss_val = sess.run([optimizer,loss], feed_dict={X:x_tr, Y:y_tr})
  if step % 100 ==0:
    print('loss:', loss_val)
```

* tf.contrib.layers.xavier_initializer() : Xavier Initialization사용

* tf.nn.relu() : hidden layer에는 relu 사용. sigmoid 보다 더 정확도 높기 때문(+ back propagation사용할 경우 vanishing gradients 문제도 해결)

  vanishing gradients :  가중치(W)가 0으로 수렴(back propagation은 미분 곱(chain rule)으로 구성, sigmoid function의 미분 값이 매우 작기 때문에 layer가 많아지면 미분값은 사실상 0)

```python
loss: 3.37974
...
loss: 0.16961718
```

```python
# accuracy
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32)) 

print('accuracy:', sess.run(accuracy, feed_dict={X:x_tst, Y:y_tst}))
```

```python
accuracy: 0.9347619
```

#### 4) He's Initialization

> Xavier Initialization의 확장판

```python
# placeholder
X = tf.placeholder(shape=[None,784], dtype=tf.float32)
Y = tf.placeholder(shape=[None,10], dtype=tf.float32)
 
# weight & bias (hidden layer 3개)
W1 = tf.get_variable('W1', shape=[784,64], initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([64]))
layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.get_variable('W2', shape=[64,32], initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([32]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.get_variable('W3', shape=[32,16], initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([16]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable('W4', shape=[16,10], initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]))

# hypothesis
logit = tf.matmul(layer3, W4) + b4
H = tf.nn.softmax(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
sess.run(tf.global_variables_initializer())

# train
epochs = 1000
for step in range(epochs):
  _, loss_val = sess.run([optimizer,loss], feed_dict={X:x_tr, Y:y_tr})
  if step % 100 ==0:
    print('loss:', loss_val)
```

* tf.contrib.layers.variance_scaling_initializer() : He Initialization 사용
* tf.nn.relu() : hidden layer에는 relu 사용. sigmoid 보다 더 정확도 높기 때문(+ back propagation사용할 경우 vanishing gradients 문제도 해결)

```python
loss: 3.7758472
...
loss: 0.14492379
```

```python
accuracy: 0.9461905
```

#### 5) Dropout

> layer  안의 일부 노드를 작동시키지 않는 것

dropout할 layer 앞에 _를 붙인다

overfitting 해결 방법 중 하나

```python
# weight & bias
W1 = tf.get_variable('W1', shape=[784,64], initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([64]))
layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)

W2 = tf.get_variable('W2', shape=[64,32], initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([32]))
_layer2 = tf.nn.relu(tf.matmul(layer1, _W2) + _b2)
layer2 = tf.nn.dropout(_layer2, rate=0.4)

W3 = tf.get_variable('W3', shape=[32,16], initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([16]))
_layer3 = tf.nn.relu(tf.matmul(layer2, _W3) + b3)
layer3 = tf.nn.dropout(_layer3, rate=0.4)

W4 = tf.get_variable('W4', shape=[16,10], initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([10]))
```

* He Initialization + dropout 
* hidden layer1, hidden layer2 노드의 0.4만큼을 dropout

### 2) tensorflow.v2

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

```python
# Raw Data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data/train.csv')

# train, test data split
x_tr, x_tst, y_tr, y_tst = train_test_split(df.drop('label', axis=1, inplace=False), df['label'], test_size=0.3, random_state=2)

# normalization
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)
```

#### 2) Train & Accuracy

> 적절한 initialization 자동 설정

```python
# 모델 생성
model = Sequential()

# input layer
model.add(Flatten(input_shape=(784,)))

# hidden layer1
model.add(Dense(64, activation='relu'))

# hidden layer2
model.add(Dense(32, activation='relu'))

# hidden layer3
model.add(Dense(16, activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))

# 모델 설정
model.compile(optimizer=Adam(learning_rate=1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(x_tr, y_tr, epochs=5, verbose=1, validation_split=0.3, batch_size=1000)

# accuracy
result = model.evaluate(x_tst, y_tst)
print(result)
```

* input_shape : 데이터 한 줄에 대한 정보 입력

* Dense() : FC layer에 사용, 여기서는 hidden layer, output layer모두 FC layer
* DNN의 hidden layer activation 함수는 relu사용(sigmoid 사용 X)

```python
Epoch 1/5
21/21 [==============================] - 1s 26ms/step - loss: 1.6458 - accuracy: 0.4190 - val_loss: 0.5003 - val_accuracy: 0.8534
...
Epoch 5/5
21/21 [==============================] - 0s 17ms/step - loss: 0.1248 - accuracy: 0.9624 - val_loss: 0.1576 - val_accuracy: 0.9546
     
394/394 [==============================] - 0s 1ms/step - loss: 0.1669 - accuracy: 0.9524
[0.1669045090675354, 0.9523809552192688]
```

#### 3) Dropout

> layer  안의 일부 노드를 작동시키지 않는 것

overfitting 해결방법 중 하나

```python
# input layer
model.add(Flatten(input_shape=(784,)))

# hidden layer1
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# hidden layer2
model.add(Dense(32, activation='relu'))

# hidden layer3
model.add(Dense(16, activation='relu'))

# output layer
model.add(Dense(10, activation='softmax'))
```

* hidden layer1의 노드의 0.2만큼을 작동 X
