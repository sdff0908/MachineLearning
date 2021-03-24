## Logic

### 1. Perceptron

The perceptron is an algorithm for supervised learning of binary classifiers.
It is a type of linear classifier.

### 2. AND gate

| A    | B    | result |
| ---- | ---- | ------ |
| 0    | 0    | 0      |
| 0    | 1    | 0      |
| 1    | 0    | 0      |
| 1    | 1    | 1      |

<img width="212" alt="캡처1" src="https://user-images.githubusercontent.com/72610879/112248568-7a058880-8c99-11eb-87cd-947912fd38a1.PNG">

#### 1) Training Data Set	

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
```

```python
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1]], dtype=np.float32) 
```

#### 2) 단층 perceptron(tf.v1)

단층 perceptron 사용하여 연산 가능

```python
# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & Bias
W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# Session & initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X:x_data,Y:y_data})
    if step % 3000 == 0:
        print('loss:', loss_val)
```

```python
loss: 0.9204479
...
loss: 0.006408548
```

#### 3) Evaluation

```python
y_predicted = tf.cast(H >= 0.5, dtype=tf.float32)
result = sess.run(y_predicted, feed_dict={X:x_data})
print(classification_report(y_data.ravel(), result.ravel()))
```

```python
              	precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         3
         1.0       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

### 3. OR gate

| A    | B    | result |
| ---- | ---- | :----- |
| 0    | 0    | 0      |
| 0    | 1    | 1      |
| 1    | 0    | 1      |
| 1    | 1    | 1      |

<img width="228" alt="캡처3" src="https://user-images.githubusercontent.com/72610879/112248588-80940000-8c99-11eb-8242-890204ce32c1.PNG">

#### 1) Training Data Set

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
```

```python
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [1]], dtype=np.float32) 
```

#### 2) 단층 perceptron(tf.v1)

단층 perceptron 사용하여 연산 가능

```python
# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & Bias
W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# Session & initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X:x_data,Y:y_data})
    if step % 3000 == 0:
        print('loss:', loss_val)
```

```python
loss: 1.7131114
...
loss: 0.006420457
```

#### 3) Evaluation

```python
y_predicted = tf.cast(H >= 0.5, dtype=tf.float32)
result = sess.run(y_predicted, feed_dict={X:x_data})
print(classification_report(y_data, result))
```

```python
              	precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         3
         1.0       1.00      1.00      1.00         1

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

### 4. XOR gate

> exclusive or : 두 값이 같으면 0, 두 값이 다르면 1

| A    | B    | result |
| ---- | ---- | ------ |
| 0    | 0    | 0      |
| 0    | 1    | 1      |
| 1    | 0    | 1      |
| 1    | 1    | 0      |

#### 1) Training Data Set

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
```

```python
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) 
```

#### 2) 단층 perceptron(tf.v1)

> 단층 perceptron으로 연산 불가

```python
# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# Weight & Bias
W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

# Hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# loss & optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

# Session & initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X:x_data,Y:y_data})
    if step % 3000 == 0:
        print('loss:', loss_val)
```

```python
loss: 0.71659195
...
loss: 0.6931472
loss: 0.6931472
loss: 0.6931472
```

loss 값이 일정 수준 이하로 떨어지지 않음

```python
# evaluation
y_predicted = tf.cast(H >= 0.5, dtype=tf.float32)
result = sess.run(y_predicted, feed_dict={X:x_data})
print(classification_report(y_data, result))
```

```python
              	precision    recall  f1-score   support

         0.0       1.00      0.50      0.67         2
         1.0       0.67      1.00      0.80         2

    accuracy                           0.75         4
   macro avg       0.83      0.75      0.73         4
weighted avg       0.83      0.75      0.73         4
```

#### 3) DNN(tf.v1)

```python
import tensorflow as tf
from sklearn.metrics import classification_report

# placeholder
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# hidden layer1
W1 = tf.Variable(tf.random.normal([2,10]))
b1 = tf.Variable(tf.random.normal([10]))
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

# hidden layer2
W2 = tf.Variable(tf.random.normal([10,6]))
b2 = tf.Variable(tf.random.normal([6]))
layer2 = tf.sigmoid(tf.matmul(layer1,W2) + b2)

# output layer
W3 = tf.Variable(tf.random.normal([6,1]))
b3 = tf.Variable(tf.random.normal([1]))

# hypothesis
logit = tf.matmul(layer2,W3) + b3
H = tf.sigmoid(logit)

# loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)

# session & initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for step in range(30000):
  _, loss_val = sess.run([optimizer, loss], feed_dict={X:x_data,Y:y_data})
  if step % 3000 == 0:
    print('loss:', loss_val)
```

```python
loss: 1.068806
...
loss: 0.062410843
```

```python
# evaluation
y_predicted = tf.cast(H >= 0.5, dtype=tf.float32)
result = sess.run(y_predicted, feed_dict={X:x_data})
print(classification_report(y_data, result))
```

```python
              	precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         2
         1.0       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

#### 3) DNN(tf.v2)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
```

```python
# 모델 생성
model = Sequential() 

# input layer
model.add(Flatten(input_shape=(2, )))

# hidden layer1
model.add(Dense(10, activation='relu'))

# hidden layer2
model.add(Dense(6, activation='relu')) 

# output layer
model.add(Dense(1, activation='sigmoid')) 

# 모델 설정
model.compile(optimizer=SGD(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 학습 및 학습 데이터 저장
history = model.fit(x_data, y_data, epochs=10000, verbose=0) 

# prediction
predict_val = model.predict(x_data)
result = tf.cast(predict_val >= 0.5, dtype=tf.float32).numpy().ravel()
print(classification_report(y_data.ravel(), result))
```

* Flatten() 

  input_shape=(2, )) : 입력으로 들어오는 형태 지정, column(A, B) 갯수 2개 → 2

* Dense(10, activation='relu')) hidden layer, output layer에 사용, 각 layer 안에 있는 logistic regression 갯수(=node 갯수) 입력

```python
              	precision    recall  f1-score   support

         0.0       1.00      1.00      1.00         2
         1.0       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
```

#### 4) Neural Network History

* XOR gate은 single layered perceptron으로 학습X
* DNN(multi-layered perceptron)으로 해결, 하지만 학습시간이 너무 오래 걸려 사실상 사용 불가
* back propagation(미분 행렬곱 연산)으로 해결
* back propagation은 layer 갯수 많으면 vanishing gradient현상 발생
* activation 함수, 초기화 기법 변화로 해결 : activation=>'relu', 초기화 기법=>Xavier 또는 He initialization



