## Multinomial Classification(bmi.csv)  

### 1. tensorflow.v1

#### 1) Data Processing

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
```

```python
# Raw data
df = pd.read_csv('./data/bmi/bmi.csv', skiprows=3)

# Check NaN values: None
df.isnull().sum()   

# Outlier: None
fig = plt.figure()
fig_1 = plt.subplot(2,3,1)
fig_2 = plt.subplot(2,3,2)
fig_3 = plt.subplot(2,3,3)
fig_1.boxplot(df['label'])
fig_2.boxplot(df['height'])
fig_3.boxplot(df['weight'])
plt.tight_layout()
fig_1.set_title('label')
fig_2.set_title('height')
fig_3.set_title('weight')
plt.show()

# Data split
x_tr, x_tst, y_tr, y_tst = train_test_split(df[['height','weight']], df['label'], test_size=0.3, random_state=0)

# Feature Scaling: MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)

# One-hot Encoding
sess = tf.Session()
y_tr = sess.run(tf.one_hot(y_tr,depth=3))
y_tst = sess.run(tf.one_hot(y_tst,depth=3)) 
```

* train_test_split

train set과 test set 분리
test_size : 전체 set 가운데 test set 비율 
test_size=0.3  → train set(70%), test set(30%)
random_state : 값 고정(numpy 시드 설정과 동일)

* MinMaxScaler

feature별로 나눌 필요 X, 한번에 계산해도 같은 값 나옴

* One-hot Encoding

y값을 0과 1로 표현
문자열을 수치로 표현하는데 사용
depth : y값 종류, 위 예시에서 y값은 [0,1,2] → 3

```python
# mini batch gradient descent
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
Y = tf.placeholder(shape=[None,3], dtype=tf.float32)

W = tf.Variable(tf.random.normal([2,3]))
b = tf.Variable(tf.random.normal([3]))

logit = tf.matmul(X,W) + b
H = tf.nn.softmax(logit)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=Y)
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

epoch = 1000
batch_size = 100

def run_train(sess,x_data,y_data):
    print('start')
    sess.run(tf.global_variables_initializer())
    total_batch = int(x_data.shape[0]/batch_size)
    for step in range(epoch):
        for i in range(total_batch):
            batch_x = x_data[i*batch_size:(i+1)*batch_size]
            batch_y = y_data[i*batch_size:(i+1)*batch_size]            
            _, loss_val = sess.run([train,loss],feed_dict={X:batch_x, Y:batch_y})
    print('loss:', loss_val)
    print('end')
```

Y의 shape=[None, 3] ← One-hot Encoding 때문에 3
1행 : height, 2행 : weight

![CodeCogsEqn (2)](md-images/CodeCogsEqn%20(2).gif)	

multivariate classification: softmax 함수 사용
binary classification: sigmoid 함수 사용
batch_size : 메모리 허용 한도 내에서 가장 큰 숫자 선택

#### 2) train & accuracy

```python
# train
run_train(sess, x_tr, y_tr)

# evaluation
correct = tf.equal(tf.argmax(H, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

result = sess.run(accuracy, feed_dict={X:x_tst, Y:y_tst})
print('accuracy:', result)
```

```python
accuracy: 0.983
```

tf.argmax(H, axis =1) : 각 행 별로 최댓값 선택 

![CodeCogsEqn (1)](md-images/CodeCogsEqn%20(1).gif)	

tf.equal : 두 값이 같으면 True, 두 값이 다르면 False
tf.cast : True = 1, False = 0 으로 계산
학습은 train set, accuracy 측정은 test set 사용

#### 3) k-fold cross validation

1. (train_test_split 이후) train set을 k등분
2. 1/k 는 cross-validation set, (1-1/k)는 train set으로 사용 
3. 2번을 k번 반복

예) k=5

![그림](md-images/%EA%B7%B8%EB%A6%BC.png)

```python
# train
correct = tf.equal(tf.argmax(H, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

cv = 5
results = []
kf = KFold(n_splits=cv, shuffle=True)
for tr_idx, cv_idx in kf.split(x_tr):
    tr_x_data = x_tr[tr_idx]
    tr_y_data = y_tr[tr_idx]
    cv_x_data = x_tr[cv_idx]
    cv_y_data = y_tr[cv_idx]
    
    run_train(sess,tr_x_data,tr_y_data)
    results.append(sess.run(accuracy, feed_dict={X:cv_x_data, Y:cv_y_data}))
    
print('results:', results)
print('k-fold accuracy:', np.mean(results))
```

```python
results: [0.985, 0.98, 0.9825, 0.98071426, 0.98642856]
k-fold accuracy: 0.9829286
```

```python
# evaluation
final_accuracy = sess.run(accuracy, feed_dict={X:x_tst, Y:y_tst})
print('accuracy:', final_accuracy)
```

````python
accuracy: 0.98366666
````

### 2. scikit-learn

#### 1) Data Processing 

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

```python
# Raw data
df = pd.read_csv('./data/bmi/bmi.csv', skiprows=3)

# Check NaN values: None
df.isnull().sum()   

# Outlier: None
fig = plt.figure()
fig_1 = plt.subplot(2,3,1)
fig_2 = plt.subplot(2,3,2)
fig_3 = plt.subplot(2,3,3)
fig_1.boxplot(df['label'])
fig_2.boxplot(df['height'])
fig_3.boxplot(df['weight'])
plt.tight_layout()
fig_1.set_title('label')
fig_2.set_title('height')
fig_3.set_title('weight')
plt.show()

# Data split
x_tr, x_tst, y_tr, y_tst = train_test_split(df[['height','weight']], df['label'], test_size=0.3, random_state=0)

# Feature Scaling: MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)
```

#### 2) train & accuracy

```python
# 모델 생성
model = LogisticRegression()

# 모델 학습
model.fit(x_tr, y_tr)  

# mean accuracy
result = model.score(x_tst, y_tst)  
print(result)
```

```python
0.9835
```

### 3. K-Neighbor Classifier

#### 1) Data Processing

```python
# Raw data
df = pd.read_csv('./data/bmi/bmi.csv', skiprows=3)

# Check NaN values: None
df.isnull().sum()   

# Outlier: None
fig = plt.figure()
fig_1 = plt.subplot(2,3,1)
fig_2 = plt.subplot(2,3,2)
fig_3 = plt.subplot(2,3,3)
fig_1.boxplot(df['label'])
fig_2.boxplot(df['height'])
fig_3.boxplot(df['weight'])
plt.tight_layout()
fig_1.set_title('label')
fig_2.set_title('height')
fig_3.set_title('weight')
plt.show()

# Data split
x_tr, x_tst, y_tr, y_tst = train_test_split(df[['height','weight']], df['label'], test_size=0.3, random_state=0)

# Feature Scaling: MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)
```

#### 2) train & accuracy

```python
# 모델 생성
knn_classifier = KNeighborsClassifier(n_neighbors=3) 

# 모델 학습
knn_classifier.fit(x_tr, y_tr)

# mean accuracy
result = knn_classifier.score(x_tst, y_tst)  
print(result)
```

n_neighbors = k

```python
0.9971666666666666
```

