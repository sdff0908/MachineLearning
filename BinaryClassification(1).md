 ## Binary Classification

### 1. Simple Logistic Regression

#### 1) Data Processing

```python
import numpy as np
import tensorflow as tf
from sklearn import linear_model
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# 1. Raw Data
df = pd.read_csv('C:/Users/Hwayeon Kim/Desktop/admission/admission.csv')
training_data = df[['gpa', 'admit']]

# 2. 결측치 확인
print(training_data.isnull().sum())  # True = 1

# 3. Delete Outlier
# admit column값은 0 또는 1 --> Outlier 처리 X
zscore_threshold = 2.0
tmp = ~(np.abs(stats.zscore(training_data['gpa'])) > zscore_threshold)
training_data = training_data.loc[tmp]

# 4. Feature Scaling
# y는 0 또는 1 --> scaling 필요 X
scaler_x = MinMaxScaler()
scaler_x.fit(training_data['gpa'].values.reshape(-1,1))
scaled_x_data = scaler_x.transform(training_data['gpa'].values.reshape(-1,1))

# 5. Training Data Set
x_data = scaled_x_data.reshape(-1,1)
y_data = training_data['admit'].values.reshape(-1,1) 
```

```python
gpa      0
admit    0
dtype: int64
```

#### 2) Tensorflow.v1

```python
# 1. Placeholder
X = tf.placeholder(shape=[None,1], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# 2. Weight & Bias
W = tf.Variable(tf.random.normal([1,1]))
b = tf.Variable(tf.random.normal([1]))

# 3. Hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# 4. Loss Function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))

# 5. Train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# 6. Session & Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], feed_dict={X: x_data, Y: y_data})
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W_val, b_val, loss_val))
        
# 7. Prediction
my_gpa = np.array([[3.72]])
scaled_gpa = scaler_x.transform(my_gpa)
tf_result = sess.run(H, feed_dict={X: scaled_gpa})
print(tf_result)
```

```python
W: [[0.9588941]], b: [1.7938426], loss: 1.665480613708496
...
W: [[0.0922185]], b: [-0.7647466], loss: 0.6248767971992493
            
[[0.4801474]]   # -> 0         
```

#### 3) Numerical Differentiation

```python
# 1. Weight & Bias
W = np.random.rand(1,1)
b = np.random.rand(1)

# 2. Loss Function
def loss_func(input_obj):
    input_W = input_obj[0].reshape(-1,1)
    input_b = input_obj[1]
    
    # linear regression의 hypothesis
    z = np.dot(x_data, input_W) + input_b
    # logistic regression의 hypothesis
    y = 1 / (1 + np.exp(-z))
    # log 연산 시 무한대 발산 방지(0에 가까운 값 설정)
    delta = 1e-7
    # cross entropy
    return -np.mean(y_data * np.log(y + delta) + (1 - y_data) * np.log(1 - y + delta))

# 3. Differentiation
def numerical_diff(f,x):
    # f: loss function
    # x: [W,b]
    
    delta_x = 1e-4
    diff_x = np.zeros_like(x)
    
    it = np.nditer(x,flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)
        
        diff_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp
        it.iternext()
    return diff_x

# 4. Train
learning_rate = 1e-4
for step in range(300000):
    input_param = np.concatenate((W.ravel(), b), axis=0) # [W b]
    diff_result = learning_rate * numerical_diff(loss_func,input_param)
    W = W - diff_result[0].reshape(-1,1)
    b = b - diff_result[1]
    
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W, b, loss_func(input_param)))
        
# 5. Prediction
def logistic_predict(x):
    z = np.dot(x,W) + b
    y = 1 / (1 + np.exp(-z))
    
    if y < 0.5:
        result = 0
    else:
        result = 1
    return result, y

my_gpa = np.array([[3.72]])
scaled_gpa = scaler_x.transform(my_gpa)
diff_result = logistic_predict(scaled_gpa)
print(diff_result)
```

```python
W: [[0.81576835]], b: [0.57882852], loss: 2.2975063459244365
...
W: [[-0.27543204]], b: [0.23124273], loss: 0.6362523990189881
            
(0, array([[0.44766789]]))        
```

#### 4) Scikit-Learn

```python
# 1. Training Data Set(Feature Scaling X)
x_data = training_data['gpa'].values.reshape(-1,1)   
y_data = training_data['admit'].values.reshape(-1,1)   

# 2. 모델 생성 및 학습
model = linear_model.LogisticRegression()
model.fit(x_data, y_data.ravel())

# 3. Prediction
my_gpa = np.array([[3.72]])
sk_result = model.predict(my_gpa)
print(sk_result)
sk_result_proba = model.predict_proba(my_gpa)
print(sk_result_proba)    # [0 일 확률, 1일 확률]
```

```python
[0]
[[0.61449651 0.38550349]]
```

### 2. Multivariate Logistic Regression

#### 1) Data Processing

```python
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


# 1. Raw Data
df = pd.read_csv('C:/Users/Hwayeon Kim/Desktop/admission/admission.csv')

# 2. Delete NaN
print(df.isnull().sum()) # NaN 없음

# 3. Check Outlier
fig = plt.figure()
fig_gre = fig.add_subplot(1,3,1)
fig_gpa = fig.add_subplot(1,3,2)
fig_rank = fig.add_subplot(1,3,3)
fig_gre.boxplot(df['gre'])
fig_gpa.boxplot(df['gpa'])
fig_rank.boxplot(df['rank'])

fig.tight_layout()  # 그래프 겹침 해결
plt.show()

# 4. Delete Outlier
zscore_threshold = 2.0
for col in df.columns:
    outlier = df[col][np.abs(stats.zscore(df[col])) > zscore_threshold]
    df = df.loc[~df[col].isin(outlier)]

# 5. Feature Scaling
# y는 0 또는 1 --> Scaling 필요 X
x_data = df.drop('admit', axis=1).values
y_data = df['admit'].values.reshape(-1,1)

scaler_x = MinMaxScaler()
scaler_x.fit(x_data)
norm_x_data = scaler_x.transform(x_data)
```

```python
admit    0
gre      0
gpa      0
rank     0
dtype: int64
```

![다운로드-1614792657699](https://user-images.githubusercontent.com/72610879/112245126-a74f3800-8c93-11eb-9b93-a9a7195d78ab.png)


#### 2) Tensorflow.v1

```python
# 1. Placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# 2. Weight & Bias
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

# 3. Hypothesis
logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

# 4. Loss Function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=Y))

# 5. Train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# 6. Sessin & Initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train,W,b,loss], feed_dict={X: norm_x_data, Y: y_data})
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W_val, b_val, loss_val))
        
# 7. Prediction
score = np.array([[600, 3.8, 1]])
scaled_score = scaler_x.transform(score)
result = sess.run(H, feed_dict={X: scaled_score})
print(result)
```

```python
W: [[-1.4075098]
 [-0.6218532]
 [-1.1270552]], b: [-0.01542989], loss: 0.7345348596572876
...
W: [[-0.5262636 ]
 [ 0.15691714]
 [-1.3751116 ]], b: [0.15214665], loss: 0.6087643504142761
[[0.5063241]]  # -> 1
```

#### 3) Numerical Differentiation

```python
# 1. Weight & Bias
W = np.random.rand(3,1)
b = np.random.rand(1)

# 2. Differentiation
def numerical_diff(f,x):
    # f: loss function
    # x: [W,b]
    
    delta_x = 1e-4
    diff_x = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tmp = x[idx]
        
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x) 
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)
        
        diff_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp
        it.iternext()
    return diff_x

# 3. Loss Function
def loss_func(input_obj):
    # input_obj = [w1, w2, w3, b]
    input_W = input_obj[:3].reshape(-1,1)
    input_b = input_obj[3]
    
    z = np.dot(norm_x_data, input_W) + input_b
    y = 1 / (1 + np.exp(-z))
    delta = 1e-7
    
    return -np.mean(y_data * np.log(y + delta) + (1 - y_data) * np.log(1 - y + delta))

# 4. Train
learning_rate = 1e-4
for step in range(300000):
    input_param = np.concatenate((W.ravel(), b), axis=0)   # [w1, w2, w3, b]
    diff_result = learning_rate * numerical_diff(loss_func, input_param)
    W = W - diff_result[:3].reshape(-1,1)
    b = b - diff_result[3]
    
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W, b, loss_func(input_param)))

# 5. Prediction
def logistic_predict(x):
    z = np.dot(x, W) + b
    y = 1 / (1 + np.exp(-z))
    
    if y < 0.5:
        result = 0
    else:
        result = 1
    
    return result, y

score = np.array([[600, 3.8, 1]])
scaled_score = scaler_x.transform(score)
result = logistic_predict(scaled_score)
print(result)
```

```python
W: [[0.05630507]
 [0.6519065 ]
 [0.98222545]], b: [0.39613264], loss: 1.144687272953989
...
W: [[ 0.0215194 ]
 [ 0.4710651 ]
 [-0.51006866]], b: [-0.76431225], loss: 0.6046762806543858            
(0, array([[0.42373109]]))
```

#### 4) Scikit-Learn

```python
# 1. 모델 생성 및 학습
model = linear_model.LogisticRegression()
model.fit(x_data, y_data.ravel())   # (x: 2차원, y: 1차원)

# 2. Prediction
score = np.array([[600, 3.8, 1]])
result = model.predict(score)
proba = model.predict_proba(score) # [0일 확률, 1일 확률]
print(result, proba)
```

x_data: Feature Scaling 하지 않은 데이터 →  score: Tensorflow와 Differentiation과 달리 Feature Scaling X

```python
[1], [[0.43740782 0.56259218]]
```

