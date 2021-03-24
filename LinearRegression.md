## Linear Regression(ozone.csv)

### 1. Simple Linear Regression

> feature(x) 1개, label(y) 1개

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn import linear_model

# 1. Raw Data
df = pd.read_csv('C:/Users/Hwayeon Kim/Desktop/ozone/ozone.csv')
training_data = df[['Temp', 'Ozone']]

# 2. 결측치 처리
training_data = 
training_data.dropna(how='any')

# 3-1. Temp Outlier: x_data 이상치 처리
zscore_threshold = 1.8
tmp = ~ (np.abs(stats.zscore(training_data['Temp'])) > zscore_threshold)
training_data = training_data.loc[tmp]
print(training_data.shape)
 
# 3-2. Ozone Outlier: y_data 이상치 처리
tmp = ~ (np.abs(stats.zscore(training_data['Ozone'])) > zscore_threshold)
training_data = training_data.loc[tmp]
print(training_data.shape)

# 4. Feature Scaling: MinMax Scaler
scaler_x = MinMaxScaler()  
scaler_y = MinMaxScaler()  
scaler_x.fit(training_data['Temp'].values.reshape(-1,1)) # 2차원 행렬 들어가야 → shape 명시!
scaler_y.fit(training_data['Ozone'].values.reshape(-1,1))

training_data['Temp'] = scaler_x.transform(training_data['Temp'].values.reshape(-1,1))
training_data['Ozone'] = scaler_y.transform(training_data['Ozone'].values.reshape(-1,1))

# 5. Training Data Set
x_data = training_data['Temp'].values.reshape(-1,1)
y_data = training_data['Ozone'].values.reshape(-1,1)
```

#### 2) Tensorflow.v1

```python
# 1. Placeholder
X = tf.placeholder(shape=[None,1], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)
# 2차원 이상일 때 반드시 shape 표현
# shape=[None,1]: row 갯수는 상관없음, 1열

# 2. Weight & Bias
W = tf.Variable(tf.random.normal([1,1]))   # 2차원 matrix  
b = tf.Variable(tf.random.normal([1]))     # 1차원 배열

# 3. Hypothesis
H = tf.matmul(X,W) + b                     # y =Wx + b

# 4. Loss Functioin
loss = tf.reduce_mean(tf.square(H-Y))     

# 5. Train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# 6. Session, Initilization(variable사용시 초기화 필요, 2.x버전에서는 초기화 작업X)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], feed_dict={X: x_data,Y: y_data})

    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W_val, b_val, loss_val))

# 7. Prediction
scaled_x = scaler_x.transform(np.array([[62]]))
tf_result = sess.run(H, feed_dict={X: scaled_x})
print('scaled_tf:', tf_result)
tf_result = scaler_y.inverse_transform(tf_result)
print('tf:', tf_result)
```

1 epoch:  training data set 전체를 이용하여 1번 학습
여기서는 300000번 학습

```python
W: [[-0.59027976]], b: [1.6155155], loss: 1.0841526985168457
...
W: [[0.6904892]], b: [0.00752447], loss: 0.03105907142162323
scaled_tf: [[-0.02647829]]
tf: [[1.5375191]]
```

#### 3) Numerical Differentiation

 ```python
# 1. Weight & Bias
W = np.random.rand(1,1)
b = np.random.rand(1)

# 2. Hypothesis
def predict(x):
    y = np.dot(x,W) + b
    return y

# 3. Loss Function
def loss_func(input_obj):
    
    # input_obj = [W, b]
    input_W = input_obj[0]
    input_b = input_obj[1]
    
    y = np.dot(x_data, input_W) + input_b
    return np.mean(np.power((t_data - y), 2))

# 4. Numerical Differentiation
def numerical_diff(f,x):
    delta_x = 1e-4
    diff_x = np.zeros_like(x) 
    it = np.nditer(x, flags = ['multi_index'])
    
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

# 5. Train
learning_rate = 1e-4
for step in range(30000):
    input_param = np.concatenate((W.ravel(), b), axis=0)    # [W b]
    diff_result = learning_rate * numerical_diff(loss_func, input_param)
    
    W = W - diff_result[0].reshape(1,1) # W 갱신
    b = b - diff_result[1]              # b 갱신
    
    if step % 3000 == 0:
        print('W: {}, b: {}'.format(W,b))

# 6. Prediction 
scaled_x = scaler_x.transform(np.array([[62]]))
diff_result = predict(scaled_x)
print('scaled_diff:', diff_result)
diff_result = scaler_y.inverse_transform(diff_result)
print('diff:', diff_result)
 ```

```python
W: [[0.64228316]], b: [0.65642369]
...
W: [[0.7704586]], b: [-0.0352289]        
scaled_diff: [[-0.01518063]]
diff: [[2.58820132]]
```

#### 4) Scikit-Learn

```python
# 1. 모델 생성 및 학습
model = linear_model.LinearRegression() # 모델 생성
model.fit(x_data, y_data)                # 모델 학습

print('W: {}, b: {}'.format(model.coef_, model.intercept_))

# 2. Prediction
scaled_x = scaler_x.transform(np.array([[62]]))
sk_result = model.predict(scaled_x)
print('scaled_sk:', sk_result)
sk_result = scaler_y.inverse_transform(sk_result)
print('sk:', sk_result) 
```

*sklearn으로 모델 학습 시 feature scaling 하지 않아도 같은 결과 나옴*
*단, 모델 학습 시 feature scaling 했을 때는 새로운 입력 값 x도 feature scaling 한 값을, 모델 학습 시 feature scaling 하지 않았을 때는 입력값 x도 feature scaling하지 않은 값을 사용*

```python
W: [[0.79468511]], b: [-0.04818192]
scaled_sk: [[-0.02410055]]
sk: [[1.75864872]]
```

#### 5) 결과 비교

![다운로드-1614775454192](https://user-images.githubusercontent.com/72610879/112247496-bfc15180-8c97-11eb-8245-e596c84589b5.png)


### 2-1. Multivariate Linear Regression

> feature(x) 2개 이상, label(y) 1개

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

# 1. Raw Data
df = pd.read_csv('C:/Users/Hwayeon Kim/Desktop/ozone/ozone.csv')
training_data = df[['Temp', 'Wind', 'Solar.R', 'Ozone']]
 
# 2. 결측치 처리
training_data = training_data.dropna(how='any')

# 3. 이상치 처리
zscore_threshold = 1.8
for col in training_data.columns:
    training_data = training_data.loc[(np.abs(stats.zscore(training_data[col])) < zscore_threshold)]

# 4. Feature Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(training_data[['Temp', 'Wind', 'Solar.R']].values)   #fit: 최소값, 최대값 계산
scaler_y.fit(training_data['Ozone'].values.reshape(-1,1))         

# 5. Training Data Set
x_data = scaler_x.transform(training_data[['Temp', 'Wind', 'Solar.R']].values)  #range에 맞게 feature scaling
y_data = scaler_y.transform(training_data['Ozone'].values.reshape(-1,1))
```

#### 2) Tensorflow.v1

```python
# 1. Placeholder
X = tf.placeholder(shape=[None,3], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

# 2. Weight & Bias
W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]))

# 3. Hypothesis
H = tf.matmul(X,W) + b

# 4. Loss Function
loss = tf.reduce_mean(tf.square(H-Y))

# 5. Train
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

# 6. Session & Initilization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], feed_dict={X: x_data, Y: y_data})
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W_val, b_val, loss_val))
        
# 7. Prediction
predict_data = np.array([[80, 10, 150]])
scaled_predict_data = scaler_x.transform(predict_data)    # Feature Scaling

tf_result = sess.run(H, feed_dict={X:scaled_predict_data})   # Hypothesis에 대입
tf_result = scaler_y.inverse_transform(tf_result)    # 예측값을 inverse Feature Scaling
print('tf:', tf_result)
```

```python
W: [[0.22514378]
 [0.42179307]
 [0.08137065]], b: [-0.15654175], loss: 0.11592749506235123
...
W: [[ 0.73540336]
 [-0.30094564]
 [ 0.19213533]], b: [0.00550489], loss: 0.02388733997941017            
tf: [[38.801006]]
```

#### 3) Numerical Differentiation

```python
# 1. Weight & Bias
W = np.random.rand(3,1)
b = np.random.rand(1)

# 2. Hypothesis
def predict(x):
    y = np.dot(x,W) + b
    return y

# 3. Loss Function
def loss_func(input_obj):
    
    # input_obj = [W1, W2, W3, b]
    input_W = input_obj[:3]
    input_b = input_obj[3]
    
    y = np.dot(x_data, input_W) + input_b
    return np.mean(np.power((y_data - y), 2))

# 4. Numerical Differentiation
def numerical_diff(f,x):
    delta_x = 1e-4
    diff_x = np.zeros_like(x) 
    it = np.nditer(x, flags = ['multi_index'])
    
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

# 5. Train
learning_rate = 1e-4
for step in range(300000):
    input_param = np.concatenate((W.ravel(), b), axis=0)    # [W1, W2, W3, b]
    diff_result = learning_rate * numerical_diff(loss_func, input_param)

    W = W - diff_result[:3].reshape(-1,1)  # W 갱신
    b = b - diff_result[3]                 # b 갱신
    
    if step % 30000 == 0:
        print('W: {}, b: {}, loss: {}'.format(W,b,loss_func(input_param)))
        
# 6. Prediction 
predict_data = np.array([[80, 10, 150]])
scaled_predict_data = scaler_x.transform(predict_data)

diff_result = predict(scaled_predict_data)
diff_result = scaler_y.inverse_transform(diff_result)
print('diff:', diff_result)
```

```python
W: [[0.25699516]
 [0.04638167]
 [0.44029263]], b: [0.00946234], loss: 0.09735314643500269
...
W: [[0.05824284]
 [0.0504668 ]
 [0.03113846]], b: [0.28113415], loss: 0.07679704831106499        
diff: [[38.73149441]]
```

#### 4) Scikit-Learn

```python
# 1. Training Data Set(Feature Scaling X)
x_data = training_data[['Temp', 'Wind', 'Solar.R']].values
y_data = training_data['Ozone'].values.reshape(-1,1)

# 2. 모델 생성 및 학습
model = linear_model.LinearRegression()
model.fit(x_data, y_data)
print('W: {}, b: {}'.format(model.coef_, model.intercept_))

# 3. Prediction
sk_result = model.predict([[80, 10, 150]])
print('sk:', sk_result)
```

```python
W: [[ 1.9400749  -2.7453281   0.05651878]], b: [-97.42698439]
sk: [[38.8035437]]
```

### 2-2. Multivariate Linear Regression(KNN)

#### 1) Data Processing

```python
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
```

```python
# Raw Data
df = pd.read_csv('./data/ozone/ozone.csv')
x_data = df[['Solar.R', 'Wind', 'Temp']]
y_data =df['Ozone']

# 독립변수(x) 결측치: median을 x_data NaN 값으로 지정
for col in x_data.columns: 
    col_med = np.nanmedian(x_data[col]) 
    x_data[col].loc[x_data[col].isnull()] = col_med    

# Outlier
# x_data Outlier
threshold = 1.8
for col in x_data.columns:
    outlier = x_data[col][np.abs(stats.zscore(x_data[col])) > threshold]
    col_mean = np.mean(x_data.loc[~x_data[col].isin(outlier), col])
    x_data.loc[x_data[col].isin(outlier), col] = col_mean
# y_data Outlier
outlier = y_data[np.abs(stats.zscore(y_data)) > threshold]
col_mean = np.mean(y_data[~y_data.isin(outlier)])
y_data[y_data.isin(outlier)] = col_mean

# Feature Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x_data)
scaler_y.fit(y_data.values.reshape(-1,1))
x_data = scaler_x.transform(x_data)
y_data = scaler_y.transform(y_data.values.reshape(-1,1))

# knn_regressor에 사용할 x,y 데이터
x_tr = x_data[~np.isnan(y_data.ravel())]
y_tr = y_data[~np.isnan(y_data.ravel())]

# 종속변수(y) 결측치
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(x_tr,y_tr) 
knn_predict = knn_regressor.predict(x_data[np.isnan(y_data.ravel())])

y_data[np.isnan(y_data.ravel())] = knn_predict
```

* np.nanmedian : NaN 값 무시하고 median 계산

  median 사용 이유: 다른 통계방식에 비해 이상치 영향 덜 받음

* n_neighbors = k

* 결측치 찾는 함수

  pandas : isnull() 

  numpy  : isnan(), 괄호 안에 1차원 배열 들어가야

* knn_regressor.fit() : y값 결측치 없는 데이터로 모델 생성

* knn_predict() : y값 결측치 있는 x데이터 입력하여 y값 결측치 예상값 구하기

#### 2) Tensorflow.v2

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
```

```python
# 모델 생성
keras_model = Sequential()

# layer 생성
# input layer
keras_model.add(Flatten(input_shape=(3,1)))
# output layer
keras_model.add(Dense(1, activation='linear'))

# optimizer, learning_rate, loss 
keras_model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')

# 모델 학습
keras_model.fit(x_data,y_data,epochs=5000,verbose=0)

# prediction
test_data = [[310,15,80]] 
keras_result = keras_model.predict(scaler_x.transform(test_data))
print('keras result:', scaler_y.inverse_transform(keras_result))
```

* Sequential : forward propagation(back propagation 사용X)

* Flatten의 input_shape : 입력 데이터 형태

  x_data의 feature 3개(column갯수) → 3
x_data가 1차원 형태이면 (3, ) 으로 입력
  
* Dense( )

  FC layer에 사용
1: output layer에 있는 노드 갯수, ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/72610879/112248416-3a3ea100-8c99-11eb-9d83-3a93bf0fe990.gif)
  activation: 사용할 함수

* mse : mean squared error (오차 제곱의 평균)

* SGD : Stochastic Gradient Descent

* verbose: 출력 방식 설정, '0, 1, 2' 중 선택 가능. 각 번호에 따라 출력 방식 다름 

```python
keras result: [[37.912903]]
```

#### 3) Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
```

```python
# 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x_data, y_data)

# prediction
test_data = [[310,15,80]] 
result = model.predict(scaler_x.transform(test_data))
print('sklearn result:', scaler_y.inverse_transform(result))
```

```python
sklearn result: [[38.02938031]]
```

