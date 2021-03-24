## Binary Classification(titanic.csv)

### 1. tensorflow.v1

#### 1) Data Processing

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
```

```python
# Raw Data
df = pd.read_csv('C:/python_ML/data/titanic/train.csv')
data = df[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

# Check NaN
display(data.isnull().sum()) # --> Age: 177 rows, Embarked: 2 rows

# Age NaN processing: Female
f = data[data['Sex'] == 'female']  
f_miss = f['Age'].loc[f['Name'].str.contains('Miss')].mean(skipna=True) 
f_mrs = f['Age'].loc[f['Name'].str.contains('Mrs')].mean(skipna=True) 

data['Age'].loc[(data['Age'].isnull()) & (data['Name'].str.contains('Miss'))] = f_miss
data['Age'].loc[(data['Age'].isnull()) & (data['Name'].str.contains('Mrs'))] = f_mrs

# Age NaN processing: Male
m = data[data['Sex'] == 'male']  
m_master = m['Age'].loc[m['Name'].str.contains('Master')].mean(skipna=True) 
m_mr = m['Age'].loc[m['Name'].str.contains('Mr')].mean(skipna=True) 

data['Age'].loc[(data['Age'].isnull()) & (data['Name'].str.contains('Master'))] = m_master
data['Age'].loc[(data['Age'].isnull())] = m_mr    # others: Mr, Dr

# Embarked processing: replace str(C', 'Q', 'S', nan) --> int
data['Embarked'].replace('C', 0, inplace=True)
data['Embarked'].replace('Q', 1, inplace=True)
data['Embarked'].replace('S', 2, inplace=True)

# Embarked NaN processing
display(data.loc[data['Embarked'].isnull()])   # SibSp = 0, Parch = 0 --> Impossible to find companion
display(data.corr())  # max coef with Embarked: (Survived : -0.17), (Pclass : 0.16)
# People who have Embarked NaN were survived, and took 1st class (survived: 1, pclass: 1) --> Embarked(assumption): Q
data['Embarked'].fillna(1, inplace=True)  

# Delete name column
data = data.drop(['Name'], axis=1)

# Sex processing: replace str --> int
gender_dict = {'male': 0, 'female': 1}
data['Sex'] = data['Sex'].map(gender_dict)

# Age Feature Scaling: MinMax Scaler
age_scaler = MinMaxScaler()
age_scaler.fit(data['Age'].values.reshape(-1,1))
age_scaled = age_scaler.transform(data['Age'].values.reshape(-1,1))
data['Age'] = age_scaled

# training data set
tr_data = data.iloc[:int(data.shape[0]*0.8)]
x_tr_data = tr_data.drop(['Survived'], axis=1)
y_tr_data = tr_data['Survived'].values.reshape(-1,1)

# cross validation data set
cv_data = data.iloc[int(data.shape[0]*0.8):]
x_cv_data = cv_data.drop(['Survived'], axis=1)
y_cv_data = cv_data['Survived'].values.reshape(-1,1)
```

#### 2) Train & Accuracy

```python
X = tf.placeholder(shape=[None,6], dtype=tf.float32)
Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

W = tf.Variable(tf.random.normal([6,1]))
b = tf.Variable(tf.random.normal([1]))

logit = tf.matmul(X,W) + b
H = tf.sigmoid(logit)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(300000):
    _, W_val, b_val, loss_val = sess.run([train, W, b, loss], feed_dict={X: x_tr_data, Y: y_tr_data})
    if step % 30000 == 0:
        print('W:{}, b:{}, loss:{}'.format(W_val, b_val, loss_val))
```

```python
W:[[-0.15626302]
 [-0.17688537]
 [-0.22468024]
 [ 1.6498226 ]
 [-0.6435488 ]
 [-0.6447275 ]], b:[-0.9524513], loss:1.0660136938095093
...
W:[[-0.45496655]
 [ 1.893823  ]
 [-0.02180993]
 [-0.07211974]
 [-0.03010474]
 [-0.07637054]], b:[-0.06607109], loss:0.48686549067497253
```

#### 3) Evaluation

```python
predict = tf.cast(H > 0.5, dtype=tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

accuracy_val = sess.run(accuracy, feed_dict={X: x_cv_data, Y: y_cv_data})
print(accuracy_val)
```

```python
0.8156425
```

#### 4) Prediction

train.csv 파일 사용하여 위에서 만든 모델로 결과값 예측

```python
# Raw Data
predict_data = pd.read_csv('C:/python_ML/data/titanic/train.csv')
idx = predict_data['PassengerId'].values.reshape(-1,1)
predict_data = predict_data[['Pclass', 'Sex', 'Age', 'Name', 'SibSp', 'Parch', 'Embarked']]

# Check NaN
predict_data.isnull().sum()  # Age NaN: 86 rows
predict_data['Age'].loc[predict_data['Name'].str.contains('Master')] = m_master
predict_data['Age'].loc[predict_data['Name'].str.contains('Mr')] = m_mr
predict_data['Age'].loc[predict_data['Name'].str.contains('Miss') | predict_data['Name'].str.contains('Ms')] = f_miss
# Ms: Earliest known proposal for the modern revival of Ms. was on the book in 1901 
# --> woman who used the word 'ms' may be a modern woman affected by feminism 
predict_data['Age'].loc[predict_data['Name'].str.contains('Mrs')] = f_mrs

# Delete Name column
predict_data.drop('Name', axis=1, inplace=True)

# Data Processing: str --> int
# Embarked column
predict_data['Embarked'].replace('C', 0, inplace=True)
predict_data['Embarked'].replace('Q', 1, inplace=True)
predict_data['Embarked'].replace('S', 2, inplace=True)
# Sex column
gender_dict = {'male': 0, 'female': 1}
predict_data['Sex'] = predict_data['Sex'].map(gender_dict)

# Age Feature Scaling
predict_age_scaled = age_scaler.transform(predict_data['Age'].values.reshape(-1,1))
predict_data['Age'] = predict_age_scaled

# Run Prediction
predict_H = tf.cast(H > 0.5, dtype=tf.float32)
predict_survived = sess.run(predict_H, feed_dict={X: predict_data})

# Create csv file
result_mat = np.concatenate((idx, predict_survived), axis=1)
result = pd.DataFrame(result_mat, columns=['PassengerId', 'Survived'])
result.set_index('PassengerId', inplace=True)
result.to_csv('prediction.csv', float_format='%.f')
```

### 2. tensorflow.v2

#### 1) Data Processing

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```

```python
# Raw Data
df = pd.read_csv('./data/titanic/train.csv')

# 필요없는 column 삭제
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=False)

# SibSp column과 Parch column 결합
df['Family'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Sex column 수치화
sex_dict = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_dict)

# Embarked columm 수치화
embarked_dict = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(embarked_dict)

# Age column 결측치 처리: 결측치가 아닌 Age 값의 median 사용
df.loc[df['Age'].isnull(), 'Age'] = np.nanmedian(df['Age'].values)

# Embarked column 결측치 처리: 결측치가 아닌 Embarled 값 중 최빈값 사용
print(stats.mode(df['Embarked'], nan_policy='omit')[0])
df.loc[df['Embarked'].isnull(), 'Embarked'] = 0

# Age column 수치화
def age_category(age):
    if (age >= 0) & (age < 25):
        return 0
    elif (age >= 25) & (age<50):
        return 1
    else:
        return 2
df['Age'] = df['Age'].map(age_category)

# x_data & y_data
x_data = df.drop('Survived', axis=1, inplace=False)
y_data = df['Survived']

# train_test_split
x_tr, x_tst, y_tr, y_tst = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# Feature Scaling
scaler = MinMaxScaler()
scaler.fit(x_tr)
x_tr = scaler.transform(x_tr)
x_tst = scaler.transform(x_tst)
```

* stats.mode() : (최빈값, 최빈값을 가진 행 갯수) 출력
* nan_policy='omit' : nan값 제외하고 계산 
* map() : dict, def 에 사용 가능

#### 2) Train & Accuracy

```python
# 모델 생성
model = Sequential()

# layer 생성
model.add(Flatten(input_shape=(x_tr.shape[1],)))
model.add(Dense(1, activation='sigmoid')) 

# 모델 설정
model.compile(optimizer=SGD(learning_rate=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
# metrics???????????

# 모델 학습
model.fit(x_tr, y_tr, epochs=1000, verbose=1)

# accuracy 측정
accuracy = model.evaluate(x_tst, y_tst)
print('tf accuracy:', accuracy)
```

* input_shape: 입력 데이터의 column갯수

* Dense()

  프로그램 상에서 FC layer의미
  binary classification → 1, activation='sigmoid' 또는 'relu'
  multinomial classification → n(y값 종류 갯수), 
  activation='softmax'

* compile()

  loss: 사용할 cost function 지정
  metrics: 사용할 모델 평가 방법 지정, 'accuracy' = tf.keras.metrics.Accuracy

* verbose=1 : epoch 당 metrics 출력, 최종 metrics 출력

```python
Epoch 1/1000
20/20 [==============================] - 0s 1ms/step - loss: 0.9773 - accuracy: 0.3628
...
Epoch 1000/1000
20/20 [==============================] - 0s 800us/step - loss: 0.4558 - accuracy: 0.7945        

9/9 [==============================] - 0s 888us/step - loss: 0.4482 - accuracy: 0.7948
tf accuracy: [0.44815173745155334, 0.7947761416435242]
```

### 3. Scikit-Learn

#### 1) Data Processing

tensorflow.v2의 Data Processing과 동일

#### 2) Train & Accuracy

```python
from sklearn.linear_model import LogisticRegression
```

```python
# 모델 생성
model = LogisticRegression()

# 모델 학습
model.fit(x_tr, y_tr)

# accuracy 측정
accuracy = model.score(x_tst, y_tst)
print('sklearn accuracy:', accuracy)
```

```python
sklearn accuracy: 0.7947761194029851
```

