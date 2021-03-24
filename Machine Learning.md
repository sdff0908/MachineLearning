# AI(Artificial Intelligence)

> 인간이 가진 학습능력, 추론 능력을 컴퓨터로 구현하려는 가장 포괄적인 개념

## 1. 구현 방법

* ...
* ...
* <u>Machine Learning</u>
  * Regression
  * SVM(Support Vector Machine)
  * Decision Tree
  * Random Tree
  * Naive Bayes
  * KNN
  * <u>Neural Network(CNN, RNN, ...)</u> = Deep Learning
  * Clustering(K-means, DBSCAN)
  * Reinforcement Learning

## 2. 머신러닝과 데이터마이닝

1. Data Mining

   기존 데이터 분석: 데이터 상관관계, 기존 데이터 바탕으로 새로운 특징 도출

2. Machine Learning

   기존 데이터 바탕으로 새로운 미래값 *예측*

## 3. 머신러닝 학습방법

1. 지도 학습(Supervised Learning) 

   Training Data Set에 입력값에(x) 따른 결과값(y) 존재

   * Regresson: 결과 예측값이 연속적인 실수

   * Classification: 결과 예측값이 소수의 집단으로 나뉨
     * Binary Classification: 결과 예측값이 0 또는 1
     * Multinomial Classification: 결과 예측값이 3가지 이상

2. 비지도 학습(Unsupervised Learning)

   Training Data Set에 입력값(x)에 따른 결과값(y) 없음
분류(Clustering)
   
3. 준지도 학습(Semisupervised Learning)

   Training Data Set에 결과값이 있는 데이터와 결과값이 없는 데이터가 섞여 있는 경우
비지도 학습 + 지도 학습
   
4. 강화학습(Reinforcement Learning)

## 4. 수치 미분

* 해석미분 : fx/dx , dx는 0에 가까운 극한값

* 수치미분: (f(x+delta_x) - f(x-delta_x)) / (2 * delta_x), 숫자를 넣어 직접 계산

### 1) 변수가 하나인 함수

```python
def numerical_diff(f, x):
    delta_x = 1e-5           # 10^(-5)
    return (f(x+delta_x) - f(x-delta_x)) / (2 * delta_x)


def my_func(x):
    return x ** 2


result = numerical_diff(my_func, 3)
print(result)
```

delta_x : 일반적으로 1e(-4)~1e(-5)로 설정. 1e(-8) 이하가 되면 연산오류 발생
1e = 10
해석미분에서 delta_x는 0에 가까운 극한값  → 해석미분과 달리 수치미분에서는 오류 발생

```python
6.000000000039306
```

### 2) 변수가 둘 이상인 함수

```python
import numpy as np

def numerical_diff(f,x):
    #x: 모든 값을 포함하는 numpy array
    
    delta_x = 1e-5
    diff_x = np.zeros_like(x)
    it = np.nditer(x, flags = ['multi_index'])
    
    while not it.finished:
        idx = it.multi_index  			#현재 iterator의 index 추출, tuple 형태
        tmp = x[idx]   					#현재 index 값 임시 저장
       
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)            #f(x + delta_x) = f([1.00001, 2.0]) 
        
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x)           #f(x - delta_x) = f([0.00009, 2.0])
        
        diff_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp
        
        it.iternext()   				#다음 iterator로 넘어가기
    return diff_x


def my_func(input_data):
    x = input_data[0]
    y = input_data[1]
    
    return 2*x + 3*x*y + y**3           #f(x, y) = 2x + 3xy +y^3    


param = np.array([1.0, 2.0])			
result = numerical_diff(my_func, param)
print(result)
```

array 안의 숫자는 실수 형태로 입력하기
정수 입력 시 오류 발생

```python
 [8. 15.] 
```

## 5. Tensorflow.v1

수치 연산하는 open source library
data flow graph 생성 → 노드 실행

### 1) 그래프 구성

![그림1-1614272979437](https://user-images.githubusercontent.com/72610879/112249115-8a6a3300-8c9a-11eb-95f0-f771e8edd14f.png)

* node:  수치연산, 데이터 입출력
* edge: 데이터 이동

### 2) 그래프 실행방법

```python
import tensorflow as tf

node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = node1 + node2
sess = tf.Session()
result = sess.run(node3, feed_dict = {node1: 10, node2: 20})
print(result)
```

tf.Session() : 1.x 버전에서 그래프 실행하기 위해 반드시 필요. 2.0 버전부터는 사용 X

```python
30.0
```

## 6. Tensorflow.v2

![tf](https://user-images.githubusercontent.com/72610879/112249139-9950e580-8c9a-11eb-9ca3-b5ea45a2fc09.png)

### 1) v1 과 v2 차이

v2: session, initializer, placeholder 필요X

```python
a = tf.Variable(tf.random.normal([2,2], dtype=tf.float32))

# 1.x 버전
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(W)

# 2.x 버전
print(W.numpy())
```

 v2에서 값만 출력하고 싶을 때 .numpy() 사용

### 2) 버전 삭제 및 특정 버전 설치

colab은 최신 버전 tensorflow 사용. 특정 버전 사용하려면 현재 사용 중인 버전 삭제해야

```python
!pip uninstall tensorflow
```

1.15버전 설치하기

```python
!pip install tensorflow==1.15
```

'런타임 -> 런타임 다시시작' 클릭하면 1.15버전 사용가능

최신 버전으로 되돌아가고 싶다면 '런타임 -> 런타임 초기화'

4) 현재 사용 중인 버전 확인하기

```python
import tensorflow as tf
print(tf.__version__)
```

## 7. 이상치 처리

이상치(outlier): feature 안에 들어있는 값이 일반적인 값에 비해 편차가 큰 값

### 1) 이상치 검출방식

* 정규분포 
* 베이즈 정리
* 카이제곱 분포
* Turkey fences
* z-score(표준 정규분포)

### 2) Turkey fences

IQR(Interquartile Range) value =  3사분위 값 - 1사분위 값
이상치: 1사분위 값 - (IQR value * 1.5) 미만인 값 , 3사분위 값 + (IQR value * 1.5) 초과인 값

![다운로드-1614279797072](https://user-images.githubusercontent.com/72610879/112249179-aa015b80-8c9a-11eb-911c-aa339e5ecfa2.png)

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,22.1])

fig = plt.figure()                   						 # figure 생성
fig_1 = fig.add_subplot(1,2,1)
fig_2 = fig.add_subplot(1,2,2)

fig_1.set_title('Original Data')
fig_1.boxplot(data)

iqr = np.percentile(data,75) -  np.percentile(data,25)

upper_fence = np.percentile(data,75) + (iqr * 1.5)
lower_fence = np.percentile(data,25) - (iqr * 1.5)

result = data[(data <= upper_fence) & (data >= lower_fence)]  # 정상범위 값
fig_2.set_title('Remove Outlier')
fig_2.boxplot(result)


fig.tight_layout()
plt.show()
```

![img](data:image/png;base64,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)	

### 3) z-score

표준 정규분포: 평균이 0이고 표준편차가 1인 정규분포
z-score: 표준 정규분포의 확률 변수

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/72610879/112249290-d3ba8280-8c9a-11eb-9bea-37adb668d40c.gif)

```python
import numpy as np
from scipy import stats

zscore_threshold = 1.8                   # 이상치: 상위 약 96% 이상, 하위 약 4% 이하
data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,22.1])

outlier = data[np.abs(stats.zscore(data) > zscore_threshold)]

data[np.isin(data,outlier,invert=True)]  # 정상범위 값
```

stats.zscore(data) : data 값을 z-score값으로 변환
np.isin(data,outlier) :  data 중에 outlier에 해당하는 값을 True로 남김
data[np.isin(data,outlier,invert=True)] : True/False 값 바꿈

## 8. 결측치(NaN) 처리

### 1) 결측치 제거

* Listwise

  NaN 존재하는 행 전체 삭제
  NaN이 아닌 의미 있는 데이터도 같이 삭제될 수 있음

* Pairwise

  의미있는 데이터가 삭제되는 걸 막기 위해 행 전체 삭제X, NaN만 모든 처리에서 제외
  Pairwise보다 Listwise 더 많이 사용(Pairwise의 경우 오히려 문제 발생 여지 있음)

### 2) 결측치 보정

* 평균화 기법

  평균(mean), 중앙값(median), 최빈값(mode) 
  장점: 쉽고 빠름
  단점: 통계분석에 영향 많이 미침

* 예측기법

  KNN, ...
  종속변수(y)가 결측치일 때 사용
  장점: 일반적으로 평균화 기법보다 더 나은 결과 도출

### 3) KNN(k-nearest neighbors)

* 정의

새로운 데이터가 들어왔을 때 기존 데이터들 거리가 가장 가까운 데이터 k개 뽑아서 y값 예측
instance-based algorithm

* k (이웃의 수)

k가 작을 경우 overfitting(지역적 특성 많이 반영하기 때문), k가 너무 크면 underfitting
적당히 많아야 좋은 결과 도출

* 특징

*사용 전 반드시 Feature Scaling해야*
매우 간단한 모델, 학습 과정 필요X
k 값을 정하는 기준이 없음

* 사용

KNN classification, KNN regression

## 9. Feature Scaling

데이터 값을 0~1 사이 값으로 조정하여 각 feature의 비율 통일

### 1) MinMax Normalization

모든 데이터 값을 최소 0, 최대 1로 변환
이상치에 매우 민감하기에 반드시 이상치 처리 후 사용

![CodeCogsEqn (1)](https://user-images.githubusercontent.com/72610879/112249430-1d0ad200-8c9b-11eb-95d6-8cf5bb6b8473.gif)

### 2) Standardization

이상치의 영향 크게 받지 않음

동일한 scale 사용할 수 없음

![CodeCogsEqn-1614354159929](https://user-images.githubusercontent.com/72610879/112249379-02385d80-8c9b-11eb-9cd1-3c058b808e29.gif)

## 10. Basic Algorithms

### 1) Linear Regression

* simple linear regression

feature(x) 개수가 1개인 linear regression

![CodeCogsEqn (4)-1615906627713](https://user-images.githubusercontent.com/72610879/112249547-53e0e800-8c9b-11eb-99f3-a0464219e5dc.gif)

* multiple linear regression

feature(x) 개수가 2개 이상인 linear regression

![CodeCogsEqn (3)-1615906395793](https://user-images.githubusercontent.com/72610879/112249556-593e3280-8c9b-11eb-940f-e12d5daaccc5.gif)

### 2) Logistic Regression

* binary classification

Linear Regression의 Hypothesis인 y = ax + b를 sigmoid 함수와 합성하여 y 값을 0, 1로 제한 

```python
#Sigmoid Function
import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(-8,8)
y_data = 1 / (1 + np.exp (-x_data))

plt.plot(x_data, y_data)
plt.show()
```

![다운로드-1614775454192](https://user-images.githubusercontent.com/72610879/112249576-63603100-8c9b-11eb-86f3-a1664516e40f.png)

* multinomial classification

Linear Regression의 Hypothesis인 y = ax + b를 softmax 함수와 합성, y 값 종류는 3개 이상 

## 11. Metrics

> 알고리즘 모델 성능 평가지표

### 1) Confusion Matrix

|                        | (실제 정답) True (1) | (실제 정답) False (0) |
| ---------------------- | -------------------- | --------------------- |
| **(예측값) True (1)**  | True Positive        | False Positive        |
| **(예측값) False (0)** | False Negative       | True Negative         |

### 2) Precision 

모델이 True로 분류한 것 중 실제 정답이 True인 비율

![CodeCogsEqn-1614751633608](https://user-images.githubusercontent.com/72610879/112249629-796df180-8c9b-11eb-99b3-4b8679618ff7.gif)

### 3) Recall

실제 정답이 True인 것 중 모델이 True라고 예측한 비율
Sensitivity, Hit Rate

![CodeCogsEqn (1)-1614751894565](https://user-images.githubusercontent.com/72610879/112249675-87237700-8c9b-11eb-8ef2-13b5a4a45d48.gif)

예) A 모델: 고양이 사진 검출율 99.9% + 그림 1장 당 5건 오검출  →  Recall ↑
      B모델: 고양이 사진 검출율 67.3% + 오검출X  →   Precision ↑
     *검출율 = Recall*

### 4) F1-Score

Precision 과 Recall 의 조화평균

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/72610879/112249811-bd60f680-8c9b-11eb-8bed-cda508156ced.gif)

### 5) Accuracy 

실제 정답과 분류 결과가 일치하는 비율

![CodeCogsEqn (3)-1614753808012](https://user-images.githubusercontent.com/72610879/112249856-d1a4f380-8c9b-11eb-89cc-d550e55177b8.gif)

### 6) Fall-Out	

실제 정답이 False인 것 중에 모델이 True로 잘못 예측한 비율

![CodeCogsEqn-1614819142063](https://user-images.githubusercontent.com/72610879/112249881-dd90b580-8c9b-11eb-91a3-31af1b9fabd7.gif)

## 12. Overfitting

### 1) 발생원인

* epoch 수가 클 때

* data 양 적을 때

  data 많으면 평균값 기준으로 모델 형성
  data 적으면 주어진 데이터에만 딱 맞는 모델 형성

* feature 갯수 많을 때

### 2) 해결방법

* 적정 epoch수 찾기 : training data set으로 계산한 accuracy와 validation data set으로 계산한 accuracy 비교하여 두 accuracy가 일치하는 지점 찾기

* 결과에 영향 주지 않거나 다른 feature와 성질이 중복되는 feature는 제외

* dropout 사용(DNN 한정) : 특정 layer 안의 일부 노드를 작동 X
* L2 regularization 사용



