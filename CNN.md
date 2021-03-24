## CNN

> Convolutional Neural Network(합성곱 신경망), Convnet

### 1. DNN vs CNN

* DNN : 이미지 데이터 학습

​            똑같은 그림을 회전, 위치 변경하면 픽셀당 데이터가 다르므로 다른 이미지로 인식
​		   *이미지 데이터  그대로 학습하는 방식은 좋지 X*

![1](https://user-images.githubusercontent.com/72610879/112251007-c0f57d00-8c9d-11eb-850e-c646d438f3a2.png)
![2](https://user-images.githubusercontent.com/72610879/112251015-c5ba3100-8c9d-11eb-91f4-89897df9dff9.png)


* CNN : 이미지 특징 학습
			이미지 추출(데이터 전처리) → DNN

### 2. Image

#### 1) pixel 

이미지 구성하는 기본단위, 해상도
해상도 ↑ = 픽셀 수 ↑

#### 2) 좌표계

이미지 좌표계 사용(≠데카르트 좌표계)

<img src="![3](https://user-images.githubusercontent.com/72610879/112251165-19c51580-8c9e-11eb-8b0e-2937b3d4d537.png)" alt="	" style="zoom:48%;" />	

(m, n)행렬 : 이미지를 ndarray 로 표현, pixel(y, x)
.jpg : 3 channels (RGB : red, green, blue)
.png : 4 channels (RGBA : red, green, blue, alpha(투명도))
channel = 속성값

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
```

```python
img = Image.open('./data/img/fruits.jpg')
pixel = np.array(img)
```

```python
print('x: {}, y: {}의 pixel값(r,g,b): {}'.format(100,200,pixel[200,100]))
print('이미지 크기:', img.size)
print('이미지 shape:', pixel.shape)
```

size :  (x, y)
shape : (y, x, 채널 수)
.jpg : r, g, b → (y, x, 3)
.png : r, g, b, a → (y, x, 4)

```python
x: 100, y: 200의 pixel값(r,g,b): [93 117  19]
이미지 크기: (640, 426)  
이미지 shape: (426, 640, 3)
```

#### 3) digital image

1. binary image

   각 픽셀 값을 0(검정) 또는 1(흰)로 표현
   1 bit로 각 픽셀 표현 가능, 하지만 실제 사용하는 데이터는 픽셀 당 8 bit(1 bit 사용, 7 bit 낭비)

2. grayscale image(흑백 이미지)

    ≠ binary image
   각 픽셀 값을 0 ~ 255 값으로 표현
   픽셀 당 8 bit 사용

3. color image(컬러 이미지)

   픽셀 당 3개 channel 사용(RGB) 
   R : 0 ~ 255 (8 bit), G : 0 ~ 255 (8 bit), B : 0~255 (8 bit) → 픽셀당 24 bit 사용
   픽셀은 삼원색의 색상값 표현

   <img src="![4](https://user-images.githubusercontent.com/72610879/112251133-06b24580-8c9e-11eb-946d-4f094cb8d3e4.png)" alt="4" style="zoom:67%;" />	

```python
print(pixel)
```

```python
[[[108 117   0]
 	...
  [104 111  44]]
 	...
 [[129 110  78]
  	...
  [ 64  67  24]]]
```

#### 4) 이미지 처리

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
```

```python
img = Image.open('./data/img/fruits.jpg')
```

* 잘라내기

왼쪽 위, 오른쪽 아래 두 개의 점 기준으로 자르기

```python
crop_img = img.crop((250,200,400,400))
plt.imshow(crop_img)
plt.show()
```

crop() 안에 튜플 입력
여기서는 (250, 200), (400, 400)을 꼭짓점으로 갖는 사각형 잘라냄

* 크기 변경

```python
resize_img = img.resize((int(img.size[0]/2), int(img.size[1]/2)))
plt.imshow(resize_img)
plt.show()
```

 resize() 안에 튜플 입력

* 회전

```python
rotate_img = img.rotate(90)
plt.imshow(rotate_img)
plt.show()
```

rotate() : 360도 기준으로 입력

* 흑백처리

각 픽셀의 RGB값 평균 구해서 픽셀 당 RGB값을 하나로 통일

```python
gray_pixel = pixel.copy()
for y in range(gray_pixel.shape[0]):
    for x in range(gray_pixel.shape[1]):
        gray_pixel[y,x] = int(np.mean(gray_pixel[y,x]))
plt.imshow(gray_pixel)
plt.show()
```

* 흑백 이미지 2차원 만들기

이미지는 3차원 matrix, 흑백이미지도 3차원이지만 2차원으로 변환 가능

```python
gray_2d_pixel = gray_pixel[:,:,0]
print(gray_2d_pixel.shape)
```

* pixel data를 이미지로 저장하기

```python
gray_2d_img = Image.fromarray(gray_2d_pixel)
gray_2d_img.save('./data/2d_img1.jpg')
gray_2d_img.save('./data/2d_img2.png') 
```

 jpg파일 png로 저장 가능

### 3. Convolution

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
```

#### 1) Image Processing

```python
# original image
original = img.imread('./data/img/girl-teddy.jpg')
plt.imshow(original)
plt.show()
print(original.shape)
```

 imread() : ndarray 가져오기

<img src="![6](https://user-images.githubusercontent.com/72610879/112251070-e2eeff80-8c9d-11eb-8288-1720423f1f22.png)" alt="6" style="zoom:50%;" />	

```python
(429, 640, 3)
```

(height, width, channel) : 3차원

```python
# reshape image : 3차원 → 4차원
input_img = original.reshape((1,) + original.shape)
print(input_img.shape)
```

convolution은 4차원 이미지 사용

```python
(1, 429, 640, 3)
```

(이미지 개수, height, width, channel 개수) : 4차원

```python
# dtype 변경 : int → float
input_img = input_img.astype(np.float32)
```

```python
# channel : 3개 → 1개
ch1_img = input_img[:,:,:,0:1] 
print(ch1_img.shape)
```

input_img[: , : , : , 0] : 3차원
input_img[: , : , : , 0:1] : 4차원

* 컬러 이미지 : channel 3개(R, G, B) -> 3차원 (height, width, 3)
* 흑백 이미지 : channel 3개 -> 3차원 (height, width, 3)
                         channel 1개 -> 3차원 (height, width, 1)
                         channel 없음 --> 2차원 (height, width)

*일반적으로 컬러이미지를 흑백 처리 후 channel 1개로 만들어 convolution에 사용*
채널 수와 feature map은 상관X, 채널 여러개여도 feature map은 1개

```python
(1, 429, 640, 1)
```

#### 2) Convolution(tf.v1)

* filter
  이미지 특징 찾아내기 위한 parameter
  정사각 형태
  필터 구성요소가 CNN 학습대상, 학습 통해 이미지 특징 더 잘 추출하는 필터 생성
  다양한 특징 반영 위해 여러개의 필터 사용 가능
  필터 당 feature map(convolution 결과) 1개
  크기가 큰 필터보다 크기가 작은 필터 여러 개 사용하는 것이 더 좋음

* stride : convolution 수행 시 filter가 이미지 위에서 이동하는 간격, stride 값이 크면  feature map 데이터 ↓

* padding : 입력 데이터 둘레에 특정 값을 채워서 결과 데이터의 크기(shape)를 조정하는 것
                    convolution하면 image 크기 ↓ ( = 정보 ↓) → 학습에 부정적
                    padding 통해 convolution 후에도 image 크기 유지, 가장자리 정보 손실 방지
* zero padding : 숫자 0을 사용한 padding, 특징에 영향을 주지 X

```python
kernel = np.array([[[[1]],[[-1]],[[-1]]],
                   [[[-1]],[[0]],[[-1]]],
                   [[[-1]],[[-1]],[[1]]]], dtype=np.float32)
print(kernel.shape)

conv = tf.nn.conv2d(ch1_img, weight, strides=[1,1,1,1], padding='VALID')
sess = tf.Session()
conv_result = sess.run(conv)  # feature map
print(conv_result.shape)
```

* kernel.shape : (filter height, filter width, filter channel, filter 개수) : 4차원
* tf.nn.conv2d() : convolution함수, conv2d(이미지 행렬, 필터 행렬, stride, padding)
* strides=[1,1,1,1] : stride 1(가로 1, 세로 1), 이미지 데이터 4차원이므로 stride도 4차원에 맞춤[1,1,1,1]
* padding='VALID' : padding 처리 X
  padding='SAME' : padding 처리
* feature map : 원본 이미지에서 특징을 추출한 새로운 이미지(원본 이미지 변형본)
                                                  feature map 크기는 filter 크기, stride 영향 받음

```python
(3, 3, 1, 1) 
(1, 427, 638, 1)
```

원본 이미지에 비해 feature map 크기 줄어듦
feature map channel 개수 = filter의 개수

```python
conv_img = conv_result[0,:,:,:]
plt.imshow(conv_img) 
plt.show()	
```

<img src="![7](https://user-images.githubusercontent.com/72610879/112251089-f4380c00-8c9d-11eb-85f2-1bc59cdd848d.png)
" alt="7" style="zoom:50%;" />	

#### 3) relu

```python
relu = tf.nn.relu(conv_result)
relu_result = sess.run(relu)
```

#### 3) Pooling(tf.v1)

* 사용 이유 : convolution의 filter개수 많을 수록 convolution 이후 얻게 되는 이미지 개수(=feature map 차원) 증가 → pooling layer 사용하여 이미지 크기 축소(데이터양 감소) + 특정 feature 강조
* 종류 : max pooling, average pooling, min pooling
                    max pooling이 특징 잘 추출, CNN의 기본 pooling 방법
* 구성 : kernel, stride
            *convolution의 kernel(filter), stride와 다름*
* 특징 : kernel, stride값에 따라 feature map 크기 변화
             필요에 따라 사용하는 과정, 필수 X

```python
pooling = tf.nn.max_pool(relu_result, ksize=[1,3,3,1], strides=[1,3,3,1], padding='VALID')
pooling_result = sess.run(pooling)
print(pooling_result.shape)
```

* max_pool() : max pooling 함수
* ksize = kernel size : 3, 가장자리 1은 차원을 맞추기 위한 임시 값, 임시 값으로 1 많이 사용 
* stride : 3, ksize와 같도록 설정

```python
(1, 142, 212, 1)
```

```python
pooling_img = pooling_result[0,:,:,:]
plt.imshow(pooling_img)
plt.show()
```

<img src="![8](https://user-images.githubusercontent.com/72610879/112251113-fd28dd80-8c9d-11eb-8204-f97349849b0e.png)" alt="8" style="zoom:50%;" />	
