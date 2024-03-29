<※참고자료:6) 벡터와 행렬 연산>

<케라스에서의 텐서>

print(d.ndim),print(d.shape) 
앞서 Numpy로 각 텐서의 ndim(차원)과 shape(크기)를 출력.
예를 들어 위의 예제에서는 3차원 텐서는 3차원이고 크기는 (2, 3, 5)였습니다. 
케라스에서는 입력의 크기(shape)를 인자로 줄 때 input_shape라는 인자를 사용합니다.

print(input_shape)

input_shape의 두 개의 인자는 (input_length, input_dim)
★input_shape=(input_length, input_dim)
input_shape는 배치 크기를 제외하고 차원을 지정함. 

예를 들어 input_shape(6, 5)라는 인자값을 사용하고 배치 크기를 32라고 지정한다면 
이 텐서의 크기는 (32, 6, 5)을 의미합니다. 
만약 배치 크기까지 지정해주고 싶다면 batch_input_shape=(8, 2, 10)와 같이 인자를 주면 
이 텐서의 크기는 (8, 2, 10)을 의미합니다.

print(input_dim)
입력의 속성 수를 의미하는 input_dim

print(input_length)
시퀀스 데이터의 길이를 의미하는 input_length 등의 인자도 사용합니다. 

사실 라고 볼 수 있습니다.


<벡터와 행렬 연산>
앞서 독립 변수 x가 2개 이상인 선형 회귀와 로지스틱 회귀에 대해서 배웠습니다. 
그런데 다음 챕터의 소프트맥스 회귀에서는 종속 변수 y의 종류도 3개 이상이 되면서 더욱 복잡해집니다. 
그리고 이러한 식들이 겹겹이 누적되면 인공 신경망의 개념이 됩니다.

케라스는 사용하기가 편리해서 이런 고민을 할 일이 상대적으로 적지만, 
Numpy나 텐서플로우의 로우-레벨(low-level)의 머신 러닝 개발을 하게되면 
각 변수들의 연산을 벡터와 행렬 연산으로 이해할 수 있어야 합니다. 

다시 말해 사용자가 데이터와 변수의 개수로부터 행렬의 크기, 더 나아 텐서의 크기를 산정할 수 있어야 합니다. 
이번 챕터에서는 기본적인 벡터와 행렬 연산에 대해서 이해해보겠습니다.

1.벡터 
벡터는 크기와 방향을 가진 양입니다. 
숫자가 나열된 형상이며 파이썬에서는 1차원 배열 또는 리스트로 표현합니다. 

2.행렬
행렬은 행과 열을 가지는 2차원 형상을 가진 구조입니다. 파이썬에서는 2차원 배열로 표현합니다. 
가로줄을 행(row)라고 하며, 세로줄을 열(column)이라고 합니다. 

3.텐서와 행렬연산
 3차원부터는 주로 텐서라고 부릅니다. 텐서는 파이썬에서는 3차원 이상의 배열로 표현합니다.
 인공 신경망은 복잡한 모델 내의 연산을 주로 행렬 연산을 통해 해결합니다. 
 그런데 여기서 말하는 행렬 연산이란 단순히 2차원 배열을 통한 행렬 연산만을 의미하는 것이 아닙니다. 
 머신 러닝의 입, 출력이 복잡해지면 3차원 텐서에 대한 이해가 필수로 요구됩니다. 
 예를 들어 인공 신경망 모델 중 하나인 RNN에서는 3차원 텐서에 대한 개념을 모르면 RNN을 이해하기가 쉽지 않습니다.

<텐서Tensor>

■위키피디아 정의
선형대수학에서, 다중선형사상(multilinear map)또는 텐서(tensor)는 
선형 관계를 나타내는 다중선형대수학의 대상이다. 
19세기에 카를 프리드리히 가우스가 곡면에 대한 미분 기하학을 만들면서 도입하였다. 
기본적인 예는 내적과 선형 변환이 있으며 미분 기하학에서 자주 등장한다. 
텐서는 기저를 선택하여 다차원 배열로 나타낼 수 있으며, 
기저를 바꾸는 변환 법칙이 존재한다. 
텐서 미적분학에서는 리치 표기법, 펜로즈 표기법, 지표 표기법, 비교적 단순한 문맥에서 사용하는 아인슈타인 표기법 
등의 다양한 표기법을 사용하여 텐서를 구체적으로 나타낸다.

1) 0차원 텐서(0D텐서): 스칼라
스칼라는 하나의 실수값으로 이루어진 데이터를 말합니다. 
또한 스칼라값을 0차원 텐서라고 합니다. 차원을 영어로 Dimensionality라고 하므로 0D 텐서라고도 합니다.

d=np.array(5)
print(d.ndim) # 텐서의 차원수=축의갯수 출력
print(d.shape) # 텐서 크기 출력,여기서는0,즉 크기없음

2) 1차원 텐서(1D텐서): 벡터
숫자를 특정 순서대로 배열한 것을 벡터라고합니다. 
또한 벡터를 1차원 텐서라고 합니다. 
주의할 점★은 벡터의 차원과 텐서의 차원은 다른 개념이라는 점입니다. 
아래의 예제는 4차원 벡터(무시)이지만, 1차원 텐서입니다. 1D 텐서라고도 합니다.
(벡터의 차원은 너무 어렵고 헷갈리므로 무시,텐서의 차원으로만 생각)

d=np.array([1, 2, 3, 4])
print(d.ndim) #1 => 1차원 텐서
print(d.shape) #(4,)=>보고 바로 1차원텐서로 이해해야 ,4차원벡터(무시)표현무시

※벡터의 차원과 텐서의 차원
벡터의 차원과 텐서의 차원의 정의로 인해 혼동할 수 있는데 
벡터에서의 차원(Dimensionality)은 하나의 축에 차원들이 존재하는 것이고, 
텐서에서의 차원(Dimensionality)은 축의 개수를 의미합니다.

3) 2차원 텐서(2D텐서): 행렬(matrix)
행과 열이 존재하는 벡터의 배열. 
즉, 행렬(matrix)을 2차원 텐서라고 합니다. 2D 텐서라고도 합니다.

d=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(d.ndim) #2 => 2차원 텐서
print(d.shape) #(3, 4) =>보고 바로 2차원텐서로 이해해야,커다란 데이터 3개에 작은 데이터 4개 있구나 파악하기!! 
            
텐서의 크기를 보고 머리 속에 떠올릴 수 있으면 모델 설계 시에 유용합니다. 
이게 어렵다면, 큰 단위부터 확장해나가며 생각하면 됩니다. 
위의 경우 3개의 커다란 데이터가 있는데, 
그 각각의 커다란 데이터는 작은 데이터 4개로 이루어졌다고 생각할 수 있습니다.

1차원 텐서를 벡터, 2차원 텐서를 행렬로 비유하였는데 
수학적으로 행렬의 열을 열벡터로 부르거나, 열벡터를 열행렬로 부르는 것과 혼동해서는 안 됩니다. =>열벡터,열행렬 무시
여기서 말하는 1차원 텐서와 2차원 텐서는 차원 자체가 달라야 합니다.

4) 3차원 텐서(3D 텐서): 텐서
행렬 또는 2차원 텐서를 단위로 한 번 더 배열하면 3차원 텐서라고 부릅니다. 
3D 텐서라고도 합니다. 
사실 위에서 언급한 0차원 ~ 2차원 텐서는 각각 스칼라, 벡터, 행렬이라고 해도 무방하므로 
3차원 이상의 텐서부터 본격적으로 텐서라고 부릅니다. 
조금 쉽게 말하면 데이터 사이언스 분야 한정으로 
주로 3차원 이상의 배열을 텐서라고 부릅니다. (엄밀한 수학적 정의로는 아닙니다.) 
그렇다면 3D 텐서는 적어도 여기서는 3차원 배열로 이해하면 되겠습니다.

이 3차원 텐서의 구조를 이해하지 않으면, 
복잡한 인공 신경망의 입, 출력값을 이해하는 것이 쉽지 않습니다. 
개념 자체는 어렵지 않지만 반드시 알아야하는 개념입니다.

d=np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
print(d.ndim)  #3  => 3차원 텐서
print(d.shape) #(2, 3, 5) =>보고 바로 3차원텐서로 이해해야,커다란 데이터 2개안에 중간 데이터가 3개,그안에 작은 데이터 5개 있구나 파악하기!! 

자연어 처리에서 특히 자주 보게 되는 것이 이 3D 텐서입니다. ex>RNN,LSTM,시계열데이터
3D 텐서는 시퀀스 데이터(sequence data)를 표현할 때 자주 사용되기 때문입니다. 
여기서 시퀀스 데이터는 주로 단어의 시퀀스를 의미하며, 
시퀀스는 주로 문장이나 문서, 뉴스 기사 등의 텍스트가 될 수 있습니다. 
이 경우 3D 텐서는 (samples, timesteps, word_dim)이 됩니다. 
또는 일괄로 처리하기 위해 데이터를 묶는 단위인 배치의 개념에 대해서 뒤에서 배울텐데 
(batch_size, timesteps, word_dim)이라고도 볼 수 있습니다.

samples/batch_size는 데이터의 개수, 
timesteps는 시퀀스의 길이, 
word_dim은 단어를 표현하는 벡터의 차원을 의미합니다. 
더 상세한 설명은 RNN 챕터에서 배우게 되겠지만 자연어 처리에서 왜 3D 텐서의 개념이 사용되는지 간단한 예를 들어봅시다. 
다음과 같은 훈련 데이터가 있다고 해봅시다.

문서1 : I like NLP
문서2 : I like DL
문서3 : DL is AI

이를 인공 신경망의 모델의 입력으로 사용하기 위해서는 각 단어를 벡터화해야 합니다. 
단어를 벡터화하는 방법으로는 print(원핫인코딩)이나 print(워드 임베딩)이라는 방법이 
대표적이나 워드 임베딩은 아직 배우지 않았으므로 원-핫 인코딩으로 모든 단어를 벡터화 해보겠습니다.

단어	One-hot vector
I	[1 0 0 0 0 0]
like[0 1 0 0 0 0]
NLP	[0 0 1 0 0 0]
DL	[0 0 0 1 0 0]
is	[0 0 0 0 1 0]
AI	[0 0 0 0 0 1]
그럼 기존에 있던 훈련 데이터를 모두 원-핫 벡터로 바꿔서 
인공 신경망의 입력으로 한 꺼번에 사용한다고 하면 다음과 같습니다. 
(이렇게 훈련 데이터를 여러개 묶어서 한 꺼번에 입력으로 사용하는 것을 배치(Batch)라고 합니다.)

[[                                                            #=> [
[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], #=> [[I, like, NLP]]
[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], #=> [[I, like, DL]]
[[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]  #=> [[DL, is, AI]]
]                                                             #=> ]

이는 (3, 3, 6)의 크기를 가지는 3D 텐서입니다.

5) 그 이상의 텐서 =>블로그 그림(시각화)보기
3차원 텐서를 배열로 합치면 4차원 텐서가 됩니다. 
4차원 텐서를 배열로 합치면 5차원 텐서가 됩니다. 
이런 식으로 텐서는 배열로서 계속해서 확장될 수 있습니다


