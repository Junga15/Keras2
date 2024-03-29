
※참고자료: [케라스] 무작정 튜토리얼 11 - LSTM(feat. RNN) 구현하기 :: ebb and flow
https://ebbnflow.tistory.com/135#

1.RNN(Recurrent Neural Network,순환신경망)이란?

1)FFNETs와 RNN 
일반적인 인공 신경망인 FFNets(Feed-Forward Neural Networks)와 이름에서 부터 어떤 점이 다른지 드러납니다. 
FFNets는 데이터를 입력하면 연산이 입력층에서 은닉층을 거쳐 출력층까지 차근차근 진행되고 
이 과정에서 입력 데이터는 모든 노드를 딱 한번씩 지나가게 됩니다. 
그러나 RNN은 은닉층의 결과가 다시 같은 은닉층의 입력으로 들어가도록 연결되어 있습니다
즉, FFNets는 시간 순서를 무시하고 현재 주어진 데이터만 가지고 판단합니다. 
하지만 RNN은 지금 들어온 입력데이터와 과거에 입력 받았던 데이터를 동시에 고려합니다

2)RNN의 사용
RNN은 위에서 설명한 특성 때문에, Sequence Data 또는 시계열 데이터를 다루는데 크게 도움이 됩니다. 
예를 들어, RNNs은 글의 문장, 유전자, 손글씨, 음성 신호, 센서가 감지한 데이타, 주가 등 
배열(sequence, 또는 시계열 데이터)의 형태를 갖는 데이터에서 패턴을 인식하는 인공 신경망 입니다.

RNNs은 궁극의 인공 신경망 구조라고 주장하는 사람들이 있을 정도로 강력합니다. 
RNNs은 배열 형태가 아닌 데이터에도 적용할 수 있습니다. 
예를 들어 이미지에 작은 이미지 패치(필터)를 순차적으로 적용하면 배열 데이터를 다루듯 RNNs을 적용할 수 있습니다.

3)시계열 데이터(Time series data)?
데이터 관측치가 시간적 순서를 가진 데이터이다. 
변수간 상관성이 존재이 존재하는 데이터를 다루며
주로 과거의 데이터를 통해 현재의 움직임과 미래를 예측하는데 사용된다.

