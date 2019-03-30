---
layout: post
title: "Transformer[1] Attention Is All You Need(2017) - Review"
date:   2018-08-13 01:30:00 +0900
categories: [deeplearning, attention, nlp, transformer, paperreview]
---

## 1. Abstract 
- Attention Mechanism
    - CNN, RNN을 기반의 encoder-decoder model에 attention mechanism을 적용하는 것이 sequence modeling에서 좋은 성능

- Transformer
    - 오직 attention mechanism만 사용하여 model을 구성
    - CNN, RNN을 전혀 사용하지 않음

-----

## 2. Introduction
- RNN
    - input과 output의 순서 등 position 정보를 잘 반영
    - previous hidden state $$h_{t-1}$$에 의해 $$h_{t}$$가 계산되므로 parallelization을 통해 연산을 하는 것이 불가능
    - 따라서, 학습하는 것에 많은 시간을 요구
    - 또한, 구조상 sequence 내에 멀리 떨어져 있는 단어들의 관계를 잘 설명하지 못함

- Attention Mechanism
    - 문장의 길이에 상관없이 input과 output의 dependency를 계산

- Transformenr
    - 반복을 피하는 대신 attention mechanism만 사용하여 input과 output의 global dependency를 계산
    - 빠른 학습속도
    - 병렬처리(parallelization) 가능

-----

## 3. Background
- ByteNet
    - sequential한 연산을 줄이고자 등장한 모델
    - CNN을 활욜한 sequential modeling으로 sequential한 연산을 해결
    - 하지만, RNN과 마찬가지로 CNN의 구조상 sequence 내에 멀리 떨어져 있는 단어들의 관계를 잘 설명하지 못함

- Self-Attention
    - 한 문장 내에서 서로 다른 위치의 단어를 연결시키는 것, 즉, 관계성(의존성)을 찾는 것
    - 한 단어의 representation을 결정할 때, 관계가 있는 단어에 가중치를 두어 결정
    - 에를 들어, "I arrived at the bank after crossing the ~"라는 문장에서 뒤에 road로 끝나는지 혹은 river로 끝나는지에 따라서 bank가 은행을 의미하는지 혹은 강둑을 의미하는지가 결정되는데, RNN이나 CNN의 경우는 self-attention이 없기 때문에 구조상 두 단어가 멀리 떨어져 있을수록 이러한 관계성을 표현하는 것이 어려워짐 

- Transformer
    - self-attention만 이용하여 modeling
    - 빠르게 단어들 사이의 연관성을 찾을 수 있음

-----

## 4. Model Architecture
![architecture](https://files.slack.com/files-pri/T1J7SCHU7-FC4GQ1PB3/architecture.png?pub_secret=b30e6b2660)
- auto-regressive
    - 자신이 현재까지 예측한 결과 $$(y_1, y_2, \cdots, y_{t-1})$$를 바탕으로 다음 output $$y_t$$을 예측

# 4.1 Encoder and Decoder Stacks
### 4.1.1 Encoder
- 6-stack layer
    - 각각의 layer는 2개의 sub-layer로 동일하게 구성
    - sub-layer에 residual network 적용

- encoding vector = $$SL2(SL1(x))$$
    - $$x$$ = input 
    - $$SL1(x)$$ = $$LayerNorm(x+SubLayer1(x))$$
    - $$SL2(x)$$ = $$LayerNorm(x+SubLayer2(x))$$
    - $$SubLayer1$$ = multi-head self-attention mechanism
    - $$SubLayer2$$ = position-wise fully-connected feed-forward network

### 4.1.2 Decoder
- 6-stack layer
    - 각각의 layer는 3개의 sub-layer로 동일하게 구성
    - sub-layer에 residual network 적용
    - masking된 layer를 사용하여 여태까지의 output만으로 예측할 수 있게 함

- encoding vector = $$SL2(SL1(SL0(x)))$$
    - $$SL0$$ = $$LayerNorm(x+SubLayer0(x))$$
    - $$SL1$$ = $$LayerNorm(x+SubLayer1(x))$$
    - $$SL2$$ = $$LayerNorm(x+SubLayer2(x))$$
    - $$SubLayer0$$ = masked multi-head self-attention mechanism
    - $$SubLayer1$$ = multi-head attention mechanism
    - $$SubLayer2$$ = position-wise fully-connected feed-forward network

# 4.2 Attention
### 4.2.1 Scaled Dot-Product Attention
![Scaled Dot-Product Attention](https://files.slack.com/files-pri/T1J7SCHU7-FC65FCCH5/1.png?pub_secret=6dc8e76459)

- $$Attention(Q,K,V)$$ = $$softmax(\frac{QK^T}{\sqrt{d_k}})V$$
    - $$Q$$ : query matrix
    - $$K$$ : key matrix with dimension $$d_k$$
    - $$V$$ : value matrix with dimension $$d_v$$
    - $$Q$$와 관계있는 $$K$$를 찾아 해당 가중치를 $$V$$에 곱해주는 것을 의미

- meaning of $$Q$$, $$K$$, $$V$$
    - 기존의 attention에 대입하여 생각해보자면, query는 decoder의 previous hidden state에 해당하고, key와 value는 encoder의 hidden state를 의미함. 즉, previous hidden state(=query)와 관계 있는 input sequence의 단어의 hidden state(=key=value)를 찾아 그 단어에 softmax를 이용한 가중치를 곱해준 것을 의미함. 따라서, 만약 self-attention일 경우에는 $$Q$$=$$K$$=$$V$$
    - 하지만, 이 논문에서는 기존 attention처럼 $$Q$$, $$K$$, $$V$$를 그대로 사용하지 않고, 각각 linear projection을 취해서 사용하므로 $$K$$ != $$V$$이며, self-attention의 경우에도 $$Q$$!=$$K$$!=$$V$$

- additive attention보다 더 빠르고, space-efficient한 dot product attention사용
- $$\sqrt{d_k}$$ scaling
    - $$d_k$$값이 커지면 $$QK^T$$값도 커지게 되므로 softmax에서 분모의 값이 exponential하게 증가하게 된다. 결과적으로, 제일 큰 값만 1에 가까워지고, 나머지 값들은 0에 가까워지게 되어 attention의 대부분의 값이 0에 가까워지게 된다. 이렇게 될 경우 vanishing gradient가 발생하여 학습이 잘 되지 않는다. 따라서, $$\sqrt{d_k}$$를 이용하여 그 값을 scaling해줘야 한다.

### 4.2.2 Multi-Head Attention
![Multi-Head Attention](https://files.slack.com/files-pri/T1J7SCHU7-FC54F249G/2.png?pub_secret=ac5cf33e2d)

- $$MultiHeadAttention$$ = $$Concat(head_1, head_2, \cdots, head_h)W^O$$
    - $$head_i$$ = $$Attention(QW_i^Q, KW_i^K, VW_i^V)$$
    - $$W_i^Q \in R^{d_{model} \times d_k}$$ .
    - $$W_i^K \in R^{d_{model} \times d_k}$$ .
    - $$W_i^V \in R^{d_{model} \times d_v}$$ .
    - $$W_i^O \in R^{hd_v \times d_{model}}$$ .
    - $$d_k$$ = $$d_v$$ = $$d_{model}/h$$

- linear projection and concatenation
    - $$Q$$, $$K$$, $$V$$를 그냥 사용하는 것이 아닌 각각 linear projection을 $$h$$번 한 후에 concatenate하여 사용
    - $$h$$번 linear projection을 하면 서로 다른 representation으로 부터 정보를 attention할 수 있기 때문에 더 효과적 

- computation cost
    - linear projection을 해주기 때문에 연산비용이 더 요구될 것 같지만, linear projection을 통해 차원을 줄여주므로 연산 비용은 linear projection을 해주지 않는single attention과 비슷

### 4.2.3 Applications of Attention in our Model

- encoder-decoder attention
    - query = decoder's previous hidden state
    - key, value = encoder's output hidden state

- encoder self attention

- decoder self attention
    - auto-regressive model이기 때문에 masking처리

# 4.3 Position-wise Feed-Forward Networks
- $$FFN(x)$$ = $$Linear(ReLU(Linear(x)))$$
    - input dimension = 512
    - hidden layer dimension = 2048
    - output dimension = 512

- 각 position에 대하여 각각 적용, 즉, 1x1 convolution과 비슷한 원리
    - 각 position에 대한 weight는 공유
    - 서로 다른 network에 대해서는 공유x
    - ex) "나는 밥을 먹었다." 라는 문장에서, "나는", "밥을", "먹었다"라는 단어에 각각 FFN적용 

# 4.4 Embeddings and Softmax
- Embedding
    - input으로 embedding된 단어를 사용

- Softmax
    - output의 결과를 확률값으로 바꾸기 위하여 softmax 사용

# 4.5 Positional Encoding
- $$PE_{pos}$$ = pos번째 단어의 PE = $$(PE_{pos, 1}, PE_{pos, 2}, \cdots, PE_{pos, d_{model}})$$
    - $$PE_{pos, 2i}$$ = $$sin(pos/10000^{2i/dimension})$$
    - $$PE_{pos, 2i+1}$$ = $$cos(pos/10000^{2i/dimension})$$

- attention은 각 단어의 순서를 고려하지 않고, 가중치만을 계산하기 때문에 위치에 대한 정보는 고려하지 않을 수도 있음. 따라서, input embedding에 positional encoding이라는 위치 정보를 더해줌

- 각 pos(단어의 위치)의 positional encoding이 모두 다름
    - $$(PE_{pos, 1}, PE_{pos, 2}, \cdots, PE_{pos, d_{model}})$$에서 각 elts 마다 가지는 파형이 다르기 때문에 각 pos마다 positional encoding이 다르게 표현됨
    - $$PE_{pos}$$는 $$PE_{pos+k}$$ Linear Combination으로 표현할 수 있기 때문에 상대적인 위치에 의해 attention을 쉽게 학습
    
-----

## 5. Why Self-Attention
![length](https://files.slack.com/files-pri/T1J7SCHU7-FC71Y8HM1/length.png?pub_secret=1cb1411503)
- 각 layer의 computational complexity를 줄여준다는 장점
- 병렬처리가 가능한 연산의 수가 증가
- Long-range dependency의 path length가 감소
    - CNN, RNN은 두 단어의 dependency를 학습하기 위하여 많은 단계를 거쳐야하지만, Self-Attention은 바로 계산할 수 있으므로 path-length가 짧아 position 간의 의존성을 더 잘 학습하게 됨

-----

## 6. Training
# 6.1 Training Data and Batching
- WMT dataset 사용
- 25000 token 정도가 한 batch에 들어가도록 batching

# 6.2 Hardware and Schedule
- base model
    - 100000 step
    - 12 hours
- big model
    - 30000 step
    - 3.5 days

# 6.3 Optimizer
- Adam Optimizer
- Learning rate
    - lr = $$d_{model}^{-0.5} \times min(step_num^{-0.5}, step_num \times warmup_step^{-1.5})$$

# 6.4 Regularization
- Residual dropout
- Label Smoothing

-----

## 7. Results
# 7.1 Machine Translation
![result1](https://files.slack.com/files-pri/T1J7SCHU7-FC6E9SACQ/result1.png?pub_secret=a225d8d515)

# 7.2 Model Variations
![result2](https://files.slack.com/files-pri/T1J7SCHU7-FC6UD2RN0/result2.png?pub_secret=545ece18c4)

# 7.3 English Constituency Parsing
![result3](https://files.slack.com/files-pri/T1J7SCHU7-FC6UGESSY/result33.png?pub_secret=ed68fd9359)

-----

## 8. Conclusion
Transformer는 오직 Attention만 사용하여 만들어진 Encoder-Decoder model이며, RNN이나 CNN보다 훨씬 빠르게 학습되고 좋은 성능을 보여준다.

-----

## 9. Reference
- [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Ybigta deepNLP-study
- [http://dalpo0814.tistory.com/49](http://dalpo0814.tistory.com/49)
- [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- [http://blog.naver.com/PostView.nhn?blogId=hist0134&logNo=221035988217&redirect=Dlog&widgetTypeCall=true](http://blog.naver.com/PostView.nhn?blogId=hist0134&logNo=221035988217&redirect=Dlog&widgetTypeCall=true)
- [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
