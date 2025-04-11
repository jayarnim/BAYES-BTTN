# Bayesian Attention Modules

- Paper : [Bayesian Attention Modules](https://doi.org/10.48550/arXiv.2010.10604)

- Summary : [TechLog](https://jayarnim.github.io/posts/Bayesian_Attention_Modules/)

## Example

- Regression

    > 범주형 변수 `Query` 와 그 문맥이 되는 범주형 변수 `Value` 가 주어져 있다고 가정하자. 이때 특정 문맥 하에서 `Query` 의 점수를 `y` 라고 하자. 이러한 가정 하에 `Query` 의 문맥 벡터를 `BTTN` 로써 구하고, 이를 토대로 `y` 를 예측하시오.

- Binary Classification

    > 범주형 변수 `Query` 와 그 문맥이 되는 범주형 변수 `Value` 가 주어져 있다고 가정하자. 이때 특정 문맥 하에서 `Query` 의 발생 확률을 `y` 라고 하자. 이러한 가정 하에 `Query` 의 문맥 벡터를 `BTTN` 로써 구하고, 이를 토대로 `y` 를 예측하시오.

- Multi Classification

    > 범주형 변수 `Query` 와 그 문맥이 되는 범주형 변수 `Value` 가 주어져 있다고 가정하자. 이때 특정 문맥 하에서 `Query` 의 역할을 `y` 라고 하자. 이 역할은 `n_class` 가짓수가 존재한다. 이러한 가정 하에 `Query` 의 문맥 벡터를 `BTTN` 로써 구하고, 이를 토대로 `y` 를 예측하시오.

## Package

- `MODEL.py` : Model Module

- `BTTN.py` : Bayesian Attention Module

- `APPROX.py` : Sampler Module from Approx Probability Distribution

- `PRIOR.py` : Empirical Prior Probability Distribution Module

- `LOSS.py` : Loss Function, such as KL Divergence and Gaussian Negative Log-Likelihood

- `LOOP.py` : Train and Predict Loop Module

## Requirements

- `torch` 2.5.1

- `cuda` 12.1

- `pandas` 2.2.3

- `numpy` 1.26.4

- `tqdm` 4.66.5