# Bayesian Attention Modules

- Paper : Bayesian Attention Modules [`PAPER`](https://doi.org/10.48550/arXiv.2010.10604) [`REPO`](https://github.com/zhougroup/BAM)

</br>

## $\text{bttn}(q,k,v)=\omega \cdot v$

- **empirical prior of $\alpha \sim \text{Log-Normal}(\phi - \frac{\sigma^{2}}{2},\sigma^{2})$**
  - $\phi=h(\text{relu}[h(k)])$

  - $h(\cdot)$ is Linear fn

- **posterior of $\alpha \sim \text{Log-Normal}(\phi - \frac{\sigma^{2}}{2},\sigma^{2})$** [`code`](https://github.com/jayarnim/BAYES-BTTN/blob/main/attn_score_fn.py)
    - scaled dot product fn: $\phi = q \cdot k / \sqrt{d}$
    
    - scaled bilinear fn: $\phi = q \cdot W_{h} \cdot k / \sqrt{d}$
    
    - concat fn: $\phi = w_{o} \cdot \text{relu}(W_{c} \cdot [q \oplus k] + b)$
    
    - additive fn: $\phi = w_{o} \cdot \text{relu}(W_{h} \cdot [q \odot k] + b)$

- **simplex projection fn** [`code`](https://github.com/jayarnim/BAYES-BTTN/blob/main/simplex_proj_fn.py)
    - smoothed linear projection fn: $\omega=\alpha^{\tau} \Bigg/ \left[\sum{\alpha^{\tau}}\right]^{\beta}$
    
    - smoothed exp projection fn: $\omega={\displaystyle\frac{\exp{\alpha}}{\tau}} / {\left[\displaystyle\sum{\frac{\exp{\alpha}}{\tau}}\right]^{\beta}}$
    
    - factors `customized`
        - $\tau$ : sharpening factor
        - $\beta$ : smoothing factor

- **$-ELBO = NLL + KL(Q \parallel \Pi)$** [`code`](https://github.com/jayarnim/BAYES-BTTN/blob/main/kl_div_fn.py)
    - $KL(Q \parallel \Pi) = \log{\displaystyle\frac{\sigma_{\Pi}}{\sigma_{Q}}} + \displaystyle\frac{\sigma_{Q}^{2} + (\mu_{Q}-\mu_{\Pi})^{2}}{2\sigma_{\Pi}^{2}} - \displaystyle\frac{1}{2}$ (s.t. Log-Normal Dist.)

</br>

## Requirements

- `torch` 2.5.1

- `cuda` 12.1

- `pandas` 2.2.3

- `numpy` 1.26.4
