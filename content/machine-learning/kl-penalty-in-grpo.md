---
title: "GRPO's KL penalty pulls toward the forward KL"
date: 2026-09-30
tags:
  - reinforcement-learning
  - llm
  - kl-divergence
---

# GRPO's KL penalty pulls toward the forward KL

RL fine-tuning of language models usually keeps the policy $\pi_\theta$ close to a reference model $\pi_{\mathrm{ref}}$ with a
KL penalty. The objective says *reverse* KL, $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$. But how the penalty is
implemented decides which gradient you actually get, and the popular choice in GRPO, differentiating the $k_3$ estimator
as a loss, follows the gradient of the *forward* KL instead.

This has been pointed out before (see [References](#references)). I found it surprising enough that I wanted to derive it
from scratch, check it numerically, and work out what $k_3$-as-a-loss is actually optimizing once you account for
sequences. The short answer: a per-position forward KL, evaluated on prefixes the policy generates. That looks a lot like
on-policy distillation toward the reference.

## Setup: three estimators of the same KL

At a single position, sample a token $a \sim \pi_\theta$ and let $r = \pi_{\mathrm{ref}}(a) / \pi_\theta(a)$. Since we
sample from $\pi_\theta$, $\mathbb{E}_{a \sim \pi_\theta}[r] = \sum_a \pi_{\mathrm{ref}}(a) = 1$. Schulman's
[note](http://joschu.net/blog/kl-approx.html) gives three per-sample estimators of
$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}) = \mathbb{E}_{a \sim \pi_\theta}[-\log r]$:

| Estimator | Formula | Unbiased for the KL value? | Always $\ge 0$? |
|---|---|---|---|
| $k_1$ | $-\log r$ | yes | no |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | no (low bias near $r = 1$) | yes |
| $k_3$ | $(r - 1) - \log r$ | yes, because $\mathbb{E}[r - 1] = 0$ | yes, because $\log x \le x - 1$ |

$k_3$ looks like the best of both worlds: unbiased, non-negative, low variance. GRPO
([DeepSeekMath](https://arxiv.org/abs/2402.03300)) adds $\beta\, k_3$ to the per-token loss.

But these are estimators of the KL's **value**. Training needs its **gradient**, and there are two ways to turn an estimator
into one.

## Two ways to use an estimator

- **In the reward.** Treat $k$ as a detached penalty on the reward and let the policy gradient carry it:
  the gradient is $\mathbb{E}\big[\,k \cdot \nabla_\theta \log \pi_\theta(a)\,\big]$ with $k$ held constant. This is what
  InstructGPT-style PPO does, where each token's reward gets $-\beta\,k_1$.
- **As a loss.** Add $k$ to the loss and backpropagate through it: the gradient is
  $\mathbb{E}\big[\,\nabla_\theta k\,\big]$, with the expectation still over $a \sim \pi_\theta$ but the sampling not
  differentiated. This is what GRPO does with $k_3$.

The KL's true gradient has two parts, because $\pi_\theta$ appears both in the sampling distribution and inside the
log-ratio. "As a loss" only differentiates the second part. Whether the result is still the right gradient depends on
the estimator.

## The per-position gradients

Write $\log r = \log \pi_{\mathrm{ref}}(a) - \log \pi_\theta(a)$, so $\nabla_\theta \log r = -\nabla_\theta \log \pi_\theta(a)$.

**$k_1$ in the reward.** $\mathbb{E}_{\pi_\theta}\big[(\log \pi_\theta - \log \pi_{\mathrm{ref}})\,\nabla \log \pi_\theta\big]$.
This is exactly $\nabla_\theta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$: the score-function form of the reverse-KL
gradient. (The other term, $\mathbb{E}[\nabla \log \pi_\theta] = 0$, vanishes.)

**$k_1$ as a loss.** $\mathbb{E}_{\pi_\theta}[\nabla(\log \pi_\theta - \log \pi_{\mathrm{ref}})] = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta] = 0$.
It does nothing in expectation.

**$k_2$ as a loss.** $\nabla \tfrac{1}{2}(\log r)^2 = \log r \cdot \nabla \log r = (\log \pi_\theta - \log \pi_{\mathrm{ref}})\,\nabla \log \pi_\theta$,
the same integrand as $k_1$ in the reward. So $k_2$ as a loss gives the **reverse**-KL gradient, an equivalence
[Liu et al. (2025)](https://arxiv.org/abs/2510.01555) prove in general for on-policy samples.

**$k_3$ as a loss.** With $\log r = x$, $k_3 = e^x - 1 - x$, so $\frac{d k_3}{dx} = e^x - 1 = r - 1$, and

$$
\nabla_\theta k_3 = (r - 1)\cdot \nabla_\theta \log r = (1 - r)\,\nabla_\theta \log \pi_\theta(a).
$$

Taking the expectation over $a \sim \pi_\theta$, the $\pi_\theta$ weights cancel against the $1/\pi_\theta$ inside $r$:

$$
\mathbb{E}_{a \sim \pi_\theta}\big[(1 - r)\,\nabla \log \pi_\theta(a)\big]
= \underbrace{\sum_a \pi_\theta(a)\,\nabla \log \pi_\theta(a)}_{=\,\nabla \sum_a \pi_\theta(a)\,=\,0}
\;-\; \sum_a \pi_{\mathrm{ref}}(a)\,\nabla \log \pi_\theta(a).
$$

The remaining term is exactly the gradient of the forward KL,

$$
\nabla_\theta\, \mathrm{KL}(\pi_{\mathrm{ref}} \,\|\, \pi_\theta)
= \nabla_\theta \Big[\sum_a \pi_{\mathrm{ref}}(a)\log \pi_{\mathrm{ref}}(a) - \sum_a \pi_{\mathrm{ref}}(a)\log \pi_\theta(a)\Big]
= -\sum_a \pi_{\mathrm{ref}}(a)\,\nabla_\theta \log \pi_\theta(a),
$$

because in the forward KL the weights are $\pi_{\mathrm{ref}}$, which don't depend on $\theta$.

> [!claim]
> At a single position, $k_1$ in the reward and $k_2$ as a loss follow the gradient of the reverse KL
> $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$. $k_3$ as a loss follows the gradient of the forward KL
> $\mathrm{KL}(\pi_{\mathrm{ref}} \,\|\, \pi_\theta)$, even though $k_3$ is an unbiased estimate of the reverse KL's value.

Near the reference the difference is small: with $x = \log r$, the $k_3$ coefficient is $1 - e^x = -x - x^2/2 - \dots$,
while $k_1$ and $k_2$ give exactly $-x$. They agree to first order, which is why
[Liu et al.](https://arxiv.org/abs/2510.01555) call $k_3$ as a loss "a first-order, biased approximation". They diverge once
the policy moves away from the reference, which is exactly when the regularizer should matter.

## Checking it numerically

On a small categorical distribution every expectation can be computed exactly, by summing over all tokens instead of
sampling:

```python
import torch

torch.manual_seed(0)
V = 6
theta = torch.randn(V, dtype=torch.float64, requires_grad=True)
logq = torch.log_softmax(torch.randn(V, dtype=torch.float64), 0)  # pi_ref, fixed
logp = torch.log_softmax(theta, 0)
p, q = logp.exp(), logq.exp()
w = p.detach()  # a ~ pi_theta; the sampling distribution is not differentiated

def grad(x):
    return torch.autograd.grad(x, theta, retain_graph=True)[0]

reverse = grad((p * (logp - logq)).sum())   # true grad of KL(pi || pi_ref)
forward = grad((q * (logq - logp)).sum())   # true grad of KL(pi_ref || pi)
log_r = logq - logp

k1_in_reward = grad((w * (logp - logq).detach() * logp).sum())
k1_as_loss = grad((w * -log_r).sum())
k2_as_loss = grad((w * 0.5 * log_r ** 2).sum())
k3_as_loss = grad((w * (log_r.exp() - 1 - log_r)).sum())

print(torch.allclose(k1_in_reward, reverse))  # True
print(torch.allclose(k1_as_loss, 0 * reverse))  # True
print(torch.allclose(k2_as_loss, reverse))  # True
print(torch.allclose(k3_as_loss, forward), torch.allclose(k3_as_loss, reverse))  # True False
```

## What $k_3$ as a loss optimizes over sequences

So far this is one position. A response is a sequence $y = (y_1, \dots, y_T)$, and the KL of sequences decomposes by the
chain rule, with the prefixes drawn from the **first** argument:

$$
\mathrm{KL}(P \,\|\, Q) = \sum_{t} \mathbb{E}_{y_{<t} \sim P}\Big[\mathrm{KL}\big(P(\cdot \mid y_{<t}) \,\|\, Q(\cdot \mid y_{<t})\big)\Big].
$$

- For the **reverse** KL, the prefixes come from $\pi_\theta$, the policy we sample from anyway. So summing per-token
  terms over our own rollouts gives the right **value**. The **gradient** is another matter: the prefix distribution
  itself depends on $\theta$. A per-token loss treats each prefix as fixed and misses that part. Putting $k_1$ in the
  reward and computing returns captures it, because each token's return includes the KL of everything after it.
  [Tang and Munos (2025)](https://arxiv.org/abs/2506.09477) call the per-token version "a partial gradient at best".
- For the **forward** KL, $\mathrm{KL}(P_{\mathrm{ref}} \,\|\, P_\theta)$, the chain rule wants prefixes from
  $\pi_{\mathrm{ref}}$. Its gradient is $-\mathbb{E}_{y \sim P_{\mathrm{ref}}}[\nabla \log P_\theta(y)]$: maximum likelihood on
  samples the reference generated.

$k_3$ as a loss matches neither. At each position it follows the forward KL of the next-token distributions, but it's
evaluated on prefixes the **policy** generated:

$$
\sum_t \mathbb{E}_{y_{<t} \sim \pi_\theta}\Big[\mathrm{KL}\big(\pi_{\mathrm{ref}}(\cdot \mid y_{<t}) \,\|\, \pi_\theta(\cdot \mid y_{<t})\big)\Big].
$$

This coincides with the sequence-level forward KL only when the two prefix distributions match, as at initialization, when
$\pi_\theta = \pi_{\mathrm{ref}}$.

One way to read it: the sequence-level forward KL is off-policy distillation (imitate the reference's own samples), while
$k_3$ as a loss is closer to **on-policy distillation** toward the reference, as in
[GKD](https://arxiv.org/abs/2306.13649): let the student generate, then match the teacher's next-token distribution at
every position it reached. For a regularizer that's arguably a reasonable thing to want, since it anchors the policy in
the states it actually visits.

## Does it matter?

Some consequences worth knowing:

- **Different regularizer, different behavior.** The forward KL is mass-covering: its gradient
  $-\sum_a \pi_{\mathrm{ref}}(a)\nabla \log \pi_\theta(a)$ pulls up every token the reference likes, in proportion to how much it
  likes it. The reverse KL is mode-seeking and lets the policy drop modes of the reference. So in expectation, $k_3$ as a
  loss resists entropy collapse more than the objective says it should.
- **The pull is noisy where it matters.** The correction for a token the policy has nearly abandoned only arrives when that
  token is sampled, which is rare, and then with a large weight $1 - r$, since $r = \pi_{\mathrm{ref}}/\pi_\theta$ is huge.
  Mass-covering in expectation, high variance in practice.
- **With a small $\beta$ it's mostly a mild anchor.** Many recent reasoning-RL recipes set $\beta = 0$ and sidestep the
  question entirely.
- **If you want the reverse KL,** use $k_1$ in the reward (with returns, for the sequential part) or $k_2$ as a loss
  (per position). If you keep $k_3$ as a loss, know that you're regularizing toward something closer to on-policy
  distillation from the reference.

The general lesson is older than GRPO: an unbiased estimator of a quantity's value, differentiated, is not an estimator of
its gradient. Before backpropagating through an estimator, check what its expected gradient actually is.

## References

- John Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html), 2020.
- Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024. (GRPO)
- Yunhao Tang and Rémi Munos, [On a few pitfalls in KL divergence gradient estimation for RL](https://arxiv.org/abs/2506.09477), 2025.
- Kezhao Liu, Jason Klein Liu, Mingtao Chen and Yiming Liu, [Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization](https://arxiv.org/abs/2510.01555), 2025.
- Xihuai Leo Wang, [Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html), 2025. States the forward-KL result for $k_3$ directly.
- Rishabh Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024. (GKD)
