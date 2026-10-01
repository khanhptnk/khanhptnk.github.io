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
KL penalty, and the objective says which one: the *reverse* KL between the two models' distributions over responses,
$\mathrm{KL}(P_\theta \,\|\, P_{\mathrm{ref}})$. But there are several common ways to implement that penalty, and they don't
all follow its gradient. This post works out exactly which objective each one optimizes:

*Table 1. What each implementation of the KL penalty optimizes, for responses $y$ sampled from the policy.
$\mathrm{KL}_t(p \Vert q)$ is the KL between the next-token distributions $p(\cdot \mid y_{<t})$ and $q(\cdot \mid y_{<t})$ at
prefix $y_{<t}$. "Prefixes frozen" means the expectation over prefixes $y_{<t} \sim P_\theta$ is held fixed when taking the
gradient: it ignores that changing an early token changes which later prefixes the policy visits.*

| Implementation | Follows the gradient of | In words |
|---|---|---|
| $k_1$ in the reward, with returns (PPO for RLHF) | $\mathrm{KL}(P_\theta \Vert P_{\mathrm{ref}})$ | the sequence-level reverse KL, as the objective says |
| $k_2$ as a per-token loss | $\sum_t \mathbb{E}_{y_{<t}}\, \mathrm{KL}_t(\pi_\theta \Vert \pi_{\mathrm{ref}})$ | a reverse KL at every position, prefixes frozen |
| $k_3$ as a per-token loss (GRPO) | $\sum_t \mathbb{E}_{y_{<t}}\, \mathrm{KL}_t(\pi_{\mathrm{ref}} \Vert \pi_\theta)$ | a **forward** KL at every position, prefixes frozen |
| $k_1$ as a per-token loss | $0$ | nothing, in expectation |

The single-position part of this, that $k_3$ as a loss follows the forward KL's gradient, has been pointed out before (see
[References](#references)). What this post adds:

1. **The sequence-level picture in Table 1**, checked exactly on a small model. It also makes precise how $k_1$ in the
   reward and $k_2$ as a loss differ: $k_2$ as a loss is exactly $k_1$ in the reward with each token charged only for its
   own KL, and the returns are what turn a per-position penalty into the sequence-level one.
2. **A reading of GRPO's penalty**: a forward KL on the policy's own prefixes is what
   [Zhao et al. (2026)](https://arxiv.org/abs/2605.16826) call DAgger-style on-policy SFT, here with the reference model as
   the expert.
3. **What it does in training**, on a toy problem with a known optimum: $k_3$ as a loss settles somewhere else, keeping
   extra entropy at the cost of reward when the reward dominates, and the reverse when the penalty dominates. One picture of
   its gradient explains both.

All code is at [github.com/khanhptnk/kl-penalty-in-grpo](https://github.com/khanhptnk/kl-penalty-in-grpo) and runs on a
CPU in a few minutes.

## Setup: three estimators of the same KL

Start with a single position. Sample a token $a \sim \pi_\theta$ and let $r = \pi_{\mathrm{ref}}(a) / \pi_\theta(a)$. Since
we sample from $\pi_\theta$, $\mathbb{E}_{a \sim \pi_\theta}[r] = \sum_a \pi_{\mathrm{ref}}(a) = 1$. Schulman's
[note](http://joschu.net/blog/kl-approx.html) gives three per-sample estimators of
$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}) = \mathbb{E}_{a \sim \pi_\theta}[-\log r]$:

*Table 2. Three per-sample estimators of $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$ from a token $a \sim \pi_\theta$, with $r = \pi_{\mathrm{ref}}(a)/\pi_\theta(a)$.*

| Estimator | Formula | Unbiased for the KL's value? | Always $\ge 0$? |
|---|---|---|---|
| $k_1$ | $-\log r$ | yes | no |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | no (low bias near $r = 1$) | yes |
| $k_3$ | $(r - 1) - \log r$ | yes, because $\mathbb{E}[r - 1] = 0$ | yes, because $\log x \le x - 1$ |

$k_3$ looks like the best of both worlds: unbiased, non-negative, low variance. GRPO
([DeepSeekMath](https://arxiv.org/abs/2402.03300)) adds $\beta\, k_3$ to the per-token loss.

But these estimate the KL's **value**. Training needs its **gradient**, and there are two ways to turn an estimator into one:

- **In the reward.** Treat $k$ as a detached penalty on the reward and let the policy gradient carry it. The gradient is
  $\mathbb{E}\big[\,k \cdot \nabla_\theta \log \pi_\theta(a)\,\big]$ with $k$ held constant. InstructGPT-style PPO does this:
  each token's reward gets $-\beta\,k_1$, and the advantage estimator (returns or GAE) passes it on to earlier tokens.
- **As a loss.** Add $k$ to the loss and backpropagate through it. The gradient is $\mathbb{E}\big[\,\nabla_\theta k\,\big]$:
  the expectation is still over $a \sim \pi_\theta$, but the sampling isn't differentiated. GRPO does this with $k_3$.

The KL's true gradient has two parts, because $\pi_\theta$ appears both in the sampling distribution and inside the
log-ratio. "As a loss" only differentiates the second part. Whether that's still the right gradient depends on the estimator.

## One position

Write $\log r = \log \pi_{\mathrm{ref}}(a) - \log \pi_\theta(a)$, so $\nabla_\theta \log r = -\nabla_\theta \log \pi_\theta(a)$.

**$k_1$ in the reward.** $\mathbb{E}_{\pi_\theta}\big[(\log \pi_\theta - \log \pi_{\mathrm{ref}})\,\nabla \log \pi_\theta\big]$.
This is exactly $\nabla_\theta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$, the score-function form of the reverse-KL
gradient. (The other term, $\mathbb{E}[\nabla \log \pi_\theta] = 0$, vanishes.)

**$k_1$ as a loss.** $\mathbb{E}_{\pi_\theta}[\nabla(\log \pi_\theta - \log \pi_{\mathrm{ref}})] = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta] = 0$.
It does nothing in expectation.

**$k_2$ as a loss.** $\nabla \tfrac{1}{2}(\log r)^2 = \log r \cdot \nabla \log r = (\log \pi_\theta - \log \pi_{\mathrm{ref}})\,\nabla \log \pi_\theta$.
That isn't just the same in expectation as $k_1$ in the reward: it is the same **sample by sample**. At a single position,
$k_2$ as a loss and $k_1$ in the reward are the same estimator, both of the reverse-KL gradient
([Liu et al., 2025](https://arxiv.org/abs/2510.01555) prove the equivalence in general for on-policy samples).

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

On a small categorical distribution every expectation can be computed exactly, by summing over the vocabulary instead of
sampling, so this checks as an identity rather than a statistical test:

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

### The shape of the pull

Both "as a loss" gradients have the form $c \cdot \nabla \log \pi_\theta(a)$ for the sampled token, so they differ only in
the coefficient $c$: $\log \pi_\theta(a) - \log \pi_{\mathrm{ref}}(a)$ for the reverse KL, and $1 - r$ for $k_3$. With
$x = \log r$, $1 - e^x = -x - x^2/2 - \dots$, so the two agree to first order near the reference, which is why
[Liu et al.](https://arxiv.org/abs/2510.01555) call $k_3$ as a loss "a first-order, biased approximation". Away from the
reference they are very different:

<figure>
<img class="theme-light" src="../assets/kl-penalty-in-grpo/kl-coefficient-light.svg" alt="The per-token gradient coefficient as a function of pi over pi_ref, for the reverse-KL implementations (a straight line in log pi over pi_ref) and for k3 as a loss (capped at 1 above, falling exponentially below).">
<img class="theme-dark" src="../assets/kl-penalty-in-grpo/kl-coefficient-dark.svg" alt="The per-token gradient coefficient as a function of pi over pi_ref, for the reverse-KL implementations (a straight line in log pi over pi_ref) and for k3 as a loss (capped at 1 above, falling exponentially below).">
<figcaption>Figure 1. The gradient coefficient c at the sampled token, as a function of how much more (right) or less (left) the policy likes that token than the reference does. A positive c lowers the token's probability, a negative c raises it.</figcaption>
</figure>

$k_3$'s pull is **asymmetric**. On a token the policy under-weights ($\pi_\theta \ll \pi_{\mathrm{ref}}$), it pushes up
exponentially hard. On a token the policy over-weights, it pushes down gently, never with a coefficient above 1. That is
the forward KL's mass-covering behavior seen token by token, and it predicts both halves of the experiment below.

## Whole sequences

A response is a sequence $y = (y_1, \dots, y_T)$, and the KL between sequence distributions decomposes by the chain rule,
with the prefixes drawn from the **first** argument:

$$
\mathrm{KL}(P \,\|\, Q) = \sum_{t} \mathbb{E}_{y_{<t} \sim P}\Big[\mathrm{KL}\big(P(\cdot \mid y_{<t}) \,\|\, Q(\cdot \mid y_{<t})\big)\Big].
$$

For the reverse KL the prefixes come from $P_\theta$, the policy we sample from anyway, so summing per-token terms over our
own rollouts gives the right **value**. The **gradient** has one more part, because the prefix distribution itself depends on
$\theta$: changing the probability of an early token changes which later prefixes the policy visits, and so how much KL it
accumulates there.

- **$k_1$ in the reward, with returns, captures that part.** Each token's return includes the KL of every token after it,
  so an early token is credited (or blamed) for the KL it leads to. Its expected gradient is exactly the sequence-level
  reverse-KL gradient.
- **A per-token loss can't.** It treats each prefix as given and only adjusts the next-token distribution at that prefix.
  So $k_2$ as a loss follows a *per-position* reverse KL with the prefixes frozen, the second row of Table 1. That is
  exactly $k_1$ in the reward with each token charged only for its own KL: dropping the returns is the whole difference.
  [Tang and Munos (2025)](https://arxiv.org/abs/2506.09477) call this "a partial gradient at best".
- **$k_3$ as a loss** is the same per-position, prefixes-frozen construction, with the forward KL at each position instead
  of the reverse one, the third row of Table 1.

In practice there's a second difference between the first two: $k_1$ in the reward travels through the advantage pipeline
(baselines, GAE, advantage normalization, PPO clipping), so its effective strength depends on those. A KL loss term is
added separately, so $\beta$ keeps its meaning.

**Checking it exactly.** A tabular autoregressive policy with 3 tokens and length 3 has 27 sequences, so every expectation
above is a finite sum. For policies at five distances from the reference, every row of Table 1 holds as an exact identity,
and so does "$k_2$ as a loss equals $k_1$ in the reward without returns". The per-position versions aren't small
corrections: one unit of logit noise away from the reference, $k_3$ as a loss is 26% off the sequence-level forward-KL
gradient and $k_2$ as a loss is 31% off the sequence-level reverse-KL gradient.

**Reading GRPO's penalty.** [Zhao et al. (2026)](https://arxiv.org/abs/2605.16826) classify distillation objectives by two
independent choices: the KL direction, and whose prefixes the supervision is applied on. Forward KL on the student's own
prefixes is, in their words, "DAgger-style on-policy SFT": the student visits states, and the teacher supplies next-token
targets there. So GRPO's KL penalty, implemented as $k_3$ as a loss, is DAgger-style on-policy SFT toward the reference
model. (What is usually called on-policy distillation uses the reverse KL on the student's prefixes, the other cell of the
same table; [GKD](https://arxiv.org/abs/2306.13649) covers both.) As a regularizer that's arguably reasonable, since it
anchors the policy in the states it actually visits. But it is not the regularizer the objective writes down.

## What it does in training

The derivations say what each implementation's gradient *is*. Two toy experiments show what it *does*. Both are small
enough that every quantity is computed exactly, so they show direction and mechanism, not magnitudes for LLM training
(see [Limitations](#limitations)).

**Where training settles.** The same 27-sequence policy, a reward of 1 for one sequence and 0 for the rest, trained from
the reference with REINFORCE (a batch-mean baseline, 64 sampled sequences per step, 10 seeds) and the KL penalty added in
each of the three ways. The objective $\mathbb{E}[R] - \beta\,\mathrm{KL}(P_\theta \,\|\, P_{\mathrm{ref}})$ has a closed-form
optimum, $P^* \propto P_{\mathrm{ref}}\, e^{R/\beta}$, so we know where a correct implementation should end up.

<figure>
<img class="theme-light" src="../assets/kl-penalty-in-grpo/kl-entropy-light.svg" alt="Two panels against the KL coefficient beta. Left, entropy: k1 in the reward follows the optimum; k3 as a loss is far above it for beta up to 0.2 and slightly below it at 0.5. Right, probability of the rewarded sequence: k1 follows the optimum; k3 is below it for small beta and above it at 0.5.">
<img class="theme-dark" src="../assets/kl-penalty-in-grpo/kl-entropy-dark.svg" alt="Two panels against the KL coefficient beta. Left, entropy: k1 in the reward follows the optimum; k3 as a loss is far above it for beta up to 0.2 and slightly below it at 0.5. Right, probability of the rewarded sequence: k1 follows the optimum; k3 is below it for small beta and above it at 0.5.">
<figcaption>Figure 2. Where training settles after 1,500 steps, for each implementation of the penalty and each β (mean and spread over 10 seeds). The dotted line is the exact optimum of the KL-regularized objective.</figcaption>
</figure>

- **$k_1$ in the reward lands on the optimum** at every $\beta$ (within 0.06 nats of entropy and 0.01 of reward
  probability, the slack of finite training), as Table 1 says it should.
- **$k_2$ as a loss under-regularizes**: it ends up with more reward than the optimum (0.92 against 0.88 at
  $\beta = 0.2$), because the per-position penalty misses the KL that early tokens cause later on.
- **$k_3$ as a loss misses in both directions.** When the reward dominates ($\beta \le 0.2$), it keeps far more entropy
  and gives up reward (0.73 against 0.88 at $\beta = 0.2$). When the penalty dominates ($\beta = 0.5$), it does the opposite:
  more reward (0.40 against 0.27), less entropy. Figure 1 explains both. Far from the reference, the exponential push on
  under-weighted sequences keeps them alive. Near it, the capped push against the over-weighted, rewarded sequence lets
  that sequence grow more than the reverse KL would allow.

So "$k_3$ as a loss keeps more entropy" is only true in one regime. The accurate statement is that it is a different
regularizer, with an asymmetric pull.

**How the pull arrives.** Figure 1's exponential side has a catch: it only acts on a token when that token is sampled.
Take a policy that has nearly dropped a token the reference likes (probability $1.5 \times 10^{-4}$ against 0.2) and train
with the KL term alone:

<figure>
<img class="theme-light" src="../assets/kl-penalty-in-grpo/kl-recovery-light.svg" alt="Probability of the dropped token over 300 steps, log scale. The exact forward-KL gradient recovers it smoothly to 0.2 by step 100. Sampled k3 runs stay flat, then jump to nearly 1 in a single step when the token is first sampled, and decay back to 0.2. Sampled k2 runs stay near 0.0002.">
<img class="theme-dark" src="../assets/kl-penalty-in-grpo/kl-recovery-dark.svg" alt="Probability of the dropped token over 300 steps, log scale. The exact forward-KL gradient recovers it smoothly to 0.2 by step 100. Sampled k3 runs stay flat, then jump to nearly 1 in a single step when the token is first sampled, and decay back to 0.2. Sampled k2 runs stay near 0.0002.">
<figcaption>Figure 3. Probability of a token the policy had nearly dropped, under the KL term alone (5 of 20 seeds per method, 64 sampled tokens per step).</figcaption>
</figure>

With $k_2$, the token never comes back: the reverse KL barely cares about mass the policy doesn't put somewhere. With
$k_3$, it comes back in 19 of 20 runs, but not the way the exact forward-KL gradient does it. Nothing happens until the
token is sampled (here about once per 100 steps), and then a single update with weight $1 - r \approx -1300$ throws its
probability to nearly 1 before it settles. The mass-covering pull is real, but it arrives as rare, violent updates, the same
heavy tail behind the statistical instability of $k_3$ that [Liu et al.](https://arxiv.org/abs/2510.01555) analyze in their
Appendix I.

## What to take away

- **If you want the reverse KL the objective writes down,** use $k_1$ in the reward with returns. $k_2$ as a loss gives
  the per-position part only.
- **If you keep $k_3$ as a loss,** you are doing DAgger-style on-policy SFT toward the reference: an asymmetric pull that
  fights dropped modes hard and over-weighted ones gently, with the hard part arriving as rare large updates.
- **Many recent reasoning-RL recipes set $\beta = 0$** (e.g. [DAPO](https://arxiv.org/abs/2503.14476)) and sidestep the
  question entirely.

The general lesson is older than GRPO: an unbiased estimator of a quantity's value, differentiated, is not an estimator of
its gradient. Before backpropagating through an estimator, check what its expected gradient actually is.

## Limitations

The toy problems were chosen so that every quantity can be computed exactly, which is what makes the identities airtight.
They establish what each implementation optimizes, and they show the mechanisms. They don't measure how much the
difference matters in LLM training, where $\beta$ is small relative to the advantages, the vocabulary is large, the
policy-gradient term dominates, and gradient clipping blunts exactly the large updates in Figure 3. An RL run on a real
model comparing $k_3$ as a loss with $k_2$ as a loss at equal $\beta$ would settle that.

## References

- John Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html), 2020.
- Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024. (GRPO)
- Qiying Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476), 2025.
- Yunhao Tang and Rémi Munos, [On a few pitfalls in KL divergence gradient estimation for RL](https://arxiv.org/abs/2506.09477), 2025.
- Kezhao Liu, Jason Klein Liu, Mingtao Chen and Yiming Liu, [Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization](https://arxiv.org/abs/2510.01555), 2025.
- Xihuai Leo Wang, [Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html), 2025. States the forward-KL result for $k_3$ directly.
- Anhao Zhao, Haoran Xin, Yingqi Fan, Junlong Tong, Wenjie Li and Xiaoyu Shen, [Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation](https://arxiv.org/abs/2605.16826), 2026.
- Rishabh Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024. (GKD)
- Code: [github.com/khanhptnk/kl-penalty-in-grpo](https://github.com/khanhptnk/kl-penalty-in-grpo).
