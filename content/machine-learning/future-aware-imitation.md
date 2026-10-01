---
title: "When immediate imitation is not enough"
date: 2026-10-01
tags:
  - imitation-learning
  - reinforcement-learning
---

# When immediate imitation is not enough

[DAgger](https://arxiv.org/abs/1011.0686) fixes the classic failure of behavior cloning by training on the states the
learner actually visits: roll out the learner, ask the expert what it would do at each visited observation, and fit
those labels with supervised learning. But the target it fits is still *local*: at each observation, put probability on
the expert's action now.

That is the right target when the learner can imitate the expert perfectly. Often it can't: the expert sees something
the learner doesn't (privileged state, a hidden goal), the model is too small, or the task is noisy. Then some mistakes
are unavoidable, and the question becomes **which** mistakes to make. A local imitation loss charges every mistake the
same. A long-horizon objective can tell an unavoidable mistake that leaves the future imitable apart from one that
doesn't.

This post builds the smallest example I could find where that difference is all there is:

1. **A toy POMDP where DAgger is provably indifferent.** Two root actions match the expert equally often, but only one
   reveals the information the learner needs to imitate the expert afterwards. DAgger's root target is exactly
   $\tfrac{1}{2}$ for each, and so is the target of AggreVaTe, which looks ahead using the expert's cost-to-go.
2. **RL on the same imitation signal breaks the tie.** PPO and a GRPO-style update, rewarded only for agreeing with the
   expert at each step, learn to take the revealing action (with probability 0.99 and 0.97). That raises an independent
   success metric from 54% to 96%.
3. **The reason is whose future gets scored.** The tie breaks only when an action is credited with the agreement the
   *learner* goes on to achieve, not the agreement the expert would.
4. **The same happens in distillation into a smaller student.** When the teacher's own path leads to behavior the
   student is too small to imitate, DAgger and the per-token distillation objectives (on-policy forward KL, and reverse KL
   with a discount of zero) follow the teacher anyway. Objectives with returns, including the sequence-level reverse KL,
   learn to leave the teacher's path at the first step.

The phenomenon isn't new: it is the *imitation gap* of learning from a privileged expert
([Weihs et al., 2021](https://arxiv.org/abs/2007.12173)). Neither is the idea of fixing it with returns: crediting a
learner with the expert agreement it goes on to achieve has close relatives in structured prediction, privileged-expert RL
and LLM distillation (see [Related work](#related-work)). I haven't found this exact algorithm, PPO on a $\pm 1$
agreement reward with returns, studied as an imitation method, but small differences in the reward or the optimizer can
change behavior a lot, so the closest ones are listed there with how they differ. What the toy adds is a construction
clean enough to derive DAgger's behavior exactly and check it against the runs, and to show that looking ahead with the
expert's cost-to-go doesn't break the tie. A distillation version shows which of today's on-policy distillation
objectives can leave a teacher's path that the student can't follow. All code is at
[github.com/khanhptnk/future-aware-imitation](https://github.com/khanhptnk/future-aware-imitation) and runs on a CPU in
a few minutes.

## The environment

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/env-light.svg" alt="Diagram: a root observation that hides z. Action 1 leads to 8 downstream steps where z is visible; action 0 leads to 8 downstream steps where z is still hidden. Either root action matches the expert with probability one half, and the expert plays z at every step.">
<img class="theme-dark" src="../assets/future-aware-imitation/env-dark.svg" alt="Diagram: a root observation that hides z. Action 1 leads to 8 downstream steps where z is visible; action 0 leads to 8 downstream steps where z is still hidden. Either root action matches the expert with probability one half, and the expert plays z at every step.">
<figcaption>Figure 1. The information-reveal environment. The root action is a coin flip as far as immediate imitation goes, but only action 1 makes the rest of the episode imitable.</figcaption>
</figure>

Each episode samples a hidden bit $z \sim \mathrm{Bernoulli}(0.5)$. The expert sees $z$ and plays $a^*_t = z$ at every step.
The learner doesn't see $z$. It makes 9 binary decisions:

- **At the root**, its observation hides $z$. Whatever it does, it matches the expert with probability $\tfrac{1}{2}$.
- **Downstream** (8 more steps), its observation depends only on the root action. After action 1 it reveals $z$
  (two observations, one per value of $z$). After action 0 it is a single observation that still hides $z$.

The policy is tabular: one Bernoulli parameter per observation, four in total. Nothing learned downstream can generalize
back to the root through shared parameters, so the root decision is decided by the training signal alone.

The training signal for RL is the same thing DAgger imitates, expert agreement, just summed over the episode. Each step
earns $r_t = +1$ if $a_t = a^*_t$ and $r_t = -1$ otherwise, and RL maximizes

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_{t=0}^{8} r_t\Big].
$$

So nothing here comes from a task reward the imitation learner doesn't see. The two approaches differ only in how they
assign credit for the same per-step labels.

## Why DAgger is exactly indifferent

At the root observation the expert's label is $z$, which is 0 half the time and 1 half the time, whatever the learner did
to get there (it's the first step). DAgger's cross-entropy at the root is therefore

$$
\mathcal{L}_{\mathrm{root}}(\pi) = -\tfrac{1}{2}\log \pi(0 \mid o_{\mathrm{root}}) - \tfrac{1}{2}\log \pi(1 \mid o_{\mathrm{root}}),
$$

minimized at $\pi(1 \mid o_{\mathrm{root}}) = \tfrac{1}{2}$. DAgger learns the downstream observations perfectly: after
action 1 it reads $z$ off the observation and copies it, and after action 0 its labels are again a 50/50 mix. But none of
that reaches the root. Every episode, DAgger flips a fair coin between a future it can imitate and one it can't.

> [!claim]
> When the learner can't infer the expert's action, the local imitation target at an observation is the expert's action
> distribution given what the learner sees there. It contains no information about which action leads to observations
> where the expert *can* be inferred.

**Looking ahead with the expert doesn't help either.** [AggreVaTe](https://arxiv.org/abs/1406.5979) replaces the label
with a cost-to-go: at a visited observation, try an action, then let the **expert** finish the episode, and prefer the
action with the better total. Here the expert sees $z$, so it agrees with itself at every remaining step no matter which
root action came first. The cost-to-go is $\pm 1$ for the root step plus $8$ afterwards for either action, and averaged
over $z$ the two actions tie at 8. Rolling out the expert assumes the expert will be around to recover. The learner
won't be: after action 0 it has no way to know $z$.

**Rolling out the learner breaks the tie.** Score each root action by the agreement the learner itself goes on to
achieve: up to $8$ after action 1 (it can copy $z$) and $0$ in expectation after action 0 (it can do no better than a
coin flip at each of 8 steps). The gap is
the entire downstream episode. That is what Monte Carlo returns give PPO and GRPO.

## Results

Each method trains from a uniform policy with 12 seeds. DAgger runs 40 iterations of 3,000 learner roll-outs and fits
the aggregated labels exactly. PPO and a minimal GRPO-style update optimize $J$ with the clipped objective. Each final
policy is evaluated on 50,000 fresh episodes. Besides the training signal, I report a separate **task success**
metric, at most 2 disagreements in the 9-step episode, which no method optimizes.

*Table 1. Information-reveal environment: mean ± standard deviation over 12 seeds. The best possible policy (always
reveal, then copy $z$) scores a return of 8, 0.5 errors and 100% success.*

| Method | P(revealing action) | Signed return | Errors / episode | Task success |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.002 | 4.00 ± 0.02 | 2.50 ± 0.01 | 54.5% ± 0.2% |
| PPO | 0.986 ± 0.000 | 7.19 ± 0.01 | 0.90 ± 0.00 | 96.2% ± 0.1% |
| GRPO | 0.967 ± 0.001 | 7.01 ± 0.01 | 0.99 ± 0.01 | 96.0% ± 0.1% |

DAgger lands exactly where the analysis puts it. Half its episodes take the revealing branch and succeed. The other
half are 9 coin flips, which succeed with probability $P(\mathrm{Bin}(9, \tfrac{1}{2}) \le 2) = 46/512$, so its success
rate is $\tfrac{1}{2} + \tfrac{1}{2} \cdot \tfrac{46}{512} = 54.49\%$. PPO and GRPO learn to reveal:

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/root-light.svg" alt="Two panels showing the probability of root action 1 against training episodes. In both environments, DAgger stays flat at 0.5 while PPO and GRPO rise steadily: to 0.99 and 0.97 in the information-reveal environment, and to 0.97 and 0.90 in the hard-expert environment.">
<img class="theme-dark" src="../assets/future-aware-imitation/root-dark.svg" alt="Two panels showing the probability of root action 1 against training episodes. In both environments, DAgger stays flat at 0.5 while PPO and GRPO rise steadily: to 0.99 and 0.97 in the information-reveal environment, and to 0.97 and 0.90 in the hard-expert environment.">
<figcaption>Figure 2. The root decision during training (mean and range over 12 seeds; the x-axis counts episodes, so methods with smaller batches end earlier). DAgger's root never moves from 1/2.</figcaption>
</figure>

The success metric moves much more than the average error count, because the two root choices give very different
error **distributions**:

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/errors-light.svg" alt="Bar chart of the number of expert disagreements per episode. DAgger is bimodal: about half its episodes have 0 or 1 errors, the other half spread from 2 to 9 around 4 or 5. PPO and GRPO put almost all episodes at 0, 1 or 2 errors.">
<img class="theme-dark" src="../assets/future-aware-imitation/errors-dark.svg" alt="Bar chart of the number of expert disagreements per episode. DAgger is bimodal: about half its episodes have 0 or 1 errors, the other half spread from 2 to 9 around 4 or 5. PPO and GRPO put almost all episodes at 0, 1 or 2 errors.">
<figcaption>Figure 3. Errors per episode in the information-reveal environment, computed exactly from each trained policy (averaged over seeds). DAgger's hidden-branch episodes form the second hump.</figcaption>
</figure>

**What RL pays for it.** RL isn't better everywhere. In the revealing branch, DAgger copies $z$ perfectly from the
first few labels. PPO, after 370,000 episodes, still disagrees with the expert at 4.4% of those steps. That is 0.35 of
its 0.9 errors per episode (0.5 is the unavoidable root coin flip), on steps that supervised learning gets right for free. The two methods are good at different
parts of the problem. Supervised labels are the efficient way to learn what the expert does where that can be inferred.
The return is the only one of the two signals that says which unavoidable mistake to make.

## A second variant: a hard-to-imitate expert

The information-reveal environment gets its asymmetry from what the learner can observe, with an expert rule that never
changes. A more artificial variant puts the asymmetry in the expert instead. The root is the same. A correct root action
enters an easy corridor where the expert always plays the same action. The two possible mistakes lead to different
places: "1 when $z = 0$" enters a recoverable corridor that is just as easy, and "0 when $z = 1$" enters a hard corridor
where the expert flips a fresh coin at every step. Again both root actions match the expert half the time, but only
action 1 never risks the hard corridor.

*Table 2. Hard-expert environment, 12 seeds. P(root action 1) is the probability of choosing the side whose mistake is
recoverable.*

| Method | P(root action 1) | Signed return | Errors / episode | Task success |
|---|---|---|---|---|
| DAgger | 0.500 ± 0.001 | 6.00 ± 0.02 | 1.50 ± 0.01 | 75.9% ± 0.3% |
| PPO | 0.970 ± 0.001 | 7.18 ± 0.01 | 0.91 ± 0.00 | 96.1% ± 0.1% |
| GRPO | 0.902 ± 0.003 | 6.83 ± 0.02 | 1.09 ± 0.01 | 92.9% ± 0.2% |

Same picture: DAgger sits at $\tfrac{1}{2}$ (its exact success rate is $75.88\%$), and both RL methods learn to make the
recoverable mistake.

## Distillation into a smaller student

In both environments so far, the learner fails because it can't see what the expert sees. The other common reason is
capacity. In distillation, the student sees exactly what the teacher sees and is simply smaller. The same question
applies: when the student can't follow the teacher, does it learn to go somewhere it can?

**Setup.** No information is hidden. The episode is a root decision followed by 8 steps in one of two branches, chosen by
the root action, and the student observes the branch and the step $t$. The teacher plays action 0 at the root, so its
own path is branch 0, where it plays the parity of $t$ ($1, 0, 1, 0, \dots$). In branch 1 it always plays 0. The student
has its own logit at the root and, in each branch, a logistic policy whose logit is a polynomial of degree $k$ in $t$.
Degree 7 is enough to fit the parity of 8 steps, so a student with $k = 7$ can imitate the teacher perfectly. A
polynomial of degree $k$ changes sign at most $k$ times, so on the teacher's path a smaller student makes at least
$\lceil (7 - k)/2 \rceil$ errors. Leaving the teacher's path at the root costs exactly one error, and branch 1 is easy at
any size.

**A deterministic teacher.** DAgger's label at the root is always 0, so it always follows the teacher; this is exact, at
every student size. AggreVaTe follows too: the teacher's cost-to-go assumes the teacher finishes the episode, and the
teacher imitates itself perfectly in either branch, so copying the root action wins. PPO and GRPO are trained on the
same $\pm 1$ agreement reward as before:

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/distill-light.svg" alt="Disagreements per episode against the student's polynomial degree from 0 to 7. DAgger falls from 4 errors at degree 0 to 2.5 at degree 6, always above the fewest errors possible when copying the teacher, and reaches 0 at degree 7. PPO and GRPO stay at about 1 error at every degree, the cost of leaving the teacher's path.">
<img class="theme-dark" src="../assets/future-aware-imitation/distill-dark.svg" alt="Disagreements per episode against the student's polynomial degree from 0 to 7. DAgger falls from 4 errors at degree 0 to 2.5 at degree 6, always above the fewest errors possible when copying the teacher, and reaches 0 at degree 7. PPO and GRPO stay at about 1 error at every degree, the cost of leaving the teacher's path.">
<figcaption>Figure 4. Distillation from a deterministic teacher into students of increasing size (mean over 12 seeds; seed-to-seed standard deviations are at most 0.01). The dotted steps are the fewest errors a student of that size can make on the teacher's path.</figcaption>
</figure>

- **Below degree 7, DAgger pays for following the teacher:** 2.5 to 4 errors per episode. That is more than the fewest
  possible on the teacher's path, because maximum likelihood on labels the student can't fit gives soft probabilities,
  not the policy with the fewest errors. (Degrees pair up, 1 with 2 and so on, because the teacher's labels at $t$ and
  $9 - t$ are opposite, so even-degree terms don't help.)
- **PPO and GRPO leave the teacher's path** in 98% to 99.8% of episodes and make about one error per episode, at every
  size below 7.
- **At degree 7 the roles reverse.** DAgger imitates perfectly. PPO still leaves the teacher's path in 90% of episodes
  (0.95 errors). The other branch is easy to learn, so RL commits to it early, and from then on it rarely visits the
  teacher's path, so it gets little signal to return. When imitation is realizable, supervised labels win.

**A stochastic teacher.** Language-model teachers are distributions, and the standard distillation objectives compare
distributions. So make the teacher put probability 0.9 on the action above in every state, including the root, and 0.1
on the other. Then compare four ways of training the student on its own roll-outs, with the teacher queried at every
state the student visits:

- **On-policy forward KL:** cross-entropy to the teacher's distribution at the student's states, which is DAgger with
  soft labels (GKD's forward KL, and the "DAgger-style on-policy SFT" of my
  [previous post](./kl-penalty-in-grpo)).
- **Reverse KL with a discount of zero:** a per-token reward $\log \pi_T(a \mid s) - \log \pi_S(a \mid s)$, with each token
  credited only with its own reward, as in
  [Thinking Machines' on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- **Reverse KL with returns:** the same reward with undiscounted returns, which follows the gradient of the
  sequence-level reverse KL $\mathrm{KL}(P_S \,\|\, P_T)$ (the first row of Table 1 in the previous post).
- **$\pm 1$ agreement:** PPO with $+1$ for agreeing with the teacher's preferred action and $-1$ otherwise.

The RL variants share the update used throughout. An episode has only $2 \times 2^8 = 512$ action sequences, so every
metric is computed exactly by summing over all of them.

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/distill-soft-light.svg" alt="Two panels against the student's degree. Left, the probability of leaving the teacher's path: on-policy forward KL and reverse KL with discount zero stay at the teacher's 0.1 at every degree; reverse KL with returns leaves with probability 0.87 at degree 0, falling to 0.56 at degree 6 and 0.2 at degree 7; plus-minus-one agreement leaves with probability near 1. Right, disagreements per episode: the two per-token objectives fall from 3.8 to 2.6, then to about 1 at degree 7; reverse KL with returns stays near 2.1 to 2.2, dropping to 1.7 at degree 7; plus-minus-one agreement stays near 1.">
<img class="theme-dark" src="../assets/future-aware-imitation/distill-soft-dark.svg" alt="Two panels against the student's degree. Left, the probability of leaving the teacher's path: on-policy forward KL and reverse KL with discount zero stay at the teacher's 0.1 at every degree; reverse KL with returns leaves with probability 0.87 at degree 0, falling to 0.56 at degree 6 and 0.2 at degree 7; plus-minus-one agreement leaves with probability near 1. Right, disagreements per episode: the two per-token objectives fall from 3.8 to 2.6, then to about 1 at degree 7; reverse KL with returns stays near 2.1 to 2.2, dropping to 1.7 at degree 7; plus-minus-one agreement stays near 1.">
<figcaption>Figure 5. Distillation from a stochastic teacher (mean over 12 seeds). Blue objectives credit each token only with its own score; orange ones use returns. Disagreements are counted against the teacher's preferred action; the teacher itself averages 0.9.</figcaption>
</figure>

*Table 3. Stochastic teacher, student of degree 1 (12 seeds, exact evaluation). "Leave" is the probability of leaving
the teacher's path at the root. Reverse KL is $\mathrm{KL}(P_S \Vert P_T)$ and forward KL is $\mathrm{KL}(P_T \Vert P_S)$,
both over whole episodes, in nats.*

| Objective | Leave | Errors | Success | Rev. KL | Fwd. KL |
|---|---|---|---|---|---|
| Teacher | 0.10 | 0.90 | 94.7% | 0 | 0 |
| Forward KL | 0.100 | 3.64 | 23.1% | 3.49 | 2.54 |
| Reverse KL, γ = 0 | 0.103 | 3.61 | 23.7% | 3.47 | 2.55 |
| Reverse KL, returns | 0.840 | 2.12 | 71.1% | 2.13 | 3.90 |
| ±1 agreement | 0.998 | 1.01 | 99.8% | 3.11 | 14.2 ± 1.8 |

- **Per-token objectives follow the teacher at every student size.** Both leave the teacher's path with the teacher's own
  probability, 0.1. With the root's own parameter, each one's target at the root is to match the teacher there,
  whatever comes after, so the root decision can't see what the student can do downstream.
- **Objectives with returns leave.** Where the optimum of the sequence-level reverse KL can be worked out by hand, reverse KL
  with returns reaches it. For $k = 0$, the student's best policy on the teacher's path is a coin flip, which costs
  $8\,\mathrm{KL}(\mathrm{Bern}(0.5) \,\|\, \mathrm{Bern}(0.9)) = 4.087$ nats. The optimum then leaves with probability
  $0.1 / (0.1 + 0.9\,e^{-4.087}) = 0.869$, at a KL of 2.162 nats; training reaches 0.866 and 2.162.
- **The two optimize different things.** $\pm 1$ agreement goes after the teacher's preferred action and collapses to a
  nearly deterministic student: almost no errors, but far from the teacher's distribution (a forward KL of 14 nats,
  with large variation across seeds). Reverse KL with returns stays a distribution and trades errors for KL; it has the
  lowest reverse KL of the four, which is its objective.
- **At degree 7 the per-token objectives win again.** On-policy forward KL matches the teacher exactly, and reverse KL with
  a discount of zero nearly does (0.05 nats). Reverse KL with returns hasn't converged after the same budget (0.54 nats).

## Related work

- **The objective.** [Ross et al. (2011)](https://arxiv.org/abs/1011.0686) define imitation's goal as minimizing the
  expected per-step loss under the learner's own state distribution. With 0-1 loss over 9 steps, that is the expected
  number of disagreements, an affine function of $J$ here. DAgger gets there through a reduction to no-regret online
  learning, which fits each iteration's data as if that distribution were fixed. Its guarantee is good when some policy
  in the class imitates the expert well on that data, and weak here, where none can.
  [Czarnecki et al. (2019)](https://arxiv.org/abs/1902.02186) make the general version of this point for policy
  distillation: on-policy distillation updates are in general not the gradient of any objective, and adding the future
  distillation loss as a reward recovers the gradient of the cumulative loss.
- **The closest algorithms.** None of these is PPO on a $\pm 1$ agreement reward with returns, and each difference can
  matter.
  - [Walsman et al. (2023)](https://openreview.net/forum?id=sciA_xgYofB) study experts with privileged information and
    include an expert-matching reward trained with PPO as a baseline: $\pm 0.1$ per step added to the task reward, with
    a learned critic and $\gamma = 0.99$. They observe that it can learn loops that keep collecting agreement reward,
    which variable-length episodes allow and the fixed horizon here rules out. Their implementation has an entropy bonus
    and doesn't normalize advantages, so $\pm 0.1$ there is not equivalent to $\pm 1$.
  - [Maes et al. (2009)](https://link.springer.com/article/10.1007/s10994-009-5140-8) cast structured prediction as RL
    with a per-decision reward of 1 for a correct label and 0 otherwise, trained with policy-gradient and SARSA methods.
  - [LOLS](https://arxiv.org/abs/1502.02206) with learned roll-outs scores each action by rolling out the learner and
    counting disagreements with the reference, then trains a cost-sensitive classifier rather than a policy gradient.
  - Several methods use a soft agreement score instead of $\pm 1$: similarity to a fully observable teacher
    ([COSIL](https://arxiv.org/abs/2211.01991)), or the teacher's log-probabilities in LLM distillation, optimized with
    returns ([MiniLLM](https://arxiv.org/abs/2306.08543), [γOPD](https://arxiv.org/abs/2609.16937)) or with a discount
    of zero, each token scored only for itself
    ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). The soft score is a real difference:
    against a deterministic expert like this one, the teacher's log-probability is $-\infty$ for every mistake, so it
    can't rank mistakes at all.
  - Further away: [DeepMimic](https://arxiv.org/abs/1804.02717) trains with PPO on a reward for tracking a reference
    motion's states, and [SQIL](https://arxiv.org/abs/1905.11108) rewards demonstrated transitions from a fixed dataset
    rather than querying an expert.
- **The imitation gap.** [Weihs et al. (2021)](https://arxiv.org/abs/2007.12173) name it: when the expert has privileged
  information, imitation converges to the expert's actions averaged over what the learner can't see, which can be far
  from the best policy the learner could execute. Their ADVISOR method weights imitation and RL losses state by state,
  depending on how well the learner can imitate there. [Warrington et al. (2021)](https://arxiv.org/abs/2012.15566)
  adapt the expert itself toward what the learner can follow. [Swamy et al. (2022)](https://arxiv.org/abs/2208.02225) study
  imitation when the expert acts on a context the learner doesn't observe, and show why on-policy training matters there.
- **Roll-out policies in learning to search.** [SEARN](https://arxiv.org/abs/0907.0786),
  [AggreVaTe](https://arxiv.org/abs/1406.5979) and [AggreVaTeD](https://arxiv.org/abs/1703.01030) assign each action a
  cost-to-go estimated by rolling out a reference or the learner. [LOLS](https://arxiv.org/abs/1502.02206) analyzes
  the choice between rolling out the reference and rolling out the learned policy, and shows that rolling out only the
  reference can do poorly when the reference is suboptimal. The AggreVaTe tie above is a cousin of that failure: the
  reference is optimal, but the learner can't follow it.

What the toy adds is a construction where the effect has nowhere else to come from: the root has a unique observation,
parameters are not shared, the expert rule is the same everywhere, and DAgger's indifference (and AggreVaTe's) is exact
rather than empirical.

## What to take away

- **A local imitation loss can't rank unavoidable mistakes.** If two actions are equally likely to match the expert, the
  per-step label treats them as equal, however different their consequences.
- **Looking ahead isn't enough by itself.** What matters is whose future gets scored: the learner's (returns, learner
  roll-outs) breaks the tie, the expert's (expert roll-outs) doesn't.
- **In distillation, per-token objectives follow the teacher where the student can't.** On-policy forward KL and
  reverse KL with a discount of zero copy the teacher's choices regardless of what the student can do afterwards. The
  sequence-level reverse KL, optimized with returns, leaves the teacher's path, and where its optimum can be computed
  ($k = 0$) it leaves exactly as much as that optimum says.
- **This doesn't make RL a better imitation learner.** When imitation is realizable, or the consequences of a mistake
  don't depend on which mistake it was, supervised labels are far more direct and sample-efficient, as the downstream
  steps here and the realizable students in the distillation experiments show. The point is a regime: the long-horizon objective carries information the immediate label doesn't
  have.

## Setup details

- **Training reward:** $\pm 1$ per step for expert agreement, undiscounted. **Success:** at most 2 disagreements in
  9 steps, used only for evaluation.
- **DAgger:** 40 iterations × 3,000 learner roll-outs. Expert labels at every visited observation are aggregated over all
  iterations and the policy is their label frequencies (pseudocount $10^{-3}$).
- **PPO:** 120 iterations × 3,072 episodes. Advantage: reward-to-go minus the batch-mean reward-to-go at the same
  observation, divided by the global standard deviation; no value network. Clipped surrogate ($\epsilon = 0.2$), 4
  full-batch epochs of gradient ascent with learning rate 0.065. Each observation's logit takes the mean gradient over
  the samples at that observation.
- **GRPO-style:** 120 iterations × 192 groups of 8 trajectories that share $z$. Each trajectory's total return is
  normalized within its group and used as the advantage of all its actions; the same clipped update. No KL term or
  reference policy.
- **Distillation:** the student's features are Legendre polynomials of $t$ rescaled to $[-1, 1]$, one set per branch,
  plus a root logit. DAgger and on-policy forward KL refit the student to the aggregated labels exactly (by Newton's
  method), with the same pseudocount. The RL variants use the PPO settings above; each state's logit takes the mean
  gradient over its samples, then the chain rule through the features.
- **Seeds:** 0–11 for training; evaluation on 50,000 episodes from `default_rng(9000 + seed)`, except with the
  stochastic teacher, where evaluation is exact.

## Limitations

These are hand-built environments with policies of a few parameters. They show a mechanism, not a practical advantage.
In the distillation experiments, the teacher's path is hard for the student by design, and the deviation is a single
early choice; in a language model, the paths the student can follow are not marked out in advance. The
success threshold is arbitrary: it is never used for training, but changing it changes the absolute numbers. PPO and
GRPO see different numbers of episodes per iteration, so compare each with DAgger, not with each other. The natural next
step is an environment where actions change what the learner can observe without being designed to, such as an agent
that can choose to look before it acts, and a method that keeps supervised labels where imitation is possible and uses
returns only to choose among unavoidable mistakes.

## References

- Stéphane Ross, Geoffrey J. Gordon and J. Andrew Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686), AISTATS 2011. (DAgger)
- Stéphane Ross and J. Andrew Bagnell, [Reinforcement and Imitation Learning via Interactive No-Regret Learning](https://arxiv.org/abs/1406.5979), 2014. (AggreVaTe)
- Wen Sun, Arun Venkatraman, Geoffrey J. Gordon, Byron Boots and J. Andrew Bagnell, [Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction](https://arxiv.org/abs/1703.01030), ICML 2017.
- Hal Daumé III, John Langford and Daniel Marcu, [Search-based Structured Prediction](https://arxiv.org/abs/0907.0786), Machine Learning, 2009. (SEARN)
- Kai-Wei Chang, Akshay Krishnamurthy, Alekh Agarwal, Hal Daumé III and John Langford, [Learning to Search Better Than Your Teacher](https://arxiv.org/abs/1502.02206), ICML 2015. (LOLS)
- Luca Weihs, Unnat Jain, Iou-Jen Liu, Jordi Salvador, Svetlana Lazebnik, Aniruddha Kembhavi and Alexander Schwing, [Bridging the Imitation Gap by Adaptive Insubordination](https://arxiv.org/abs/2007.12173), NeurIPS 2021.
- Andrew Warrington, J. Wilder Lavington, Adam Ścibior, Mark Schmidt and Frank Wood, [Robust Asymmetric Learning in POMDPs](https://arxiv.org/abs/2012.15566), ICML 2021.
- Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell and Zhiwei Steven Wu, [Sequence Model Imitation Learning with Unobserved Contexts](https://arxiv.org/abs/2208.02225), NeurIPS 2022.
- Francis Maes, Ludovic Denoyer and Patrick Gallinari, [Structured prediction with reinforcement learning](https://link.springer.com/article/10.1007/s10994-009-5140-8), Machine Learning, 2009.
- Wojciech M. Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant M. Jayakumar, Grzegorz Swirszcz and Max Jaderberg, [Distilling Policy Distillation](https://arxiv.org/abs/1902.02186), AISTATS 2019.
- Aaron Walsman, Muru Zhang, Sanjiban Choudhury, Dieter Fox and Ali Farhadi, [Impossibly Good Experts and How to Follow Them](https://openreview.net/forum?id=sciA_xgYofB), ICLR 2023.
- Hai Nguyen, Andrea Baisero, Dian Wang, Christopher Amato and Robert Platt, [Leveraging Fully Observable Policies for Learning under Partial Observability](https://arxiv.org/abs/2211.01991), CoRL 2022.
- Shiqi Liu et al., [Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation](https://arxiv.org/abs/2609.16937), 2026. (γOPD)
- Thinking Machines Lab, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
- Yuxian Gu, Li Dong, Furu Wei and Minlie Huang, [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.
- Xue Bin Peng, Pieter Abbeel, Sergey Levine and Michiel van de Panne, [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717), SIGGRAPH 2018.
- Siddharth Reddy, Anca D. Dragan and Sergey Levine, [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/abs/1905.11108), ICLR 2020.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024. (GRPO)
- Code: [github.com/khanhptnk/future-aware-imitation](https://github.com/khanhptnk/future-aware-imitation).
