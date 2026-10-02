---
title: "When immediate imitation is not enough"
date: 2026-10-01
tags:
  - imitation-learning
  - reinforcement-learning
---

# When immediate imitation is not enough

[DAgger](https://arxiv.org/abs/1011.0686) fixes the classic failure of behavior cloning by training on the states the
learner actually visits: roll out the learner, ask the expert what it would do at each visited state, and fit those
labels with supervised learning. That is the right thing to do when the learner can imitate the expert. Often it can't,
for one of three reasons:

- **Privileged information.** The expert sees something the learner doesn't, such as a hidden goal or the true state.
- **A stochastic expert.** In some states the expert's choices are random, so no one, however much they observe, can
  predict them.
- **Limited capacity.** The learner sees everything but is too small to represent the expert, as when a large model is
  distilled into a small one.

## Two targets

Call $\Pi$ the set of policies the learner can represent: policies that depend only on what it observes and fit its size. When the
learner can't imitate the expert, that is, can't reproduce the expert's actions, a method has to settle somewhere in
$\Pi$, and there are two natural places:

- **The projection of the expert onto $\Pi$:** the policy in $\Pi$ closest to the expert, by whatever measure of
  closeness the method uses. For DAgger that is its imitation loss on the expert's actions; for AggreVaTe, the expert's
  cost-to-go. This is where imitation learning settles.
- **The best policy in $\Pi$:** the policy in $\Pi$ with the highest return.

<span class="emph-red">When the learner can't imitate the expert, the projection of the expert and the best policy in
the learner's class are in general not the same policy.</span>

If the learner can reproduce the expert's actions, both are the expert. In the first environment below, DAgger's projection takes each first action half the time: the expert's choice, averaged over
what the learner can't see. The best policy in $\Pi$ always takes the action that reveals what the learner needs to know
to imitate the expert afterwards, even though that action matches the expert only half the time (no first action can do
better).

The methods in this post split by which of the two they find:

- **DAgger** and **[AggreVaTe](https://arxiv.org/abs/1406.5979)** find the projection.
- **[LOLS](https://arxiv.org/abs/1502.02206)** with learner roll-outs and **Agreement PPO (APPO)** find the best policy
  in $\Pi$. APPO is a
  simple way to do it: run PPO on the learner's own roll-outs with a reward of $+1$ when the learner's action matches
  the expert's and $-1$ when it doesn't, and credit each action with the agreement the learner itself goes on to
  achieve. It uses exactly the expert queries DAgger uses; only what it does with them differs.

For each of the three reasons, I built the smallest environment I could find where this is all there is, and case 3
makes the same comparison among distillation objectives. All methods get the same budget and the same tuning protocol;
Figure 1 summarizes the results.

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/summary-light.svg" alt="Bar chart of task success in three settings for DAgger, AggreVaTe, LOLS and APPO. Privileged information: DAgger 55, AggreVaTe 71, LOLS 100, APPO 100. Stochastic expert: DAgger 76, AggreVaTe 92, LOLS 100, APPO 100. Limited capacity with a degree-1 student: DAgger 17, AggreVaTe 0, LOLS 100, APPO 100.">
<img class="theme-dark" src="../assets/future-aware-imitation/summary-dark.svg" alt="Bar chart of task success in three settings for DAgger, AggreVaTe, LOLS and APPO. Privileged information: DAgger 55, AggreVaTe 71, LOLS 100, APPO 100. Stochastic expert: DAgger 76, AggreVaTe 92, LOLS 100, APPO 100. Limited capacity with a degree-1 student: DAgger 17, AggreVaTe 0, LOLS 100, APPO 100.">
<figcaption>Figure 1. Task success, meaning at most 2 disagreements with the expert in a 9-step episode (a metric no method trains on), mean over 12 seeds. Blue methods find the projection of the expert onto the learner's class; orange ones the best policy in that class. Every run reports the policy it converges to (or its last, if it never does). In the first two settings each AggreVaTe run is a coin flip at the root, so its bars average 12 coin flips (see case 1). For limited capacity the student is a degree-1 polynomial; Figure 8 covers every size.</figcaption>
</figure>

*Table 1. The first move of the best policy in the learner's class, in each setting, and which first move each method's
training signal prefers. DAgger and AggreVaTe aim at the projection of the expert; LOLS and APPO at the best policy in
the class.*

| Setting | Best policy's first move | DAgger, AggreVaTe | LOLS, APPO |
|---|---|---|---|
| Privileged information | reveal the hidden information | indifferent | reveal |
| Stochastic expert | the side whose mistake is recoverable | indifferent | that side |
| Limited capacity (degree ≤ 4) | leave the teacher's path | follow the teacher | leave |

LOLS and APPO reach the same policies here; [What APPO costs](#what-appo-costs) compares them and lists APPO's costs.

With privileged information this is the *imitation gap* ([Weihs et al., 2021](https://arxiv.org/abs/2007.12173)), and
LOLS already fixes it with roll-outs of the learner. Close relatives of APPO, and how they differ, are in
[Related work](#related-work). What the
toys add is constructions clean enough to compute what every method is aiming at, exactly, and check it against the
runs. All code is at [github.com/khanhptnk/future-aware-imitation](https://github.com/khanhptnk/future-aware-imitation)
and runs on a CPU in about an hour, tuning included.

## The setup, precisely

**States, observations, rewards.** At step $t$ the environment is in a state $s_t$, which includes everything the expert's
choices depend on. The learner's policy $\pi(a \mid o_t)$ sees only an observation $o_t$, which can leave things out.
Every action is binary, 0 or 1. The expert plays $a^E_t$, and the learner earns $r_t = +1$ when its action matches it
and $r_t = -1$ otherwise. Every episode has 9 steps, and the shared objective is the expected agreement,

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\Big[\sum_{t=0}^{8} r_t\Big],
$$

which is 9 minus twice the expected number of disagreements. The expert scores $J = 9$.

**Two ways to value an action.** Take action $a$ in state $s$ at step $t$, then let some policy play the rest of the
episode. The expected return from $t$ on depends on who plays the rest:

- $Q^E_t(s, a)$: the **expert** plays the rest. Its advantage is $A^E_t(s, a) = Q^E_t(s, a) - V^E_t(s)$, where
  $V^E_t(s)$ is the expert's own value.
- $Q^\pi_t(s, a)$: the **learner** plays the rest.

The two targets differ because these two values differ.

> [!note] When is following the expert's advantage enough?
> The performance difference lemma says that for any policy $\pi$,
>
> $$
> J(\pi^E) - J(\pi) = \sum_t \mathbb{E}_{s_t \sim d^t_\pi}\,\mathbb{E}_{a \sim \pi}\big[-A^E_t(s_t, a)\big],
> $$
>
> where $d^t_\pi$ is the distribution of states $\pi$ visits at step $t$. If the expert is optimal, every term is
> $\ge 0$, and the policy that plays $\arg\max_a A^E_t(s, a)$ **at every state** makes every term 0: it is optimal,
> whatever states it visits. That needs the policy to choose separately at every state. A learner can't, wherever two
> states with different best actions look the same to it (it sees only $o_t$) or can't be fit separately (its few
> parameters are shared across states). Then not every term can be 0, the gap $J(\pi^E) - J(\pi)$ also depends on
> *which states the learner visits* ($d^t_\pi$), and choosing by $A^E$ at each observation no longer finds the best
> policy in $\Pi$: it finds a projection of the expert. The three environments below are instances of this.

**The methods.** All of them roll out the learner and query the expert at the states it visits. Because the learner
can't choose separately at every state, every one of them, in effect, pools what it learns over states: those that share
an observation, or those that share parameters. What differs is **what** they pool:

| Method | The value of an action | Finds |
|---|---|---|
| DAgger | none: it fits the expert's action as a label | the projection (by its loss on the expert's actions) |
| AggreVaTe | $Q^E$, the return when the expert plays the rest | the projection (by the expert's cost-to-go) |
| LOLS ($\beta = 0$) | $Q^\pi$, the return when the learner plays the rest, every action tried | the best policy in $\Pi$ |
| APPO | $Q^\pi$, the episode's own return from that step on | the best policy in $\Pi$ |

("Finds" is what each method's training signal aims at. LOLS and APPO can stop at a local optimum; in these toys they
reach the best policy in $\Pi$ at their tuned settings (APPO at degree 7 of case 3 only when played greedily), though
APPO with too large a step size stops short at degree 7.)

AggreVaTe ([Ross & Bagnell, 2014](https://arxiv.org/abs/1406.5979)) picks a random step of a learner roll-out, takes a
random action there, and lets the expert finish. LOLS ([Chang et al., 2015](https://arxiv.org/abs/1502.02206)) tries every
action at a random step (here, both) and lets the learner finish each (or, with probability $\beta$, the expert). APPO
does no explicit exploration: it credits every action the learner takes with that episode's return from that step on,
minus the average at the same observation, and takes PPO's clipped gradient steps. (A GRPO-style variant, Agreement GRPO
or AGRPO, reaches the same root decisions as APPO in every setting; see Setup details.) With $\beta = 0$, LOLS and APPO
maximize the same objective, $J$, by two different reinforcement learning methods: approximate policy iteration with
explicit roll-outs, and policy gradient. I report both $\beta = 0$ and the LOLS paper's recommended $\beta = 0.5$.

**Protocol.** Every method simulates the same number of episodes per training run, 368,640, counting roll-ins and
roll-outs alike (so LOLS, which plays three episodes per roll-in, gets a third as many iterations as APPO). In every
round (iteration) of training, each method records the policy it played and that
policy's own training loss on the round's data: its imitation loss, its cost-sensitive loss, or, for the RL methods,
minus the batch's return. A run has converged once that loss hasn't improved by more than its standard error for a
quarter of the budget, and the policy at that moment is reported. The tables also give each method's *regret over
training*: the errors per episode of the policy played in each round, averaged over all rounds, minus the fewest
possible. It is the analogue, in the shared objective, of the average loss that the no-regret analyses of DAgger and
AggreVaTe bound on their own losses. *Task success* means at most 2 disagreements in the 9 steps; no method trains on
it.

Each tunable knob, the learning rate of APPO (twelve values from 0.0075 to 16) and of the RL distillation objectives in
case 3 (sixteen values from 0.0005 to 16), and $\beta \in \{0, 0.5\}$ for LOLS, is searched separately in every setting on
five tuning seeds, selecting by $J$ (the distillation objectives by their own objectives; see case 3); the chosen setting
is then trained on twelve other seeds, which are what's reported. APPO's chosen step size is often the largest in the
grid, 16, but its $J$ there is within 0.01 of the best possible, so a larger one can't do meaningfully better. DAgger and
AggreVaTe fit their data exactly and have nothing to tune. In every environment, the decision that matters is the first
one, at a *root* observation that has its own parameter, so nothing learned later leaks into it through shared
parameters.

## Case 1: privileged information

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/env-light.svg" alt="Diagram: a root observation that hides z. Action 1 leads to 8 steps where z is visible; action 0 leads to 8 steps where z is still hidden. Either action matches the expert half the time, and the expert plays z at every step.">
<img class="theme-dark" src="../assets/future-aware-imitation/env-dark.svg" alt="Diagram: a root observation that hides z. Action 1 leads to 8 steps where z is visible; action 0 leads to 8 steps where z is still hidden. Either action matches the expert half the time, and the expert plays z at every step.">
<figcaption>Figure 2. The privileged-information environment. As far as immediate imitation goes, the root action is a coin flip, but only action 1 makes the rest of the episode imitable.</figcaption>
</figure>

**Setup.** Each episode draws a hidden bit $z$, 0 or 1 with equal probability. The state is the learner's observation
together with $z$. The expert sees $z$ and plays $a^E_t = z$ at every step. The learner doesn't see $z$ at the root, so
whatever it does there, it matches the expert with probability $\tfrac{1}{2}$. Its later observations depend only on its
root action: after action 1 they show $z$, after action 0 they don't. The policy is tabular, one probability per
observation.

**The two targets.**

- *The expert's policy* plays $z$ at every step, $J = 9$. It is not in $\Pi$: it would need $z$ at the root.
- *Its projection onto $\Pi$ (DAgger's):* at the root, the expert's action given what the learner sees is 0 or 1 with probability
  $\tfrac{1}{2}$ each, so the projection takes each half the time. Afterwards it copies $z$ after a reveal and guesses
  after a hide. $J = \tfrac{1}{2} \cdot 8 + \tfrac{1}{2} \cdot 0 = 4$.
- *The best policy in $\Pi$:* take action 1 at the root (reveal), then copy $z$: $J = 0 + 8 = 8$. In the episodes where
  $z = 0$ it disagrees with the expert at the root, on purpose. Taking action 0 instead would match the expert just as
  often at the root and then leave 8 steps of guessing: $J = 0$.

**What each method sees at the root.** The root observation is the same whether $z = 0$ or $z = 1$, so each method's data
at the root mix the two kinds of episode, half each, and its root decision follows the average:

| What is pooled at the root | $z = 0$ | $z = 1$ | Average over $z$ | Prefers |
|---|---|---|---|---|
| DAgger: the expert's action | 0 | 1 | half 0, half 1 | neither |
| AggreVaTe: $Q^E$, action 0 / action 1 | 9 / 7 | 7 / 9 | **8 / 8** | neither |
| LOLS, APPO: $Q^\pi$, action 0 / action 1 | 1 / 7 | $-1$ / 9 | **0 / 8** | action 1 |

Each entry is the return from the root on: the root step's $\pm 1$, plus 8 more steps played by the expert ($+8$
whatever happened at the root, since the expert always knows $z$) or by the learner ($+8$ after a reveal, an average of
0 after a hide, once it has learned to copy $z$).

- **DAgger** fits the expert's action, which is $z$: half the labels say 0 and half say 1, and its cross-entropy at the
  root is minimized at probability $\tfrac{1}{2}$. Every episode, it flips a fair coin between a future it can imitate
  and one it can't.
- **AggreVaTe** aims at $\arg\max_a A^E$, the action the expert rates best, which is the expert's own action. In each
  state that action is $z$, so the two kinds of episode want opposite actions, and averaged over $z$ they cancel. Its
  estimates are correct; the target isn't available. Revealing $z$ is worth nothing to an expert that already knows it,
  so nothing in $Q^E$ rewards it. In the runs, its estimated gap between the two root actions ranges from $-0.025$ to
  $+0.027$ across seeds at the end of training, within sampling noise.
- **LOLS and APPO** pool $Q^\pi$, the return the learner itself gets. Revealing costs at most 2 at the root and gains 8
  afterwards, whatever $z$ is, so both kinds of episode prefer it and the average keeps the preference. (More precisely,
  if the learner copies $z$ correctly with probability $c$ after a reveal, the gain afterwards is $8(2c - 1)$, and
  revealing wins in both kinds of episode once $c > 0.625$. Early in training $c$ is about $\tfrac{1}{2}$, so the
  preference for revealing appears only as the learner learns to copy $z$.)

In terms of the box above: at the root, both actions have expert advantage $-1$ on average, a tie, so that gap's root
term doesn't care. The 8 that separates revealing from hiding sits in the later terms: whether the learner then visits states
where it can copy $z$ (each term 0) or states where it must guess (each term 1). Only a score that includes the learner's
own future, $Q^\pi$, charges those later terms to the root action that causes them.

**Results.**

*Table 2. Privileged information: mean ± standard deviation over 12 seeds, 50,000 evaluation episodes each. "Reveals" is
the probability of the revealing action at the root; "greedy" plays each policy's more likely action. "Regret" is the
errors per episode over training, averaged over every round, minus the 0.5 of the best policy in the learner's class,
which has 100% success. ↑ higher is better, ↓ lower is better; the best value in each column is in bold.*

| Method | Reveals ↑ | Errors ↓ | Success (%) ↑ | Errors, greedy ↓ | Regret ↓ |
|---|---|---|---|---|---|
| DAgger | 0.500 ± 0.001 | 2.50 ± 0.01 | 54.5 ± 0.3 | 1.50 ± 1.81 | 2.01 ± 0.02 |
| AggreVaTe | 0.42 ± 0.51 | 2.83 ± 2.06 | 70.8 ± 25.7 | 2.83 ± 2.06 | 2.26 ± 1.35 |
| LOLS | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | 0.13 ± 0.05 |
| LOLS, $\beta = 0.5$ | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | 0.15 ± 0.05 |
| APPO | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | **0.08 ± 0.02** |

DAgger lands where the analysis puts it. Half its episodes take the revealing branch and succeed; the other half
are 9 coin flips, which succeed with probability $P(\mathrm{Bin}(9, \tfrac{1}{2}) \le 2) = 46/512$, so its success rate is
$\tfrac{1}{2} + \tfrac{1}{2} \cdot \tfrac{46}{512} = 54.49\%$. Played greedily, DAgger is a per-seed coin flip at the
root. LOLS, with either $\beta$, and APPO reveal in every seed and reach the best policy in $\Pi$; APPO also has the
lowest regret over training.

AggreVaTe's classifier plays one root action deterministically, and since its estimated gap is noise, that action keeps
flipping during training (Figure 3). Where it converges is a coin flip: it reveals in 5 of the 12 seeds here, and in
47% of 600 more. A converged policy that reveals makes 0.5 errors per episode and one that hides makes 4.5, so its
errors average about DAgger's, with a much larger spread across seeds.

The AggreVaTe and DAgger papers end differently: they return the checkpoint that scores best on held-out episodes.
Here that would rescue AggreVaTe in about 9 runs out of 10 (DAgger, whose root stays at $\tfrac{1}{2}$, not at all),
because its root keeps switching sides during training and a revealing checkpoint scores $J = 8$ against the others'
$0$. That rescue is a search over checkpoints by the learner's own
return, the information LOLS and APPO train on, and it works only because the decision that matters is one binary choice
that noise flips back and forth. With many such decisions, a checkpoint would have to get all of them right by chance,
and where the training signal points the wrong way (case 3), no checkpoint leaves the teacher's path at all.

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/root-light.svg" alt="Probability of the revealing action against simulated episodes, mean over seeds. DAgger stays at 0.5; AggreVaTe wanders between about 0.15 and 0.6 as its seeds switch actions with sampling noise; APPO reaches 1 within about 15,000 episodes and LOLS within about 30,000.">
<img class="theme-dark" src="../assets/future-aware-imitation/root-dark.svg" alt="Probability of the revealing action against simulated episodes, mean over seeds. DAgger stays at 0.5; AggreVaTe wanders between about 0.15 and 0.6 as its seeds switch actions with sampling noise; APPO reaches 1 within about 15,000 episodes and LOLS within about 30,000.">
<figcaption>Figure 3. The root decision during training (mean over 12 seeds; shaded: range across seeds for DAgger and APPO). Every method's x-axis ends at the same budget. Each AggreVaTe seed plays one root action deterministically, and which one keeps changing as sampling noise flips its estimated gap. Each run reports its policy at the round where convergence is declared.</figcaption>
</figure>

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/errors-light.svg" alt="Bar chart of the number of expert disagreements per episode. DAgger is bimodal: about half its episodes have 0 or 1 errors, the rest spread from 2 to 9. AggreVaTe has about 70% of its mass at 0 or 1 errors and the rest at 8 or 9. LOLS and APPO have all episodes at 0 or 1 errors.">
<img class="theme-dark" src="../assets/future-aware-imitation/errors-dark.svg" alt="Bar chart of the number of expert disagreements per episode. DAgger is bimodal: about half its episodes have 0 or 1 errors, the rest spread from 2 to 9. AggreVaTe has about 70% of its mass at 0 or 1 errors and the rest at 8 or 9. LOLS and APPO have all episodes at 0 or 1 errors.">
<figcaption>Figure 4. Errors per episode with privileged information, computed exactly from each reported policy and averaged over seeds.</figcaption>
</figure>

## Case 2: a stochastic expert

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/env-hard-light.svg" alt="Diagram: at the root the expert flips a coin z. Action 1 leads to an easy corridor if z is 1 and a recoverable corridor if z is 0. Action 0 leads to an easy corridor if z is 0 and a hard corridor, where the expert keeps flipping coins, if z is 1. Either action matches the expert half the time; downstream the expert plays 0 except in the hard corridor.">
<img class="theme-dark" src="../assets/future-aware-imitation/env-hard-dark.svg" alt="Diagram: at the root the expert flips a coin z. Action 1 leads to an easy corridor if z is 1 and a recoverable corridor if z is 0. Action 0 leads to an easy corridor if z is 0 and a hard corridor, where the expert keeps flipping coins, if z is 1. Either action matches the expert half the time; downstream the expert plays 0 except in the hard corridor.">
<figcaption>Figure 5. The stochastic-expert environment. Both root actions match the expert's coin half the time, but only action 0's mistake leads to states where the expert keeps acting randomly.</figcaption>
</figure>

**Setup.** At the root the expert flips a fair coin $z$ and plays it. Nothing anyone observes predicts the coin, so
whatever the learner does there, it matches the expert half the time. Here the root is unpredictable because the expert
is random, not because it sees something the learner doesn't. What differs from
case 1 is where mistakes lead. Matching the expert's root action enters an easy corridor. The mistake "1 when the expert
played 0" enters a recoverable corridor, and "0 when it played 1" a hard corridor. In the easy and recoverable corridors
the expert always plays 0; in the hard corridor it keeps flipping a fresh coin at every step. The learner sees which
corridor it is in. This construction is more artificial than case 1, because the asymmetry is put into the expert
directly.

**The two targets.**

- *The expert's policy* flips a coin at the root and then plays 0: its own root choice always leads it into the easy
  corridor. $J = 9$, since it always agrees with itself. No learner can reproduce its coin flips: a copy of its policy,
  flipping its own coins, agrees with it only half the time at the root.
- *Its projection onto $\Pi$ (DAgger's)* is exactly that copy: each root action half the time, then 0 in the easy and
  recoverable corridors and a coin flip in the hard one. It enters the hard corridor in a quarter of the episodes:
  $J = \tfrac{3}{4} \cdot 8 = 6$. Here the learner *can* represent the expert's policy, and the projection is the
  expert's own distribution, yet it still isn't the best policy.
- *The best policy in $\Pi$:* action 1 at the root, then 0. When the expert's coin says 0, its root action is a mistake,
  but the recoverable one; it never risks the hard corridor. $J = 0 + 8 = 8$.

**What each method sees at the root.**

| What is pooled at the root | $z = 0$ | $z = 1$ | Average over $z$ | Prefers |
|---|---|---|---|---|
| DAgger: the expert's action | 0 | 1 | half 0, half 1 | neither |
| AggreVaTe: $Q^E$, action 0 / action 1 | 9 / 7 | 7 / 9 | **8 / 8** | neither |
| LOLS, APPO: $Q^\pi$, action 0 / action 1 | 9 / 7 | $-1$ / 9 | **4 / 8** | action 1 |

The expert agrees with itself in every corridor, coin flips included, so $Q^E$ is again blind to where the root action
leads. Under $Q^\pi$ the two kinds of episode now **disagree**: when $z = 0$, action 0 is the better one (it is correct
and leads to the easy corridor). But the average still prefers action 1 by 4, because action 0 gains 2 when $z = 0$ and
loses 10 when $z = 1$, where it leads the learner into 8 steps of coin flips. Pooling over $z$ is unavoidable; pooling
the learner's own returns is what makes the expected cost of the hard corridor visible.

DAgger's exact success rate follows the same way as in case 1: it learns the easy and recoverable corridors perfectly
and plays $\tfrac{1}{2}$ in the hard one, so it succeeds with probability
$\tfrac{1}{2} + \tfrac{1}{4} + \tfrac{1}{4} \cdot P(\mathrm{Bin}(8, \tfrac{1}{2}) \le 1) = 75.88\%$.

**Results.**

*Table 3. Stochastic expert, 12 seeds. "Recoverable side" is the probability of root action 1, whose mistake is the
recoverable one. The best policy in the learner's class makes 0.5 errors per episode with 100% success; regret as in
Table 2. ↑ higher is better, ↓ lower is better; the best value in each column is in bold.*

| Method | Recoverable side ↑ | Errors ↓ | Success (%) ↑ | Errors, greedy ↓ | Regret ↓ |
|---|---|---|---|---|---|
| DAgger | 0.500 ± 0.002 | 1.50 ± 0.01 | 75.9 ± 0.2 | 1.33 ± 1.03 | 1.03 ± 0.02 |
| AggreVaTe | 0.83 ± 0.39 | 0.83 ± 0.78 | 92.0 ± 18.8 | 0.83 ± 0.78 | 0.42 ± 0.27 |
| LOLS | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | 0.12 ± 0.03 |
| LOLS, $\beta = 0.5$ | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | 0.12 ± 0.03 |
| APPO | **1.000 ± 0.000** | **0.50 ± 0.00** | **100.0 ± 0.0** | **0.50 ± 0.00** | **0.07 ± 0.03** |

AggreVaTe's estimated values tie again. It converges to the recoverable side in 10 of the 12 seeds here, but in only
45% of 600 more: these 12 happen to favor it. Its side persists for long stretches of training, so an average over
12 seeds varies a lot.

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/root-hard-light.svg" alt="Probability of the recoverable side against simulated episodes, mean over seeds. DAgger stays at 0.5; AggreVaTe wanders as its seeds switch sides; LOLS and APPO rise to about 1.">
<img class="theme-dark" src="../assets/future-aware-imitation/root-hard-dark.svg" alt="Probability of the recoverable side against simulated episodes, mean over seeds. DAgger stays at 0.5; AggreVaTe wanders as its seeds switch sides; LOLS and APPO rise to about 1.">
<figcaption>Figure 6. The root decision during training with a stochastic expert (mean over 12 seeds).</figcaption>
</figure>

## Case 3: limited capacity (distillation)

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/env-distill-light.svg" alt="Diagram: a root where nothing is hidden and the teacher plays 0. Action 0 follows the teacher's path, where the teacher plays the parity of t, which needs a degree-7 student. Action 1 leads to another path where the teacher plays 0, easy for any student. The student is a degree-k polynomial in t.">
<img class="theme-dark" src="../assets/future-aware-imitation/env-distill-dark.svg" alt="Diagram: a root where nothing is hidden and the teacher plays 0. Action 0 follows the teacher's path, where the teacher plays the parity of t, which needs a degree-7 student. Action 1 leads to another path where the teacher plays 0, easy for any student. The student is a degree-k polynomial in t.">
<figcaption>Figure 7. The limited-capacity environment. Copying the teacher's first action leads to behavior a small student can't fit; deviating once leads to behavior any student can.</figcaption>
</figure>

**Setup.** No information is hidden: the student sees exactly what the teacher sees and is simply smaller. The episode is
a root decision followed by 8 steps in one of two branches, chosen by the root action, and the student observes the
branch and the step $t$. The teacher plays 0 at the root, so its own path is branch 0, where it plays the parity of $t$
($1, 0, 1, 0, \dots$). In branch 1 it always plays 0. The student has its own logit at the root and, in each branch, a
logistic policy whose logit is a polynomial of degree $k$ in $t$. Degree 7 is enough to fit the parity of 8 steps. A
polynomial of degree $k$ changes sign at most $k$ times, so on the teacher's path a smaller student makes at least
$\lceil (7 - k)/2 \rceil$ errors. Leaving the teacher's path costs exactly one error, and branch 1 is easy at any size.

**The two targets.** Here $\Pi$ is the set of degree-$k$ students.

- *The teacher's policy* follows its own path and plays the parity, $J = 9$. It is in $\Pi$ only for $k = 7$.
- *Its projection onto $\Pi$:* copy the teacher's root action, then the degree-$k$ policy closest to the parity on the
  teacher's path. For $k = 1$ that path costs at least 3 errors, so $J \le 1 + (8 - 2 \cdot 3) = 3$.
- *The best policy in $\Pi$:* for $k \le 4$, leave at the root and play 0: one error, $J = 7$. (For $k = 5$ and $6$
  leaving and following tie at one error; for $k = 7$ the best policy in $\Pi$ is the teacher.)

**What each method sees at the root.** Here nothing is hidden, so there is only one root state and nothing to average.
DAgger's label there is the teacher's action, always 0: copy. The values of the two root actions:

| At the root, degree 1 | Copy (action 0) | Leave (action 1) | Prefers |
|---|---|---|---|
| AggreVaTe: $Q^E$ | $1 + 8 = 9$ | $-1 + 8 = 7$ | copy |
| LOLS, APPO: $Q^\pi$ | at most 3 | $-1 + 8 = 7$ | leave |

The teacher's best action, which is what DAgger and AggreVaTe aim at, is copying: the teacher can play its own path
perfectly. The student can't. Only $Q^\pi$ counts the at-least-3 errors the student would make on the teacher's path.

**Results.**

*Table 4. Limited capacity with a deterministic teacher (12 seeds; standard deviations across seeds are at most 0.01,
except for the regret of LOLS with $\beta = 0.5$ at degree 1, 0.04, and of APPO at degree 7, 0.02).
"Leave" is the probability of leaving the teacher's path at the root; it has no better direction, since leaving is right
for a degree-1 student and wrong for a degree-7 one, which can represent the teacher. The fewest errors possible are 1 at
degree 1 and 0 at degree 7; regret is the errors per episode over training minus those. At degree 7, DAgger's and
APPO's losses are still improving slowly at the end of the budget, so they report their last policy. ↓ lower is
better; the best value in each column is in bold.*

| Method | Degree 1: leave | Degree 1: errors ↓ | Degree 1: regret ↓ | Degree 7: leave | Degree 7: errors ↓ | Degree 7: regret ↓ |
|---|---|---|---|---|---|---|
| DAgger | 0.000 | 3.82 | 2.81 | 0.000 | **0.00** | **0.04** |
| AggreVaTe | 0.000 | 4.00 | 3.01 | 0.000 | **0.00** | 0.08 |
| LOLS | 1.000 | **1.00** | 0.16 | 0.000 | **0.00** | 0.11 |
| LOLS, $\beta = 0.5$ | 1.000 | **1.00** | 0.20 | 0.000 | **0.00** | 0.11 |
| APPO | 1.000 | **1.00** | **0.05** | 0.002 | 0.06 | 0.47 |

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/distill-light.svg" alt="Disagreements per episode against the student's polynomial degree from 0 to 7. DAgger falls from 4 errors at degree 0 to 2.5 at degree 6; AggreVaTe makes 4 errors up to degree 2 and 2 errors from degree 3 to 6; both reach 0 at degree 7. LOLS and APPO make 1 error at every degree below 7 and about 0 at degree 7.">
<img class="theme-dark" src="../assets/future-aware-imitation/distill-dark.svg" alt="Disagreements per episode against the student's polynomial degree from 0 to 7. DAgger falls from 4 errors at degree 0 to 2.5 at degree 6; AggreVaTe makes 4 errors up to degree 2 and 2 errors from degree 3 to 6; both reach 0 at degree 7. LOLS and APPO make 1 error at every degree below 7 and about 0 at degree 7.">
<figcaption>Figure 8. Distillation from a deterministic teacher into students of every size (mean over 12 seeds). The dotted steps are the fewest errors a student of that size can make on the teacher's path; the dashed line is the one error of leaving it.</figcaption>
</figure>

- **Below degree 7, DAgger and AggreVaTe pay for following the teacher:** 2 to 4 errors per episode. DAgger makes more
  than the fewest possible on the teacher's path at every degree from 1 to 6, because maximum likelihood on labels the
  student can't fit gives soft probabilities, not the policy with the fewest errors; AggreVaTe does at degrees 1, 2, 5
  and 6, because its classifier is fit with a logistic surrogate rather than by counting errors. (Degrees pair up, 1 with 2 and so on, because the teacher's
  labels at $t$ and $9 - t$ are opposite, so even-degree terms don't help.) AggreVaTe's 2 errors from degree 3 to 6 fall
  within the success threshold, which is why errors per episode is the better measure here.
- **Here no luck helps them.** In every round, DAgger and AggreVaTe copy the teacher's root action, because their
  training signal prefers it.
- **LOLS and APPO leave the teacher's path** and make exactly one error per episode at every size below 7.
- **LOLS with the paper's mixture, $\beta = 0.5$, leaves only for the smallest students.** It leaves in every seed at
  degrees 0 to 2, in about half the seeds at degrees 3 and 4, and never at degrees 5 and 6, where it makes 2 errors per
  episode, like AggreVaTe. Half its roll-outs are the teacher's, which plays its own path perfectly, so they make copying
  look better than it is for the student. Tuning picks $\beta = 0$ at every size.
- **At degree 7 the two targets coincide (both are the teacher), and everyone follows it.** DAgger, AggreVaTe and LOLS imitate it
  exactly. APPO does too when played greedily, but only at the right step size: tuning picked 0.5, while every step size
  of 2 or more commits to the easy branch before learning the parity and ends with a return of 7 instead of 9.

### With a stochastic teacher

Language-model teachers are distributions, and the standard distillation objectives compare distributions. So make the
teacher put probability 0.9 on the deterministic teacher's action in every state, including the root, and 0.1 on the other. The per-token
objectives below find projections of the teacher, each by its own divergence; the objectives with unnormalized returns find the
best student for their objective. Seven ways of training the student, with the same budget:

- **Off-policy KD:** cross-entropy to the teacher's distribution on episodes the *teacher* generates, as in word-level
  and sequence-level knowledge distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531);
  [Kim & Rush, 2016](https://arxiv.org/abs/1606.07947)). (In practice sequence-level KD trains on the teacher's
  beam-search output, its most likely sequence, which follows the teacher's path too.)
- **On-policy forward KL:** the same on episodes the student generates, which is DAgger with soft labels (GKD's forward
  KL, and the "DAgger-style on-policy SFT" of [Zhao et al. (2026)](https://arxiv.org/abs/2605.16826), discussed in my
  [previous post](./kl-penalty-in-grpo)).
- **On-policy JSD:** GKD's generalized Jensen-Shannon divergence with $\beta = 0.5$
  ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)), on the student's episodes.
- **Reverse KL with a discount of zero:** a per-token reward $\log \pi_T(a \mid s) - \log \pi_S(a \mid s)$, with each token
  credited only with its own reward, as in
  [Thinking Machines' on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- **Reverse KL with returns:** the same reward with undiscounted returns, which follows the gradient of the
  sequence-level reverse KL $\mathrm{KL}(P_S \,\|\, P_T)$ (see the [previous post](./kl-penalty-in-grpo)), the objective
  at the core of [MiniLLM](https://arxiv.org/abs/2306.08543).
- **MiniLLM as published:** the same objective with MiniLLM's estimator, which starts from supervised fine-tuning on
  the teacher's samples, mixes teacher actions into the student's episodes, and divides the return of the steps after
  $t$ by their number (length normalization); details in Setup details.
- **APPO:** $+1$ for agreeing with the teacher's preferred action and $-1$ otherwise, as before. It therefore trains
  exactly as with the deterministic teacher; only the evaluation differs.

The RL objectives share APPO's update (MiniLLM uses its own), and their learning rates are tuned by their own
objectives: the sequence-level reverse KL for the reverse-KL ones, including MiniLLM, and agreement for APPO. An
episode has only $2 \times 2^8 = 512$ action sequences, so every metric is computed exactly by summing over all of them.

<figure>
<img class="theme-light" src="../assets/future-aware-imitation/distill-soft-light.svg" alt="Two panels against the student's degree. Left, the probability of leaving the teacher's path: off-policy KD, on-policy forward KL and on-policy JSD stay at the teacher's 0.1 at every degree; reverse KL with discount zero, at its tuned step size, stays between 0.30 and 0.39 below degree 7 and 0.1 at degree 7; reverse KL with returns leaves with probability 0.87 at degree 0, falling to 0.50 at degree 6 and 0.1 at degree 7; MiniLLM as published stays between 0.11 and 0.13; APPO leaves with probability 1 below degree 7 and almost never at degree 7. Right, disagreements per episode: the per-token objectives fall from about 3.8 to 2.6, then to about 0.9 at degree 7, and MiniLLM follows them except at degree 6 (1.8); reverse KL with returns stays near 2.1 to 2.2, then 1.8 at degree 6 and 0.9; APPO stays at 1 and reaches about 0.06 at degree 7.">
<img class="theme-dark" src="../assets/future-aware-imitation/distill-soft-dark.svg" alt="Two panels against the student's degree. Left, the probability of leaving the teacher's path: off-policy KD, on-policy forward KL and on-policy JSD stay at the teacher's 0.1 at every degree; reverse KL with discount zero, at its tuned step size, stays between 0.30 and 0.39 below degree 7 and 0.1 at degree 7; reverse KL with returns leaves with probability 0.87 at degree 0, falling to 0.50 at degree 6 and 0.1 at degree 7; MiniLLM as published stays between 0.11 and 0.13; APPO leaves with probability 1 below degree 7 and almost never at degree 7. Right, disagreements per episode: the per-token objectives fall from about 3.8 to 2.6, then to about 0.9 at degree 7, and MiniLLM follows them except at degree 6 (1.8); reverse KL with returns stays near 2.1 to 2.2, then 1.8 at degree 6 and 0.9; APPO stays at 1 and reaches about 0.06 at degree 7.">
<figcaption>Figure 9. Distillation from a stochastic teacher (mean over 12 seeds), each objective at its tuned step size. Blue objectives credit each token only with its own score; orange ones use returns. Reverse KL with discount zero ends partway to its fixed point (see below). Disagreements are counted against the teacher's preferred action; the teacher itself averages 0.9.</figcaption>
</figure>

*Table 5. Stochastic teacher, student of degree 1 (12 seeds, exact evaluation). "Leave" is the probability of leaving
the teacher's path at the root (no better direction: the objectives disagree on it). Errors are counted against the
teacher's preferred action. Reverse KL is $\mathrm{KL}(P_S \Vert P_T)$ and forward KL is $\mathrm{KL}(P_T \Vert P_S)$, over
whole episodes, in nats. Standard deviations across seeds are at most 0.03 except where shown. ↓ lower is better; the best
value in each column is in bold. "Over training" is the reverse KL of the policy played in each round, averaged over
all rounds. † Not converged in 5 of 12 seeds: still partway to its fixed point at the end of the budget; see below.*

| Objective | Leave | Errors ↓ | Rev. KL ↓ | Fwd. KL ↓ | Rev. KL over training ↓ |
|---|---|---|---|---|---|
| Teacher (reference) | 0.10 | 0.90 | 0 | 0 | |
| Off-policy KD | 0.100 | 3.64 | 3.49 | **2.54** | 3.50 |
| On-policy fwd. KL | 0.100 | 3.64 | 3.49 | **2.54** | 3.50 |
| On-policy JSD | 0.100 | 3.61 | 3.48 | 2.55 | 3.49 |
| Rev. KL, discount 0 † | 0.384 | 3.26 | 2.74 | 2.77 | 3.07 |
| Rev. KL, returns | 0.840 | 2.12 | **2.13** | 3.89 | **2.19** |
| MiniLLM (as published) | 0.132 | 3.57 | 3.37 | 2.55 | 3.41 |
| APPO | 1.000 | **1.00** | 3.15 | 50 ± 15 | 3.20 |

- **Per-token objectives follow the teacher at every student size.** Off-policy KD, on-policy forward KL
  and on-policy JSD leave the teacher's path with the teacher's own probability, 0.1. This holds for any per-token
  divergence: the root has its own parameter, and each such objective's target there is the teacher's root
  distribution, whatever comes after. Reverse KL with a discount of zero is per-token too, and trained to convergence
  it does the same (next point).
- **Reverse KL with a discount of zero is best stopped early.** Trained to convergence (four times the budget, at step
  sizes from 0.065 to 0.25), it too leaves with probability about 0.10, at a sequence-level reverse KL of about 3.48 for
  a degree-1 student. Its training path
  passes through better points, and tuning its step size by that same reverse KL finds them: it picks 0.002, small
  enough that at the end of the budget its root probability is still falling from the initial $\tfrac{1}{2}$ toward the
  teacher's 0.1 (its loss has stopped improving in only 7 of 12 seeds). That is the 0.38 in Table 5. Even this best
  stopping point is worse than reverse KL with returns (2.74 against 2.13 nats).
- **Objectives with returns leave.** Where the optimum of the sequence-level reverse KL can be worked out by hand, reverse
  KL with returns reaches it. For $k = 0$, the best a student can do on the teacher's path is a coin flip, which costs
  $8\,\mathrm{KL}(\mathrm{Bern}(0.5) \,\|\, \mathrm{Bern}(0.9)) = 4.087$ nats. The optimum then leaves with probability
  $0.1 / (0.1 + 0.9\,e^{-4.087}) = 0.869$, at a KL of 2.162 nats; training reaches 0.865 and 2.163.
- **MiniLLM as published barely leaves** (0.13 at degrees 0 to 2). Its length normalization divides the
  future's log-ratio by the number of steps left, so at the root it weighs a branch's average cost per step instead of
  its total, an eighth as much. It also starts from fine-tuning on the teacher's samples, at the teacher's 0.1. It
  lands between the per-token objectives and the unnormalized one in reverse KL (3.37 nats). Length normalization was
  introduced to stop language models from preferring short responses; with a fixed horizon there is nothing for it to
  fix, and here it mostly removes the future from the root's score.
- **APPO and reverse KL with returns optimize different things.** APPO goes after the teacher's preferred action and
  collapses to a nearly deterministic student: one error per episode, but far from the teacher's distribution (a forward
  KL of 50 nats, varying
  a lot across seeds). Reverse KL with returns stays a distribution and trades errors for KL; it has the lowest reverse
  KL of the seven, which is its objective.
- **At degree 7 the supervised objectives win.** Off-policy KD, on-policy forward KL and JSD match the teacher exactly,
  as does MiniLLM, which starts from fine-tuning on the teacher's samples; the two RL reverse-KL objectives come within
  0.03 nats. APPO plays the teacher's preferred action almost deterministically
  (0.06 errors per episode), which puts it 0.74 nats of reverse KL away from the teacher's distribution.

## What APPO costs

APPO's advantage is specific to choosing among unavoidable mistakes. The experiments above also show its costs:

- **Where imitation is possible, supervised labels are simpler.** With a degree-7 student, DAgger imitates the teacher
  exactly with nothing to tune. APPO matches it only in a narrow band of step sizes around 0.5; at 1 it mostly commits,
  and at 2 or more it always commits, to whichever branch is easy to learn first. Below degree 7, any step size from about 0.25 up works.
- **It targets the expert's most likely action.** Against a stochastic teacher it collapses to that action and abandons
  the teacher's distribution. If matching the distribution is the goal, the sequence-level reverse KL with returns makes
  the same kind of choice at the root while staying a distribution.
- **LOLS with $\beta = 0$ reaches the same policies in these toys,** with nothing to tune once $\beta = 0$, because it
  compares every action from the same state and fits a deterministic classifier (its regret over training is higher
  than APPO's below degree 7 and lower at degree 7). Its price is the ability to restart from intermediate states and
  play one extra episode per action (two here; a whole vocabulary for a language model), which APPO doesn't need.
- **An entropy bonus doesn't fix the collapse without costing errors.** APPO here has no entropy bonus. Adding one
  (Table 6) changes little up to $c = 0.1$. From $c = 0.3$ on, the policy stays random where it should be decisive, so
  errors rise while the root decisions mostly hold. With the stochastic teacher at degree 7, $c = 0.3$ brings the
  reverse and forward KLs to 0.58 and 3.43 nats, still far from reverse KL with returns (0.02 each), and $c = 1$ costs
  2.94 errors per episode. With a degree-1 student it has no effect: the logits saturate within a few iterations, and
  there the entropy gradient, $-\theta\, p(1 - p)$ for a logit $\theta$ with $p = \sigma(\theta)$, vanishes.

*Table 6. APPO with an entropy bonus of weight $c$, at APPO's tuned step size (12 seeds; the training loss includes
the bonus). APPO trains identically with either teacher, so the last two rows evaluate the same degree-7 students
against the stochastic teacher's distribution. ↓ lower is better (no bold: errors and KL trade off as $c$ grows).*

| | $c = 0$ | $c = 0.1$ | $c = 0.3$ | $c = 1$ |
|---|---|---|---|---|
| Case 1: errors ↓ | 0.50 | 0.51 | 1.16 | 1.69 |
| Case 2: errors ↓ | 0.50 | 0.51 | 1.22 | 1.85 |
| Case 3, degree 1: errors ↓ | 1.00 | 1.00 | 1.00 | 1.00 |
| Case 3, degree 7: errors ↓ (either teacher) | 0.06 | 0.08 | 0.15 | 2.94 |
| Stochastic teacher: reverse KL ↓ (nats) | 0.74 | 0.70 | 0.58 | 2.11 |
| Stochastic teacher: forward KL ↓ (nats) | 5.24 | 4.68 | 3.43 | 1.58 |

## Related work

- **The objective.** [Ross et al. (2011)](https://arxiv.org/abs/1011.0686) define imitation's goal as minimizing the
  expected per-step loss under the learner's own state distribution. With 0-1 loss over 9 steps, that is the expected
  number of disagreements, an affine function of $J$ here. DAgger gets there through a reduction to no-regret online
  learning, which fits each iteration's data as if that distribution were fixed. Its guarantee is good when some policy
  in the class imitates the expert well on that data, and weak here, where none can.
  [Czarnecki et al. (2019)](https://arxiv.org/abs/1902.02186) make the general version of this point for policy
  distillation: on-policy distillation updates are in general not the gradient of any objective, and adding the future
  distillation loss as a reward recovers the gradient of the cumulative loss.
- **Learning to search.** [SEARN](https://arxiv.org/abs/0907.0786), [AggreVaTe](https://arxiv.org/abs/1406.5979),
  [AggreVaTeD](https://arxiv.org/abs/1703.01030) and [LOLS](https://arxiv.org/abs/1502.02206) assign each action a
  cost-to-go estimated by rolling out a reference policy, the learner, or a mixture. AggreVaTe was built for a task cost,
  where the expert's cost-to-go tells a cheap mistake from a catastrophic one, assuming the learner can recover like the
  expert; the toys here are exactly where that assumption fails. LOLS analyzes the choice of roll-out policy and shows
  that rolling out only the reference can do poorly when the reference is suboptimal or no policy in the class can
  follow it, the second being the case here. It recommends a mixture of reference and learner roll-outs ($\beta = 0.5$);
  with learner roll-outs only, which it calls the reinforcement-learning setting, it is the closest existing method to
  APPO, and in these experiments it does as well.
- **The closest RL-on-agreement algorithms.** None of these is APPO, and each difference can matter.
  - [Walsman et al. (2023)](https://openreview.net/forum?id=sciA_xgYofB) study experts with privileged information and
    include an expert-matching reward trained with PPO as a baseline: $\pm 0.1$ per step added to the task reward, with
    a learned critic and $\gamma = 0.99$. They observe that it can learn loops that keep collecting agreement reward,
    which variable-length episodes allow and the fixed horizon here rules out. Their implementation has an entropy bonus
    and doesn't normalize advantages, so $\pm 0.1$ there is not equivalent to $\pm 1$.
  - [Maes et al. (2009)](https://link.springer.com/article/10.1007/s10994-009-5140-8) cast structured prediction as RL
    with a per-decision reward of 1 for a correct label and 0 otherwise, trained with policy-gradient and SARSA methods.
    The setting differs (structured prediction from fully observed inputs) and so does the optimizer (no PPO clipping
    or per-observation baseline).
  - Several methods use a soft agreement score instead of $\pm 1$: similarity to a fully observable teacher
    ([COSIL](https://arxiv.org/abs/2211.01991)), or the teacher's log-probabilities in LLM distillation, optimized with
    returns ([MiniLLM](https://arxiv.org/abs/2306.08543), whose published estimator normalizes them by length;
    [γOPD](https://arxiv.org/abs/2609.16937), which discounts them, by default inside a framework that also uses a
    verifier reward) or with a discount of zero ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). Against an expert that
    is deterministic where mistakes happen, as in cases 1 and 3, the teacher's log-probability is $-\infty$ for every mistake, so it can't rank
    mistakes at all.
  - Further away: [DeepMimic](https://arxiv.org/abs/1804.02717) trains with PPO on a reward for tracking a reference
    motion's states, and [SQIL](https://arxiv.org/abs/1905.11108) rewards demonstrated transitions from a fixed dataset
    rather than querying an expert.
- **The imitation gap.** [Weihs et al. (2021)](https://arxiv.org/abs/2007.12173) name it: when the expert has privileged
  information, imitation converges to the expert's actions averaged over what the learner can't see, which can be far
  from the best policy the learner could execute. Their ADVISOR method weights imitation and RL losses state by state,
  depending on how well the learner can imitate there. [Warrington et al. (2021)](https://arxiv.org/abs/2012.15566)
  adapt the expert itself toward what the learner can follow. [Swamy et al. (2022)](https://arxiv.org/abs/2208.02225) study
  imitation when the expert acts on a context the learner doesn't observe, and show why on-policy training matters there.

## What to take away

- **When the learner can't imitate the expert, the projection of the expert onto its class and the best policy in
  that class are in general not the same policy.** The best policy can deliberately disagree with the expert, for example to reveal
  information or to avoid states it can't handle. The projection can't: it is as close to the expert as the method's
  measure of closeness allows, wherever that leads.
- **Imitation finds the projection.** DAgger fits the expert's action, and AggreVaTe the expert's best action
  ($\arg\max_a A^E$), which is optimal only for a policy that can choose at every state. Pooled over the states the
  learner can't tell apart or can't fit separately, those targets can cancel (cases 1 and 2) or point the wrong way
  (case 3).
- **What separates the two kinds of method is whose future scores an action, not looking ahead as such.** AggreVaTe
  looks ahead with the expert's future, and its training signal is as blind as DAgger's: where it converges in cases 1
  and 2 is a coin flip. LOLS and APPO look ahead with the learner's future
  and find its best policy. In distillation, the same split separates the per-token objectives, which follow the
  teacher, from those with returns, unless the returns are shrunk, as by MiniLLM's length normalization.
- **This doesn't make RL a better imitation learner in general.** When imitation is possible, supervised labels are
  simpler and exact. The point is a regime: the long-horizon objective carries information the immediate label doesn't.

## Setup details

- **Budget and tuning:** 368,640 simulated episodes per training run for every method, roll-outs included. Learning rates
  for APPO are searched over twelve values from 0.0075 to 16 (each roughly double the last), those of the RL
  distillation objectives over sixteen values from 0.0005 to 16, and $\beta \in \{0, 0.5\}$ for LOLS, separately in
  every setting, on seeds 100–104; the chosen values are retrained and reported on seeds 0–11. Convergence is checked at 20 checkpoints, one after every 18,432 episodes
  of the budget whatever a method's iteration size: a run has converged when its training loss (each round's, on that
  round's data) has not improved on its best by more than one standard error for 5 checkpoints in a row, and the policy
  played at that moment, 5 checkpoints after the last improvement, is reported; a run that never converges reports its
  last policy. MiniLLM's supervised phase counts toward its budget but not toward the convergence check or regret (its
  loss there is a different objective). Regret over training averages every round, evaluated on 10,000 episodes each
  (exactly with the stochastic teacher). Final evaluation uses 50,000 fresh episodes from `default_rng(9000 + seed)`,
  except with the stochastic teacher, where it is exact. Every trial and chosen value is in `results/*.json` in the code
  repository.
- **DAgger:** 120 iterations × 3,072 learner roll-outs. Expert labels at every visited state are aggregated over all
  iterations and the policy is their label frequencies (pseudocount $10^{-3}$). The roll-ins are the learner's alone
  from the first iteration on, for DAgger and AggreVaTe alike; the papers allow mixing in the expert, which wouldn't
  change the root's data.
- **AggreVaTe:** 60 iterations × 3,072 roll-ins, each with one expert roll-out after a random action at a random step.
  **LOLS:** 40 iterations × 3,072 roll-ins, each with one roll-out per action (two, since the actions are binary) at a
  random step; tuning chose $\beta = 0$ in every setting ($\beta = 0.5$ ties in cases 1 and 2 and at degrees 0 to 2 and 7,
  and ties go to the first value tried; it is also reported on its own). The LOLS paper tries every step of the
  roll-in; one random step per roll-in, as in AggreVaTe, is an unbiased subsample of that. Both aggregate the values
  over iterations and play the action with the higher mean value at each observation. For the polynomial student, the
  cost-sensitive problem is solved as a weighted logistic regression (label: the better action; weight: samples × the
  value gap) and played deterministically.
- **APPO:** 120 iterations × 3,072 episodes. Advantage: reward-to-go minus the batch-mean reward-to-go at the same
  observation, divided by the global standard deviation; no value network. Clipped surrogate ($\epsilon = 0.2$), 4
  full-batch epochs of gradient ascent. Each observation's logit takes the mean gradient over the samples at that
  observation. No entropy bonus and no KL term (see [What APPO costs](#what-appo-costs) for an entropy ablation). Chosen
  learning rate: 16 in cases 1 and 2 and for students of degree 0 to 6, 0.5 for degree 7. At 16 the clipping isn't a
  trust region (the first epoch is never clipped, and early updates move the root logit by up to 11), so APPO behaves
  like a large-step policy gradient with a per-observation baseline.
- **AGRPO** (in the code, not shown): the same reward with a minimal GRPO-style update, 240 iterations × 192 groups of 8
  trajectories; each trajectory's total return, normalized within its group, is the advantage of all its actions. No KL
  penalty, and the same clipped update as APPO (4 epochs per batch, where the GRPO paper takes one).
- **Limited capacity:** the student's features are Legendre polynomials of $t$ rescaled to $[-1, 1]$, one set per
  branch, plus a root logit. DAgger, off-policy KD and on-policy forward KL refit the student to the aggregated labels
  exactly (by Newton's method); on-policy JSD takes 60 Adam steps per iteration (step size 0.1, not tuned). MiniLLM
  spends a sixth of its budget on supervised fine-tuning (maximum likelihood on 20 iterations of teacher-sampled
  episodes), then 100 iterations of its update with $\alpha = 0.2$, 4 epochs and $\epsilon = 0.2$, with no baseline or
  advantage normalization, as in the paper; each episode mixes teacher actions with probability 0.2, with per-token
  importance weights, and each step's own term is computed exactly. Its pre-training loss is left out, since there is
  no pre-training corpus. Off-policy KD and on-policy forward KL coincide here, because within a branch every step is
  visited equally often whoever generates the episodes. The RL variants use APPO's update through the features: each
  state's mean gradient, as above, then the chain rule to the polynomial's coefficients. So
  every visited state counts equally in the update, not in proportion to how often it is visited (in a fixed-horizon
  branch every step is visited equally often anyway; only the root and the two branches differ).

## Limitations

These are hand-built environments with policies of a few parameters. They show a mechanism, not a practical advantage.
In each one, the choice that matters is a single first decision, and the environment marks out which branch is
imitable; in a real task, or a language model, the futures a learner can follow are not marked out in advance. That
single decision is also binary, so a lucky checkpoint can get it right (case 1); a harder test would have many. The
success threshold is arbitrary: it is never used for training, but changing it changes the absolute numbers. The
natural next steps are an environment where actions change what the learner can observe without being designed to, such as an
agent that can choose to look before it acts, and a method that keeps supervised labels where imitation is possible and
uses returns only to choose among unavoidable mistakes.

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
- Geoffrey Hinton, Oriol Vinyals and Jeff Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015.
- Yoon Kim and Alexander M. Rush, [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947), EMNLP 2016.
- Rishabh Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024. (GKD)
- Yuxian Gu, Li Dong, Furu Wei and Minlie Huang, [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.
- Anhao Zhao, Haoran Xin, Yingqi Fan, Junlong Tong, Wenjie Li and Xiaoyu Shen, [Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation](https://arxiv.org/abs/2605.16826), 2026.
- Shiqi Liu et al., [Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation](https://arxiv.org/abs/2609.16937), 2026. (γOPD)
- Thinking Machines Lab, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.
- Xue Bin Peng, Pieter Abbeel, Sergey Levine and Michiel van de Panne, [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717), SIGGRAPH 2018.
- Siddharth Reddy, Anca D. Dragan and Sergey Levine, [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/abs/1905.11108), ICLR 2020.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Zhihong Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024. (GRPO)
- Code: [github.com/khanhptnk/future-aware-imitation](https://github.com/khanhptnk/future-aware-imitation).
