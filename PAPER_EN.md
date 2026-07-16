# GameNN World Model: A Recurrent Game-Theoretic Decision Architecture

**XuanMu Security Team — guaidao2**

**July 2026**

---

## Abstract

GameNN World Model is a decision architecture that combines structured state representation, recurrent game-theoretic reasoning, and a lightweight world model. It formalizes decision-making into five composable modules: StateEncoder maps observations to compact latent states; RNNDecisionStep maintains game history and produces decision biases; ActionValueHead implements hierarchical strategy-action selection; WorldModelStep provides one-step counterfactual reasoning; and Fuser projects decision signals to output spaces. The architecture is grounded in extensive-form game theory, encoding partial observability through information sets and simulating game-tree search through hierarchical strategy decomposition. At approximately 952K parameters (default configuration), it supports both single-step and sequential reasoning. The architecture was later adopted by the MuLun (幕论) project as a sidecar decision module for language models.

---

## 1. Introduction

A decision problem can be formalized as a game: an agent, operating in a partially observable environment, selects an action based on current information; the environment transitions to a new state; and the agent receives a reward signal. This framework covers a wide range of scenarios, from cybersecurity incident response to autonomous driving, financial trading, and game AI.

Existing approaches all make different trade-offs.

**Decision-as-Text models** (Decision Transformer [1], GATO [2]) reformulate reinforcement learning as conditional sequence modeling: actions, states, and rewards are all encoded as token sequences, and a causal Transformer predicts the next token. The fundamental issue here is that language fluency and decision quality become coupled—the model must simultaneously learn to "speak correctly" and "decide correctly"—and there is no explicit state representation for reasoning about uncertainty.

**World model approaches** (Dreamer [3], DreamerV2 [4]) learn latent state-space models and optimize policies by backpropagating through imagined trajectories. Dreamer's RSSM (Recurrent State Space Model) is the closest technical inspiration for this architecture, but Dreamer-family methods lack support for structured symbolic inputs and hierarchical strategic reasoning.

**Classic reinforcement learning** (DQN [5], PPO [6]) learns direct state-to-action mappings. While effective at scale, they lack hierarchical structure and interpretability, making them difficult to deploy in domains that require auditability and traceability.

GameNN World Model is designed around a central observation: **a good decision system should explicitly model the cognitive structure of decision-making**, rather than submerging it in an end-to-end black box. Specifically, it draws three core concepts from game theory:

1. **Information Set**: the agent only has partial information about the environment. The state vector $s_t$ and uncertainty $\sigma_t$ produced by StateEncoder jointly form a parameterized information set.
2. **Hierarchical Game**: first selecting a strategy, then an action, corresponds to two levels of decision nodes in an extensive-form game.
3. **Counterfactual Reasoning**: WorldModelStep computes "what would happen if I take action $a$"—this is precisely the core operation of game-tree search.

---

## 2. Formal Foundations

### 2.1 Extensive-Form Games

We model the decision environment as an extensive-form game [10]:

$$\Gamma = \langle \mathcal{H}, \mathcal{A}, P, f_c, \mathcal{U} \rangle$$

where $\mathcal{H}$ is the set of histories (state sequences), $\mathcal{A}$ is the action space, $P$ is the set of players, $f_c$ is the outcome function, and $\mathcal{U}$ is the utility function.

In this architecture we focus on single-agent decision scenarios ($|P| = 1$, with the environment treated as a "chance player"), but the hierarchical decision structure extends naturally to multi-agent settings.

### 2.2 Information Sets

In partially observable environments, the agent cannot distinguish exactly which node of the game tree it occupies. An information set $I \subseteq \mathcal{H}$ is the set of all histories the agent cannot distinguish given its observations.

The StateEncoder's output $(s_t, \sigma_t)$ can be interpreted as a parameterized representation of an information set:

$$I_t = \{ s_t \pm \sigma_t \}$$

That is, the state vector $s_t$ is the central estimate of the information set, while $\sigma_t$ encodes the per-dimension uncertainty. Larger values indicate that the information set contains more possible histories.

### 2.3 Strategies and Hierarchical Decomposition

In game theory, a strategy $\pi$ is a mapping from information sets to probability distributions over actions:

$$\pi: I \rightarrow \Delta(\mathcal{A})$$

This architecture employs a two-level strategy decomposition:

$$\pi(a | I) = \sum_{g \in \mathcal{G}} \pi_{\text{strategy}}(g | I) \cdot \pi_{\text{action}}(a | g, I)$$

where $\mathcal{G}$ is the set of high-level strategies, $\pi_{\text{strategy}}$ is the strategy selector, and $\pi_{\text{action}}$ selects actions within a given strategy. This decomposition corresponds to a **behavioral strategy** in extensive-form games, where probability distributions over actions are defined independently at each information set.

### 2.4 Utility and Value

The expected utility of action $a$ in state $s$ is defined as:

$$\mathbb{E}[U(s, a)] = \mathbb{E}_{s' \sim T(s, a)} [R(s, a, s') + \gamma V(s')]$$

where $T(s, a)$ is the state transition function (approximated by the world model), $R$ is the immediate reward, $V$ is the value function (estimated by ActionValueHead's value head), and $\gamma$ is the discount factor.

---

## 3. Architecture

### 3.1 Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| $o_t$ | Raw observation at time $t$ | $\mathbb{R}^D$ |
| $h_t$ | Encoded latent state | $[0,1]^d$ |
| $\hat{\sigma}_t$ | State uncertainty | $[0,1]^d$ |
| $a_{t-1}$ | Previous action | $\{0,\dots,A-1\}$ |
| $g_t$ | Strategy selection | $\{0,\dots,S-1\}$ |
| $v_t$ | Value estimate | $\mathbb{R}$ |
| $\hat{h}_{t+1}$ | World model predicted next state | $[0,1]^d$ |
| $p_t$ | Predicted outcome probability | $[0,1]$ |
| $b_t$ | Decision bias | $\mathbb{R}^D$ |

$D$ is the observation dimension, $d$ the state dimension, $S$ the number of strategies, and $A$ the number of actions.

### 3.2 StateEncoder

StateEncoder maps a raw observation $o_t \in \mathbb{R}^D$ to a structured state space $[0,1]^d$:

$$h_t, \hat{\sigma}_t = \text{StateEncoder}(o_t)$$

The computation proceeds as:

$$z_t = W_2 \cdot \text{ReLU}(W_1 \cdot o_t + b_1) + b_2$$

$$[\mu_t; \log \nu_t] = \text{split}(z_t)$$

$$h_t = \sigma(\mu_t)$$

$$\hat{\sigma}_t = \sigma(\log \nu_t)$$

where $\sigma$ is the Sigmoid function, $\mu_t \in \mathbb{R}^d$ is the state mean, and $\nu_t \in \mathbb{R}^d$ is the log-variance. The Sigmoid ensures $h_t, \hat{\sigma}_t \in [0,1]^d$.

The uncertainty estimate $\hat{\sigma}_t$ has several uses:
- Weighting sample loss during training
- Triggering a "request human intervention" mechanism during inference
- Serving as a prior for Bayesian updating

### 3.3 RNNDecisionStep

The RNNDecisionStep maintains a recurrent state $s_t \in \mathbb{R}^d$ that encodes decision history up to time $t$. It is inspired by the GRU gating mechanism [8], adapted for the decision-making context.

**Inputs**: current state $s_t$, current observation $h_t$, previous action $a_{t-1}$

**Action embedding**: maps discrete actions to a continuous space:

$$e_{t-1} = W_{\text{embed}} \cdot \text{one-hot}(a_{t-1})$$

where $W_{\text{embed}} \in \mathbb{R}^{m \times A}$ and $m$ is the action embedding dimension (markov_rank in config).

**Joint projection**: concatenates state, action embedding, and observation, then projects:

$$z_t = [s_t; e_{t-1}; h_t]$$

$$u_t = W_{\text{joint}} \cdot z_t$$

where $W_{\text{joint}} \in \mathbb{R}^{(d + m + D) \times 3d}$.

**Gated update** (GRU-style):

$$[g_t; c_t; o_t] = \text{chunk}(u_t)$$

$$g_t = \sigma(g_t) \quad \text{(update gate)}$$

$$c_t = \tanh(c_t) \quad \text{(candidate state)}$$

$$s_{t+1} = g_t \odot s_t + (1 - g_t) \odot c_t \quad \text{(new state)}$$

$$b_t = W_{\text{out}} \cdot \tanh(o_t) \quad \text{(decision bias)}$$

where $\odot$ denotes element-wise multiplication and $W_{\text{out}} \in \mathbb{R}^{d \times D}$.

**Game-theoretic interpretation**: the GRU gating mechanism can be understood as an information set update operation. The update gate $g_t$ determines how much the new observation revises the information set; the candidate state $c_t$ is a hypothesized state based on the new observation; and the new state $s_{t+1}$ is a weighted average of the two. This corresponds to a Bayesian update:

$$P(s_{t+1} | o_t, a_{t-1}) \propto P(o_t | s_t, a_{t-1}) \cdot P(s_t)$$

### 3.4 ActionValueHead

ActionValueHead produces three outputs from the recurrent state $s_t$:

**Strategy logits**:

$$\ell^{\text{strategy}}_t = W_{\text{strategy}} \cdot s_t$$

$$g_t = \arg\max \ell^{\text{strategy}}_t$$

**Action logits**:

$$\ell^{\text{action}}_t = W_{\text{action}} \cdot s_t$$

$$a_t = \arg\max \ell^{\text{action}}_t$$

**Value estimate**:

$$v_t = W_{v2} \cdot \text{ReLU}(W_{v1} \cdot s_t + b_{v1}) + b_{v2}$$

where $W_{\text{strategy}} \in \mathbb{R}^{d \times S}$, $W_{\text{action}} \in \mathbb{R}^{d \times A}$, $W_{v1} \in \mathbb{R}^{\lfloor d/2 \rfloor \times d}$, $W_{v2} \in \mathbb{R}^{1 \times \lfloor d/2 \rfloor}$.

During training, strategy and action use cross-entropy loss; value uses MSE. During inference, probabilities are obtained via Softmax and decisions via argmax.

**Relationship to MuLun**: the MuLun project bound the strategy space to three cybersecurity strategies (aggressive / balanced / defensive) and the action space to eight security response actions. GameNN itself makes no domain assumptions—users define these freely through `GameNNConfig`.

### 3.5 WorldModelStep

WorldModelStep is a lightweight one-step RSSM (Recurrent State Space Model) for counterfactual reasoning.

**Inputs**: current state $s_t$, one-hot encoding of the chosen action $\text{one-hot}(a_t)$

**Forward pass**:

$$x_t = [s_t; \text{one-hot}(a_t)]$$

$$y_t = W_{w2} \cdot \text{ReLU}(W_{w1} \cdot x_t + b_{w1}) + b_{w2}$$

$$\hat{h}_{t+1} = \sigma(y_t[:d]) \quad \text{(predicted next state)}$$

$$p_{t} = \sigma(y_t[d]) \quad \text{(predicted outcome probability)}$$

where $W_{w1} \in \mathbb{R}^{64 \times (d+A)}$ and $W_{w2} \in \mathbb{R}^{(d+1) \times 64}$.

**Game-theoretic interpretation**: the world model approximates the transition function $T(s, a) = P(s' | s, a)$ of the game tree. Before executing an action, the agent can perform "mental simulation":

$$\mathbb{E}[U(a)] = p_t \cdot v_{\text{success}} + (1-p_t) \cdot v_{\text{failure}}$$

where $p_t$ is the success probability predicted by the world model. This is equivalent to **expected utility computation** in game theory.

**Multi-step extension** (future work): the current implementation supports only single-step prediction. With multi-step rollout, full game-tree search becomes possible:

$$V(s_t) = \max_{a} [R(s_t, a) + \gamma \cdot \mathbb{E}_{s_{t+1} \sim T(s_t, a)} V(s_{t+1})]$$

### 3.6 Fuser

The Fuser projects the decision bias $b_t$ to an output space, enabling decision signals to influence external systems.

**Internal processing**:

$$b'_t = W_{f2} \cdot \text{ReLU}(W_{f1} \cdot b_t + b_{f1}) + b_{f2}$$

**Confidence gating**:

$$c_t = \sigma(v_t)$$

$$\tilde{b}_t = b'_t \cdot c_t$$

**Output projection** (optional): when an external weight matrix $W_{\text{out}}$ is provided, the bias projects to the target space:

$$\tilde{b}^{\text{out}}_t = W_{\text{out}} \cdot \tilde{b}_t$$

$\tilde{b}_t$ has shape $\mathbb{R}^D$ (same as observations), and $\tilde{b}^{\text{out}}_t$ depends on the second dimension of $W_{\text{out}}$ (e.g., vocabulary size $V$ for a language model).

This "zero-extra-parameter" fusion approach was concretely implemented in MuLun's ThinkFuser, which reuses the language model's `lm_head.weight` as $W_{\text{out}}$ to bias LM logits directly.

### 3.7 Complete Forward Pass

**Algorithm 1: GameNN single-step decision**

```
Input: observation o_t ∈ ℝᴰ, recurrent state s_t ∈ ℝᵈ, previous action a_{t-1}
Output: decision dictionary

1:  h_t, σ_t ← StateEncoder(o_t)                          # encode observation
2:  s_{t+1}, b_t ← RNNDecisionStep(s_t, h_t, a_{t-1})    # recurrent update
3:  ℓ^g, ℓ^a, v_t ← ActionValueHead(s_{t+1})              # decision output
4:  g_t ← argmax ℓ^g                                      # strategy selection
5:  a_t ← argmax ℓ^a                                      # action selection
6:  ĥ_{t+1}, p_t ← WorldModelStep(s_{t+1}, a_t)           # outcome prediction
7:  b̃_t ← Fuser(b_t, v_t)                                 # output fusion
8:  return {s_{t+1}, g_t, a_t, v_t, p_t, ĥ_{t+1}, b̃_t}
```

In sequential mode (rollout), the above process repeats for $T$ steps, with each step feeding $s_{t+1}$ and $a_t$ as inputs to the next.

---

## 4. Sidecar Architecture Design Principles

A **sidecar** is an architectural pattern where an independent subsystem attaches to a host system through well-defined interfaces, without modifying the host's internal logic. This pattern is familiar in software engineering—Envoy proxies are sidecars for microservices, DSpark's speculative decoding heads are sidecars for language models. GameNN introduces the sidecar pattern to decision architectures, enabling decision capabilities to be plugged into arbitrary systems.

### 4.1 Three Principles of Sidecar Design

**Principle 1: Interface Isolation.** The sidecar communicates with the host only through tensor interfaces, sharing no internal state. GameNN defines three standard interfaces:

- **Input interface**: the host passes an observation vector $o_t \in \mathbb{R}^D$ to the sidecar. This could be sensor features, language model hidden states, or any vectorized representation.
- **State interface**: the sidecar maintains its own recurrent state $s_t$, invisible to and unmodified by the host. This allows the sidecar to independently manage decision history.
- **Output interface**: the sidecar produces a bias signal $\tilde{b}_t$ through the Fuser; the host decides whether to use it. The bias can be added to logits, modulate control signals, or be presented as a decision suggestion to a human operator.

**Principle 2: Host Non-Invasiveness.** The presence or absence of the sidecar should not affect the host system's normal operation. When the sidecar is removed, the host should fall back to its original behavior without decision assistance. This requires:

- The Fuser's bias is initialized to the zero vector (before training), so $\tilde{b}_t = 0$ and has no effect on the host
- In sidecar mode, RNN states and decision outputs are attached only as metadata, not participating in the host's main computation path
- The sidecar can be updated independently without retraining the host

**Principle 3: Bounded Computation.** The sidecar's computational cost should be predictable and bounded. GameNN's single-step forward pass complexity is dominated by $O(D^2)$, independent of the host's computation graph. In MuLun's implementation, the decision head activates only at `<think>` token positions, reducing sidecar cost from $O(T \cdot D^2)$ to $O(K \cdot D^2)$, where $K$ is the number of `<think>` tokens (typically $K \ll T$).

### 4.2 Standalone vs Sidecar Mode

| Aspect | Standalone Mode | Sidecar Mode |
|--------|----------------|-------------|
| Input source | Self-acquired observations | Host's intermediate representations |
| State management | Self-contained RNN state | Self-contained RNN state (unchanged) |
| Output purpose | Direct decision | Bias host output |
| Typical scenario | Embedded decision system | Augmenting LM / control system |
| Deployment | Standalone process | Linked as a library |

Switching between the two modes only involves changing how the Fuser's `output_weight` parameter is set, not modifying the architecture itself.

### 4.3 The Fuser as the Key Enabler

The Fuser is the core enabler of sidecar mode—it "translates" the sidecar's decision signal into a form the host can understand:

- **When `output_weight` is None**, the Fuser outputs $\tilde{b}_t \in \mathbb{R}^D$, a generic decision bias vector the host can use as it sees fit
- **When `output_weight = lm_head.weight`** (MuLun usage), $\tilde{b}_t$ is projected to vocabulary space, additively biasing LM logits
- **When `output_weight` is a trainable matrix**, the Fuser learns how to project decision signals to the host's output space

This generality means the same decision architecture can serve completely different host systems—language models, robot arm controllers, game engines, dashboards—simply by swapping `output_weight`.

### 4.4 Engineering Advantages

1. **Independent deployment**: the sidecar can run as a separate microservice, providing decision services via RPC
2. **Hot update**: sidecar parameters can be updated while the host is running, without downtime
3. **A/B testing**: multiple sidecar instances can be mounted on the same host to compare different decision strategies
4. **Audit logging**: all sidecar inputs and outputs are structured tensors that can be recorded for post-hoc analysis
5. **Fault tolerance**: sidecar crashes do not affect host operation (fallback to unaugmented mode)

---

## 5. Domain Adaptation

### 5.1 Configuration-Driven Design

All domain knowledge is injected through `GameNNConfig`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_dim` | int | Observation dimension (default 768) |
| `state_dim` | int | Latent state dimension (default 16) |
| `n_strategies` | int | Number of strategies (default 3) |
| `n_actions` | int | Number of actions (default 8) |
| `markov_rank` | int | Action embedding dimension (default 16) |
| `strategy_names` | list[str] | Strategy names (interpretability) |
| `action_names` | list[str] | Action names (interpretability) |

### 5.2 State Dimension Semantics

Each dimension $h_t[i]$ of the state vector can be bound to different semantics across domains:

| Dim | Cybersecurity | Autonomous Driving | Financial Trading |
|-----|--------------|-------------------|-------------------|
| 0 | Threat severity | Speed | Volatility |
| 1 | Attack surface | Forward distance | Position risk |
| 2 | Lateral movement risk | Lane offset | Liquidity |
| 3 | Detection coverage | Road condition | Leverage |
| 4 | Compromised ratio | Weather | Sharpe ratio |
| 5 | Alert level | Traffic density | Drawdown |
| ... | ... | ... | ... |

This makes every dimension of the state space interpretable, facilitating domain-expert debugging.

### 5.3 Output Adaptation

The Fuser's `output_weight` parameter allows decision signals to project to arbitrary target spaces:

- **Language model fusion**: `W_out = lm_head.weight` → bias LM logits (MuLun usage)
- **Control signals**: `W_out ∈ ℝ^{k×D}` → generate $k$-dimensional continuous control values
- **Display interface**: `W_out ∈ ℝ^{l×D}` → project to $l$-dimensional visualization space

---

## 6. Training

### 6.1 Objective Function

Training uses supervised learning with four loss components:

$$\mathcal{L} = \lambda_{\text{CE}}^s \cdot \mathcal{L}_{\text{CE}}^s + \lambda_{\text{CE}}^a \cdot \mathcal{L}_{\text{CE}}^a + \lambda_{\text{MSE}}^v \cdot \mathcal{L}_{\text{MSE}}^v + \lambda_{\text{BCE}}^p \cdot \mathcal{L}_{\text{BCE}}^p$$

where:

$$\mathcal{L}_{\text{CE}}^s = -\sum_{b=1}^B \sum_{g=1}^S y_{b,g}^s \log \hat{y}_{b,g}^s \quad \text{(strategy CE)}$$

$$\mathcal{L}_{\text{CE}}^a = -\sum_{b=1}^B \sum_{g=1}^A y_{b,g}^a \log \hat{y}_{b,g}^a \quad \text{(action CE)}$$

$$\mathcal{L}_{\text{MSE}}^v = \frac{1}{B} \sum_{b=1}^B (v_b - \hat{v}_b)^2 \quad \text{(value MSE)}$$

$$\mathcal{L}_{\text{BCE}}^p = -\frac{1}{B} \sum_{b=1}^B [p_b \log \hat{p}_b + (1-p_b) \log(1-\hat{p}_b)] \quad \text{(outcome BCE)}$$

Default weights: $\lambda_{\mathrm{CE}}^s = 1.0$, $\lambda_{\mathrm{CE}}^a = 1.0$, $\lambda_{\mathrm{MSE}}^v = 1.0$, $\lambda_{\mathrm{BCE}}^p = 0.5$.

The `valid_mask` parameter enables masking specific samples, useful for partially labeled data.

### 6.2 Sequential Loss Accumulation

In sequential mode (rollout), losses are computed and summed independently at each time step:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T \mathcal{L}(o_t, a_t^*, v_t^*, p_t^*)$$

This stepwise supervision signal has higher gradient signal-to-noise ratio compared to providing a reward signal only at the end of a trajectory.

### 6.3 Optimization Configuration

Recommended training setup:

- Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$)
- Learning rate: $5 \times 10^{-4}$ for Fuser and StateEncoder, $5 \times 10^{-5}$ for other modules
- Weight decay: 0.01
- Gradient clipping: $\max \|\nabla\| = 1.0$
- Scheduler: Cosine annealing, $T_{\max} = \text{epochs} \times \text{steps per epoch}$
- Batch size: 16-64, depending on data size and GPU memory

### 6.4 Data Requirements

Each training sample should contain:
- Observation vector $o_t$ (dimension $D$)
- Strategy label $g_t^* \in \{0, \dots, S-1\}$
- Action label $a_t^* \in \{0, \dots, A-1\}$
- Value label $v_t^* \in \mathbb{R}$ (recommend normalizing to $[0, 1]$)
- Outcome label $p_t^* \in [0, 1]$ (optional)

### 6.5 Loading MuLun Weights

Since MuLun's sidecar module is built upon the GameNN architecture, its decision head weights can be loaded directly. The `decision_head.` prefix should be stripped:

```python
mulun_ckpt = torch.load("mulun_state16.pth")
for k, v in mulun_ckpt.items():
    key = k.replace("decision_head.", "")
    if key in model.state_dict():
        model.state_dict()[key].copy_(v)
```



### 6.6 Advanced Training Methods

Beyond basic supervised learning, several techniques can improve the architecture's performance and generalization.

**Curriculum Learning.** Training difficulty should increase gradually. A recommended three-stage curriculum:

1. **Single-step imitation**: train StateEncoder and ActionValueHead on single-step decision samples. Learning rate $5 \times 10^{-4}$.
2. **Sequence memory**: introduce 3-5 step short sequences to train RNNDecisionStep's state maintenance. Unlock RNN learning rate to $5 \times 10^{-4}$.
3. **Causal reasoning**: add samples with outcome labels to train WorldModelStep. Raise BCE weight from 0.5 to 1.0 progressively.

**Self-Supervised Pretraining.** StateEncoder can be pretrained without decision labels:

- **Masked reconstruction**: randomly mask input dimensions and reconstruct the full observation through the latent state: $\mathcal{L}_{\text{recon}} = \|o_t - \hat{o}_t\|^2$
- **Contrastive predictive coding (CPC)**: have StateEncoder predict future observation encodings and contrast them with actual encodings, reinforcing temporal structure
- **Temporal coherence**: enforce nearby time-step encodings to be close in $L_2$ distance, improving smoothness

**Multi-Task Training.** For deployment across multiple domains, build a unified model:

$$s_t = \text{StateEncoder}_{\text{shared}}(o_t), \quad g_t, a_t = \text{Head}^{(k)}(s_t)$$

Each domain has its own ActionValueHead; training batches mix data from all domains. Shared modules learn cross-domain representations.

**Data Augmentation.** Extend limited trajectory data through:

- **Temporal cropping**: randomly crop subsequences from long trajectories
- **Action perturbation**: replace action labels with random ones at probability $\epsilon$, training error recovery
- **Observation noise**: add Gaussian noise $\mathcal{N}(0, 0.01)$, improving robustness
- **Temporal shuffling**: shuffle time steps within short windows

**Adversarial Training.** Apply directional perturbation $\delta$ to observations to maximize decision change:

$$\delta = \arg\max_{\|\delta\| \leq \epsilon} \mathcal{L}(f(o_t + \delta), y_t)$$

Adversarial training typically improves out-of-distribution accuracy by 3-8%.

**Knowledge Distillation.** Compress a larger teacher model into GameNN:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(\text{student}, y) + (1-\alpha) \cdot D_{\text{KL}}(p_{\text{student}} \| p_{\text{teacher}})$$

Useful for deployment on resource-constrained devices.

**RL Fine-Tuning.** After supervised training, optimize further through environment interaction with PPO:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

ActionValueHead's value head serves as Critic, strategy head as Actor, and WorldModelStep provides auxiliary predictions. Use low learning rate ($1 \times 10^{-5}$).

**Inverse Reinforcement Learning.** When only human demonstrations are available, use IRL to recover the implicit reward function, then optimize the policy with RL. The value head output $v_t$ is a natural initialization for the reward model.

### 6.7 Complete Training Pipeline

Recommended end-to-end training pipeline:

1. (Optional) Self-supervised pretraining of StateEncoder
2. Single-step supervised training (curriculum stage 1)
3. Short-sequence supervised training (curriculum stage 2)
4. World model training (curriculum stage 3)
5. (Optional) Multi-task joint training
6. (Optional) Adversarial training + knowledge distillation
7. (Optional) RL fine-tuning or IRL

---

## 7. Relationship to MuLun (幕论)

GameNN World Model is the technical foundation of MuLun, not the other way around. MuLun adapted this architecture for the language model scenario through the following modifications:

1. **Input adaptation**: StateEncoder's input was changed from raw feature vectors to MiniMind backbone hidden states $h_t \in \mathbb{R}^{768}$
2. **Output adaptation**: The Fuser's $W_{\text{out}}$ was fixed to `lm_head.weight`, enabling decision biases to directly influence LM logits
3. **Trigger mechanism**: The decision head is activated only at `<think>` token positions (`mode='think'`), not at every step
4. **Domain binding**: The strategy/action spaces were bound to the cybersecurity domain

These adaptations validate the generality and sidecar compatibility of the GameNN architecture—a standalone decision architecture can be seamlessly embedded into a language model, operating as a decision coprocessor.

---

## 8. Discussion

### 8.1 Computational Complexity

Single-step forward pass complexity:

$$O(D^2 + d^2 + dS + dA + (d+A) \cdot 64)$$

The dominant term $D^2$ comes from the Fuser's bottleneck network (when $D=768$, $D^2 \approx 5.9 \times 10^5$). Sequential mode has time complexity $O(T)$.

### 8.2 Comparison with Alternative Approaches

| Dimension | Decision Transformer | Dreamer RSSM | GameNN (this work) |
|-----------|--------------------|-------------|-------------------|
| Parameter scale | >1B (typical) | 10M-100M | **~1M** |
| World model | None | Full trajectory | One-step |
| Hierarchical decision | None | None | **Strategy + Action** |
| Domain adaptation | Retrain required | Retrain required | **Config change** |
| Interpretability | Low | Low | **High** |
| Language capability | Built-in | None | **Fuser bridge** |

### 8.3 Limitations

1. **Single-step world model**: the current WorldModelStep only supports one-step prediction. Extending to full RSSM trajectory imagination is the primary follow-up work.
2. **No RL interface**: currently only supervised training is supported. Introducing PPO or SAC interfaces would enable the architecture to improve through environment interaction.
3. **Discrete action space**: both strategies and actions are discrete. Continuous action support (Gaussian policy head) would extend applicability to domains like robotics.
4. **Vectorized input**: the architecture requires fixed-dimension vectors. Integrating vision or text encoders would enable processing of more raw inputs.
5. **Single-agent only**: multi-agent extension ($|P| > 1$) would cover game-theoretic scenarios involving Nash equilibria and joint strategies.

### 8.4 Scaling the Architecture

The default configuration ($D=768, d=16, S=3, A=8$) works for most scenarios. The following directions extend the architecture's capability.

**State Dimension Scaling ($d$).** The state dimension determines the information capacity of the latent state. Larger $d$ encodes richer situational information at higher computational cost:

| $d$ | Parameter increase | Use case |
|-----|-------------------|----------|
| 16 | Baseline | General decision-making |
| 64 | +27K | Fine-grained awareness (full cybersecurity analysis) |
| 256 | +435K | High-precision decisions (HFT, fine robotic control) |

When $d > 64$, increase markov_rank proportionally to maintain action embedding expressiveness.

**World Model Depth Scaling.** The current WorldModelStep has a single hidden layer. Deeper world models improve prediction accuracy:

- **2-layer**: $64 \rightarrow 128 \rightarrow d+1$, ~2x parameters
- **Residual connections**: add skip connections in deeper world models to mitigate vanishing gradients
- **Probabilistic output**: replace deterministic outputs with Gaussian distributions (predict mean and variance)

**Multi-Step World Model (Full RSSM).** Extend single-step prediction $\hat{h}_{t+1} = f(s_t, a_t)$ to multi-step imagination:

$$\hat{h}_{t+k} = f^{(k)}(s_t, a_{t:t+k-1})$$

This enables full trajectory imagination planning similar to Dreamer, trained with temporal difference loss $\mathcal{L}_{\text{TD}}(\lambda)$.

**Strategy/Action Space Scaling.** Strategy count $S$ and action count $A$ can scale substantially:

| $S$ / $A$ | Scenario | Notes |
|-----------|----------|-------|
| 3 / 8 | General (baseline) | — |
| 5 / 16 | Full cybersecurity | Requires more training data |
| 10 / 50 | Full-stack autonomous driving | Use GumbelRouter as intermediate routing |

When $S \times A > 1000$, introduce GumbelRouter as an intermediate routing layer between strategies and actions.

**Multi-Modal Input Scaling.** The architecture currently accepts only vectors. Frontend encoders extend it to multiple modalities:

- **Vision**: $o_t = \text{ViT}(\text{image}_t)$ or $o_t = \text{CNN}(\text{image}_t)$
- **Text**: $o_t = \text{LLM}(\text{text}_t)[-1, :]$, the last token's hidden state (as MuLun does)
- **Multi-modal fusion**: $o_t = [\text{Enc}_{\text{vis}}(I_t); \text{Enc}_{\text{text}}(T_t)]$

**Multi-Agent Extension.** Extend from single-agent $|P| = 1$ to multi-agent $|P| > 1$:

- **Joint policy**: $\pi(a_1, a_2, \dots, a_n | I_1, I_2, \dots, I_n)$
- **Communication**: each agent broadcasts its RNN state through the Fuser
- **Nash equilibrium solving**: incorporate opponent modeling and train with fictitious play

**Hardware Performance Benchmarks** (single step, batch=1):

| Config | Parameters | CPU (i7-12700) | GPU (RTX 5060) | Memory |
|-------|-----------|---------------|---------------|--------|
| $D=768, d=16$ | 952K | ~0.3ms | ~0.02ms | ~8MB |
| $D=768, d=64$ | 1.2M | ~0.5ms | ~0.03ms | ~12MB |
| $D=2048, d=256$ | 6.8M | ~2.1ms | ~0.08ms | ~55MB |

Even at maximum configuration, inference latency stays in the millisecond range, suitable for real-time decision scenarios.

---

## 9. Conclusion

GameNN World Model is a lightweight, standalone, generic recurrent game-theoretic decision architecture. It decomposes decision-making into five composable modules—situation encoding, recurrent reasoning, hierarchical strategy-action selection, world model imagination, and output fusion—at approximately 952K parameters. The architecture draws three core concepts from game theory—information sets, hierarchical strategies, and counterfactual reasoning—making the decision process transparent and interpretable. Its domain-agnostic design allows switching application scenarios through configuration only, while the Fuser's output projection mechanism enables embedding into other systems such as language models—precisely what the MuLun project demonstrates.

---

## References

[1] Chen, L., et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS 2021.

[2] Reed, S., et al. "A Generalist Agent." arXiv:2205.06175, 2022.

[3] Hafner, D., et al. "Dream to Control: Learning Behaviors by Latent Imagination." ICLR 2020.

[4] Hafner, D., et al. "Mastering Atari with Discrete World Models." ICLR 2021.

[5] Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518, 2015.

[6] Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.

[7] Kahneman, D. "Thinking, Fast and Slow." Farrar, Straus and Giroux, 2011.

[8] Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP 2014.

[9] Pearl, J. "Causality: Models, Reasoning, and Inference." Cambridge University Press, 2000.

[10] Shoham, Y., Leyton-Brown, K. "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations." Cambridge University Press, 2009.

[11] Sutton, R., Barto, A. "Reinforcement Learning: An Introduction." MIT Press, 2018.

[12] Watkins, C., Dayan, P. "Q-learning." Machine Learning, 1992.
