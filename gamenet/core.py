"""
GameNN World Model — Core Neural Modules.

The fundamental building blocks of the GameNN architecture:

    StateEncoder     — Encodes raw observations into structured latent states
    GumbelRouter     — (Optional) Gumbel-Softmax strategy/action router
    RNNDecisionStep  — GRU-style recurrent decision step with action memory
    ActionValueHead  — Produces strategy logits, action logits, and value estimates
    WorldModelStep   — One-step RSSM-inspired world model for outcome prediction
    Fuser            — Projects decision state back to output space

All modules are domain-agnostic and fully configurable via GameNNConfig.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import GameNNConfig


class StateEncoder(nn.Module):
    """
    Encodes a raw observation vector into a structured latent state.

    Uses a VAE-style dual-head: produces both a state mean (posterior)
    and an uncertainty estimate (log-variance), both squashed to [0, 1].

    Architecture
    ------------
        observation → Linear(hidden//2) → ReLU → Linear(state_dim * 2)
        → split → sigmoid(mean) = state, sigmoid(logvar) = uncertainty
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.state_dim * 2),
        )

    def forward(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: [B, D] raw input vector

        Returns:
            state:       [B, state_dim] structured latent state (sigmoid)
            uncertainty: [B, state_dim] per-dimension uncertainty (sigmoid)
        """
        out = self.proj(observation)
        mean, logvar = out.chunk(2, dim=-1)
        state = torch.sigmoid(mean)
        uncertainty = torch.sigmoid(logvar)
        return state, uncertainty


class GumbelRouter(nn.Module):
    """
    Gumbel-Softmax router for discrete choice selection.

    Can be used independently for strategy selection, action selection,
    or any discrete decision point.

    Architecture
    ------------
        input → Linear(hidden) → ReLU → Linear(hidden//2) → ReLU → Linear(n_choices)
    """

    def __init__(self, in_dim: int, n_choices: int, hidden: int = 64):
        super().__init__()
        self.n_choices = n_choices
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_choices),
        )

    def forward(
        self, x: torch.Tensor, tau: float = 1.0, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, in_dim] input features
            tau:  temperature for Gumbel-Softmax
            hard: whether to use hard (straight-through) sampling

        Returns:
            probs:     [B, n_choices] soft probabilities
            argmax:    [B] hard index of the chosen option
        """
        logits = self.net(x)
        if self.training and not hard:
            probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        else:
            probs = F.softmax(logits / max(tau, 0.01), dim=-1)
        return probs, probs.argmax(dim=-1)


class RNNDecisionStep(nn.Module):
    """
    Core recurrent decision step — the "R" in RSSM.

    Maintains a hidden state that carries decision context across time steps.
    Uses a GRU-style gated update:

        z = [state; action_embedding; observation]
        gate = sigmoid(proj_gate(z))
        candidate = tanh(proj_candidate(z))
        new_state = gate * state + (1 - gate) * candidate

    Design inspired by DeepSpec DSpark's RNNHead, repurposed for
    structured sequential decision-making.
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.markov_rank = config.markov_rank

        # Joint projection: [state | action_emb | observation] → gate + candidate + output
        self.joint_proj = nn.Linear(
            config.state_dim + config.markov_rank + config.hidden_dim,
            3 * config.state_dim,
        )

        # Action embedding: maps one-hot action to markov_rank
        self.action_embed = nn.Linear(config.n_actions, config.markov_rank, bias=False)

        # Output bias projection: state → observation space
        self.output_proj = nn.Linear(config.state_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state:       [B, state_dim] current structured state
            observation: [B, hidden_dim] current observation/context
            prev_action: [B] or None — previous action index (zero if None)

        Returns:
            new_state: [B, state_dim] updated state after this decision step
            bias:      [B, hidden_dim] decision bias projected to observation space
        """
        B = state.shape[0]
        device = state.device

        if prev_action is None:
            prev_action = torch.zeros(B, dtype=torch.long, device=device)

        action_emb = self.action_embed(
            F.one_hot(prev_action, num_classes=self.action_embed.in_features).float()
        )

        # GRU-style gated update
        z = torch.cat([state, action_emb, observation], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)

        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.output_proj(torch.tanh(output_raw))

        return new_state, bias

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initialize the recurrent state to zeros for a new decision sequence."""
        return torch.zeros(batch, self.state_dim, device=device)


class ActionValueHead(nn.Module):
    """
    Produces structured decision output from the current state.

    Three parallel heads:
        - strategy_head: selects high-level strategy (discrete)
        - action_head:   selects concrete action (discrete)
        - value_head:    estimates expected value/confidence (scalar)

    This separates "what to do" (strategy) from "how to do it" (action),
    mimicking hierarchical decision-making.
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.n_strategies = config.n_strategies
        self.n_actions = config.n_actions

        self.strategy_net = nn.Linear(config.state_dim, config.n_strategies)
        self.action_net = nn.Linear(config.state_dim, config.n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(config.state_dim, max(config.state_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(config.state_dim // 2, 1), 1),
        )

    def forward(self, state: torch.Tensor) -> dict:
        """
        Args:
            state: [B, state_dim] current structured state

        Returns:
            dict with:
                strategy_logits: [B, n_strategies]
                action_logits:   [B, n_actions]
                value:           [B, 1]
        """
        return {
            "strategy_logits": self.strategy_net(state),
            "action_logits": self.action_net(state),
            "value": self.value_net(state),
        }


class WorldModelStep(nn.Module):
    """
    One-step RSSM-inspired world model.

    Predicts the outcome of taking a given action from the current state:
        - next predicted state
        - outcome/containment probability (a scalar measure of success)

    This enables "what-if" reasoning: the agent can simulate the consequence
    of an action before committing to it.

    Architecture
    ------------
        [state; action_onehot] → Linear(64) → ReLU → Linear(state_dim + 1)
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.n_actions = config.n_actions

        self.net = nn.Sequential(
            nn.Linear(config.state_dim + config.n_actions, config.world_model_hidden),
            nn.ReLU(),
            nn.Linear(config.world_model_hidden, config.state_dim + 1),
        )

    def forward(
        self, state: torch.Tensor, action_onehot: torch.Tensor
    ) -> dict:
        """
        Args:
            state:         [B, state_dim] current state
            action_onehot: [B, n_actions] one-hot encoding of the chosen action

        Returns:
            dict with:
                predicted_state: [B, state_dim] imagined next state
                outcome_prob:    [B, 1] predicted outcome/containment probability
        """
        x = torch.cat([state, action_onehot], dim=-1)
        out = self.net(x)
        predicted_state = torch.sigmoid(out[:, :self.state_dim])
        outcome_prob = torch.sigmoid(out[:, -1:])
        return {
            "predicted_state": predicted_state,
            "outcome_prob": outcome_prob,
        }


class Fuser(nn.Module):
    """
    Projects the decision state and bias back to the observation/output space.

    The Fuser is what makes the decision architecture "closable": it allows
    the decision output to influence downstream processing (e.g. biasing
    language model logits, modulating control signals, or producing interpretable
    decision summaries).

    When an output projection weight is provided (e.g. reusing a language
    model head), the bias is projected to that space for zero-extra-parameter
    fusion. Otherwise, the bias is returned directly.
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.bias_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
        )

    def forward(
        self,
        bias: torch.Tensor,
        state: torch.Tensor,
        value: torch.Tensor,
        output_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            bias:          [B, hidden_dim] decision bias from RNNDecisionStep
            state:         [B, state_dim] current structured state (used for gating)
            value:         [B, 1] confidence/value estimate
            output_weight: [V, D] or [D_out, hidden_dim] optional output projection weight

        Returns:
            fused_bias: [B, hidden_dim] or [B, V] if output_weight provided
        """
        bias = self.bias_proj(bias)  # [B, D]
        if output_weight is not None:
            bias = F.linear(bias, output_weight)  # [B, V]
        confidence = torch.sigmoid(value)
        return bias * confidence


__all__ = [
    "StateEncoder",
    "GumbelRouter",
    "RNNDecisionStep",
    "ActionValueHead",
    "WorldModelStep",
    "Fuser",
]
