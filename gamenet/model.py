"""
GameNN World Model — Main Architecture.

GameNNModel is the complete, standalone decision architecture.
It assembles all core modules into a unified recurrent neural network
for structured decision-making with world model imagination.

This is NOT a sidecar — it is the backbone itself.

Architecture Flow
-----------------
    observation → StateEncoder → state[state_dim]
        ↓
    RNNDecisionStep: (state, observation, prev_action) → new_state, bias
        ↓
    ActionValueHead: new_state → strategy_logits, action_logits, value
        ↓
    WorldModelStep: (new_state, action) → predicted_state, outcome_prob
        ↓
    Fuser: (bias, state, value) → fused_output (for downstream tasks)

Usage
-----
    >>> from gamenet import GameNNConfig, GameNNModel
    >>> config = GameNNConfig(state_dim=16, n_actions=8, hidden_dim=768)
    >>> model = GameNNModel(config)
    >>> obs = torch.randn(4, 768)  # batch=4, dim=768
    >>> out = model(obs)  # single-step decision
    >>> out["strategy_name"]
    ['strategy_0', 'strategy_1', ...]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from .config import GameNNConfig
from .core import (
    StateEncoder,
    GumbelRouter,
    RNNDecisionStep,
    ActionValueHead,
    WorldModelStep,
    Fuser,
)


class GameNNModel(nn.Module):
    """
    GameNN World Model — Complete standalone decision architecture.

    Composable modules:
        1. StateEncoder     — observation → structured state + uncertainty
        2. RNNDecisionStep  — (state, context, action) → new_state + bias
        3. ActionValueHead  — state → strategy / action / value
        4. WorldModelStep   — (state, action) → predicted_state + outcome_prob
        5. Fuser            — (bias, state, value) → fused output

    The architecture supports three operational modes:
        - single-step:  process one observation at a time
        - sequential:   process a sequence, maintaining RNN state across steps
        - rollout:      sequential processing with world model imagination
    """

    def __init__(self, config: GameNNConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.n_strategies = config.n_strategies
        self.n_actions = config.n_actions

        # ── Core modules ──
        self.state_encoder = StateEncoder(config)
        self.rnn_step = RNNDecisionStep(config)
        self.action_value = ActionValueHead(config)
        self.world_model = WorldModelStep(config)
        self.fuser = Fuser(config)

        # Domain labels (for interpretability)
        self.strategy_names = config.strategy_names
        self.action_names = config.action_names

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initialize the recurrent decision state for a new sequence."""
        return self.rnn_step.init_state(batch, device)

    def forward(
        self,
        observation: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
        prev_action: Optional[torch.Tensor] = None,
        output_weight: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> Dict[str, Any]:
        """
        Single decision step.

        Args:
            observation:   [B, hidden_dim] input observation vector
            rnn_state:     [B, state_dim] previous RNN state (zero if None)
            prev_action:   [B] previous action index (zero if None)
            output_weight: [V, D] optional output projection for the fuser
            return_all:    if True, returns full dict; otherwise minimal

        Returns:
            dict with:
                new_rnn_state:  [B, state_dim]
                state:          [B, state_dim]
                uncertainty:    [B, state_dim]
                strategy_probs: [B, n_strategies]
                strategy_idx:   [B]
                strategy_name:  [B] list of names
                action_probs:   [B, n_actions]
                action_idx:     [B]
                action_name:    [B] list of names
                value:          [B, 1]
                outcome_prob:   [B, 1] world model prediction
                predicted_state:[B, state_dim] world model prediction
                fused_output:   [B, hidden_dim] or [B, V]
        """
        B = observation.shape[0]

        # 1. Encode observation → structured state
        state, uncertainty = self.state_encoder(observation)

        # 2. RNN step: integrate action history and produce bias
        if rnn_state is None:
            rnn_state = self.init_state(B, observation.device)
        new_state, decision_bias = self.rnn_step(state, observation, prev_action)

        # 3. Decision heads: strategy, action, value
        decision = self.action_value(new_state)
        strategy_probs = F.softmax(decision["strategy_logits"], dim=-1)
        action_probs = F.softmax(decision["action_logits"], dim=-1)
        strategy_idx = strategy_probs.argmax(dim=-1)
        action_idx = action_probs.argmax(dim=-1)

        # 4. World model: imagine the outcome
        action_onehot = F.one_hot(action_idx, num_classes=self.n_actions).float()
        wm_out = self.world_model(new_state, action_onehot)

        # 5. Fuser: project decision bias for downstream use
        fused = self.fuser(decision_bias, new_state, decision["value"], output_weight)

        result = {
            "new_rnn_state": new_state,
            "state": state,
            "uncertainty": uncertainty,
            "strategy_probs": strategy_probs,
            "strategy_idx": strategy_idx,
            "strategy_name": [self.strategy_names[i] for i in strategy_idx.tolist()],
            "action_probs": action_probs,
            "action_idx": action_idx,
            "action_name": [self.action_names[i] for i in action_idx.tolist()],
            "value": decision["value"],
            "outcome_prob": wm_out["outcome_prob"],
            "predicted_state": wm_out["predicted_state"],
            "fused_output": fused,
        }
        return result

    @torch.inference_mode()
    def rollout(
        self,
        observations: torch.Tensor,
        return_sequence: bool = False,
        output_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Process a full sequence of observations, maintaining RNN state.

        Args:
            observations:   [B, T, hidden_dim] sequence of observations
            return_sequence: if True, returns per-step decisions (T stack)
            output_weight:  optional output projection for the fuser

        Returns:
            dict with the same keys as forward(), but values may be
            sequences [B, T, ...] if return_sequence=True,
            or the last step's output [B, ...] if return_sequence=False.
        """
        B, T, D = observations.shape
        prev_action = None
        rnn_state = self.init_state(B, observations.device)
        steps = []

        for t in range(T):
            obs_t = observations[:, t, :]
            out = self.forward(
                obs_t,
                rnn_state=rnn_state,
                prev_action=prev_action,
                output_weight=output_weight,
            )
            rnn_state = out["new_rnn_state"]
            prev_action = out["action_idx"]
            steps.append(out)

        if return_sequence:
            # Stack per-step outputs
            return {
                k: torch.stack([s[k] for s in steps], dim=1)
                if isinstance(steps[0][k], torch.Tensor)
                else [s[k] for s in steps]
                for k in steps[0].keys()
            }
        else:
            return steps[-1]

    def compute_loss(
        self,
        decision_out: Dict[str, Any],
        target_strategy: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
        target_outcome: Optional[torch.Tensor] = None,
        target_value: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        strategy_weight: float = 1.0,
        action_weight: float = 1.0,
        value_weight: float = 1.0,
        outcome_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised learning losses for the decision architecture.

        Args:
            decision_out: output dict from forward()
            target_strategy: [B] strategy indices
            target_action:   [B] action indices
            target_outcome:  [B] outcome probability targets
            target_value:    [B] value targets
            valid_mask:      [B] bool — which samples have valid targets
            *_weight:        loss component weights

        Returns:
            dict with individual losses and total "loss"
        """
        losses = {}
        total = 0.0

        def _apply_mask(tensor, mask):
            return tensor[mask] if mask is not None else tensor

        # Strategy cross-entropy
        if target_strategy is not None:
            logits = _apply_mask(decision_out["strategy_probs"], valid_mask)
            targets = _apply_mask(target_strategy, valid_mask)
            if targets.numel() > 0:
                loss = F.cross_entropy(logits, targets)
                losses["strategy_loss"] = loss
                total += strategy_weight * loss

        # Action cross-entropy
        if target_action is not None:
            logits = _apply_mask(decision_out["action_probs"], valid_mask)
            targets = _apply_mask(target_action, valid_mask)
            if targets.numel() > 0:
                loss = F.cross_entropy(logits, targets)
                losses["action_loss"] = loss
                total += action_weight * loss

        # Value (confidence) MSE
        if target_value is not None:
            pred = _apply_mask(
                torch.sigmoid(decision_out["value"].squeeze(-1)), valid_mask
            )
            targets = _apply_mask(target_value, valid_mask)
            if targets.numel() > 0:
                loss = F.mse_loss(pred, targets)
                losses["value_loss"] = loss
                total += value_weight * loss

        # World model outcome prediction BCE
        if target_outcome is not None:
            pred = _apply_mask(decision_out["outcome_prob"].squeeze(-1), valid_mask)
            targets = _apply_mask(target_outcome, valid_mask)
            if targets.numel() > 0:
                loss = F.binary_cross_entropy(pred, targets)
                losses["outcome_loss"] = loss
                total += outcome_weight * loss

        losses["loss"] = total
        return losses


__all__ = ["GameNNModel"]
