# GameNN World Model

**结构化博弈决策的递归神经网络架构** · Game-driven Recurrent Decision Architecture

> **架构创始人**：[@guaidao2](https://github.com/guaidao2) · 玄幕安全团队

---

## 简介 (Overview)

GameNN World Model 是一种**独立的递归博弈决策架构**，将结构化状态表示、分层策略选择和单步世界模型想象融合为统一的神经网络。该架构是本文首次提出的原创设计，已被 MuLun（幕论）项目采纳为语言模型的侧枝决策模块。

## 核心特性 (Key Features)

- **轻量** — 仅 ~952K 参数，可在 CPU 上实时运行
- **独立** — 无需语言模型或外部系统，即插即用
- **领域无关** — 通过配置切换领域，不改代码
- **可解释** — 显式输出状态/策略/动作/置信度/后果预测

## 快速开始 (Quick Start)

```bash
pip install torch>=2.0.0 numpy>=1.24.0
```

```python
from gamenet import GameNNConfig, GameNNModel
import torch

# 配置
config = GameNNConfig(hidden_dim=768, state_dim=16, n_strategies=3, n_actions=8)
model = GameNNModel(config)

# 单步决策
obs = torch.randn(2, 768)
out = model(obs)
print(out["strategy_name"], out["action_name"], out["outcome_prob"])

# 序列决策
obs_seq = torch.randn(2, 5, 768)
out_seq = model.rollout(obs_seq, return_sequence=True)
```

## 架构 (Architecture)

```
observation → StateEncoder → state → RNNDecisionStep → new_state
                                                           ↓
                                             ActionValueHead → strategy + action + value
                                             WorldModelStep → predicted_state + outcome_prob
                                             Fuser → fused_output
```

## 文档 (Documentation)

- [PAPER_ZH.md](./PAPER_ZH.md) — 中文论文
- [PAPER_EN.md](./PAPER_EN.md) — English paper
- [examples/security_demo.py](./examples/security_demo.py) — 网络安全领域示例

## 引用 (Citation)

```bibtex
@misc{gamenet2026,
  title = {GameNN World Model: A Standalone Recurrent Decision Architecture},
  author = {XuanMu Security Team},
  year = {2026},
}
```
