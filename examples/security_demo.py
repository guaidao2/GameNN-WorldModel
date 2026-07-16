"""
示例：在网络安全领域配置 GameNN World Model。

展示如何将通用架构绑定到具体领域。
"""
import torch
import sys
sys.path.insert(0, "..")

from gamenet import GameNNConfig, GameNNModel


# ── 领域配置 ──────────────────────────────────────────────────
CYBER_STRATEGIES = ["aggressive", "balanced", "defensive"]
CYBER_ACTIONS = [
    "BLOCK_IP",          # 封禁攻击源IP
    "PATCH_VULN",        # 修补漏洞
    "ISOLATE_HOST",      # 隔离受影响主机
    "RESTORE_BACKUP",    # 从备份恢复系统
    "DEEP_SCAN",         # 深度扫描分析攻击范围
    "HUNT_THREATS",      # 主动威胁狩猎
    "DEPLOY_HONEYPOT",   # 部署蜜罐诱捕
    "ESCALATE_INCIDENT", # 升级事件至应急响应团队
]

config = GameNNConfig(
    hidden_dim=768,
    state_dim=16,
    n_strategies=3,
    n_actions=8,
    markov_rank=16,
    strategy_names=CYBER_STRATEGIES,
    action_names=CYBER_ACTIONS,
)

model = GameNNModel(config)
print(f"GameNN World Model — 网络安全领域")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"  状态维度: {config.state_dim}")
print(f"  策略空间: {config.strategy_names}")
print(f"  动作空间: {config.action_names}")
print()

# ── 模拟一批安全态势观测 ────────────────────────────────────
batch_size = 2
observations = torch.randn(batch_size, config.hidden_dim)

out = model(observations)

print("=== 单步决策输出 ===")
for i in range(batch_size):
    print(f"\n样本 {i}:")
    print(f"  策略: {out['strategy_name'][i]}")
    print(f"  动作: {out['action_name'][i]}")
    print(f"  置信度: {torch.sigmoid(out['value'][i]).item():.3f}")
    print(f"  预期遏制概率: {out['outcome_prob'][i].item():.3f}")
    print(f"  预测状态 (前8维): {out['predicted_state'][i, :8].tolist()}")
    print(f"  不确定度 (前8维): {out['uncertainty'][i, :8].tolist()}")

# ── 序列决策 ──────────────────────────────────────────────────
seq_len = 4
obs_seq = torch.randn(batch_size, seq_len, config.hidden_dim)
out_seq = model.rollout(obs_seq, return_sequence=True)

print(f"\n=== 序列决策 ({seq_len} 步) ===")
for i in range(batch_size):
    print(f"\n样本 {i} 决策序列:")
    for t in range(seq_len):
        print(f"  步 {t}: 策略={out_seq['strategy_name'][t][i]}, "
              f"动作={out_seq['action_name'][t][i]}, "
              f"置信度={torch.sigmoid(out_seq['value'][t, i]).item():.3f}")

# ── 演示训练损失计算 ─────────────────────────────────────────
print("\n=== 训练损失计算演示 ===")
target_strategy = torch.randint(0, config.n_strategies, (batch_size,))
target_action = torch.randint(0, config.n_actions, (batch_size,))
target_value = torch.rand(batch_size)
target_outcome = torch.rand(batch_size)

losses = model.compute_loss(
    out,
    target_strategy=target_strategy,
    target_action=target_action,
    target_value=target_value,
    target_outcome=target_outcome,
)

for name, val in losses.items():
    print(f"  {name}: {val.item():.6f}")
