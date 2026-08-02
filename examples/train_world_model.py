"""
GameNN 世界模型真实目标训练脚本
（friend-audit 修复验证：世界模型必须被真实数据训练，而非随机 target）

环境：5x5 网格 + 4 动作（上下左右），边界阻挡。
真实监督信号：
  - target_outcome    = 1.0 if 移动成功 else 0.0
  - target_next_state = 实际到达的下一状态（MSE 自监督）

运行：python examples/train_world_model.py --steps 2000
验证：world loss（outcome + next_state）单调下降 → 世界模型真实学习
"""
import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gamenet.config import GameNNConfig
from gamenet.model import GameNNModel

ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up/down/left/right


class GridEnv:
    """5x5 网格。观测 = hidden_dim 维（位置 onehot 填前 25 维）"""
    def __init__(self, size=5, seed=0, obs_dim=768):
        self.size = size
        self.obs_dim = obs_dim
        self.rng = np.random.RandomState(seed)
        self.pos = [0, 0]

    def reset(self):
        self.pos = [self.rng.randint(self.size), self.rng.randint(self.size)]
        return self._obs()

    def _obs(self):
        o = np.zeros(self.obs_dim)
        o[self.pos[0] * self.size + self.pos[1]] = 1.0
        return o

    def step(self, a):
        """返回 (obs', reward, moved)"""
        dx, dy = ACTIONS[a]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        moved = 0 <= nx < self.size and 0 <= ny < self.size
        if moved:
            self.pos = [nx, ny]
        return self._obs(), 1.0 if moved else 0.0, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = GameNNConfig(hidden_dim=768, state_dim=16, n_actions=4)
    model = GameNNModel(cfg)
    # 隔离验证：只训练 world_model（其余冻结——state/next_enc 目标稳定，
    # 避免移动目标问题；这也正是 friend-audit 指控的核心：世界模型能否学）
    for p in model.parameters():
        p.requires_grad = False
    for p in model.world_model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.world_model.parameters(), lr=args.lr)
    env = GridEnv(seed=args.seed)

    # 批量收集真实转移 (obs, a, obs_next, moved) → 32 步一批训练（降单样本噪声）
    prev_action = 0
    buf_o, buf_a, buf_on, buf_m = [], [], [], []
    hist, next_hist = [], []
    for t in range(args.steps):
        obs = torch.tensor(env._obs(), dtype=torch.float32)
        a = env.rng.randint(4)
        obs_next, reward, moved = env.step(a)
        buf_o.append(obs)
        buf_a.append(a)
        buf_on.append(torch.tensor(obs_next, dtype=torch.float32))
        buf_m.append(1.0 if moved else 0.0)
        prev_action = a
        if len(buf_o) == 32:
            obs_t = torch.stack(buf_o)
            out = model(obs_t, prev_action=torch.tensor(buf_a))
            with torch.no_grad():
                next_enc, _ = model.state_encoder(torch.stack(buf_on))
            target_outcome = torch.tensor(buf_m)
            losses = model.compute_loss(
                out,
                target_outcome=target_outcome,
                target_next_state=next_enc.detach(),
                outcome_weight=0.5,
                next_state_weight=1.0,
            )
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            hist.append(losses["loss"].item())
            next_hist.append(losses["next_state_loss"].item())
            buf_o, buf_a, buf_on, buf_m = [], [], [], []
            if len(hist) % 10 == 0:
                print(f"  Batch {len(hist):4d} | wm_loss={np.mean(hist[-10:]):.4f} "
                      f"next={np.mean(next_hist[-10:]):.4f}")

    # 验证：后半 vs 前半（next_state 为主——世界模型学会环境转移）
    half = len(hist) // 2
    early, late = np.mean(hist[:half]), np.mean(hist[half:])
    ne, nl = np.mean(next_hist[:half]), np.mean(next_hist[half:])
    print(f"\n总 wm_loss: early={early:.4f} late={late:.4f} ({early / max(late, 1e-9):.2f}x)")
    print(f"next_state: early={ne:.4f} late={nl:.4f} ({ne / max(nl, 1e-9):.2f}x)")
    ok = nl < ne * 0.5
    print(f"判定: {'OK（世界模型真实学习环境动力学）' if ok else 'FAIL（未收敛）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
