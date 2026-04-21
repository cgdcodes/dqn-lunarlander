"""
DQN Agent for LunarLander-v3
Assignment 1 — Deep Q-Network Implementation
============================================================
Parts covered:
  A – Environment setup & random-policy baseline
  B – DQN: ReplayBuffer, Q-Network, epsilon-greedy, target network
  C – Training loop, evaluation, and learning-curve plots
  D – Hyperparameter sweep helper
"""

import os
import random
import time
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym

# ─────────────────────────────────────────────
# 0. Reproducibility helper
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# A. Environment setup & random baseline
# ─────────────────────────────────────────────

def make_env(seed: int = 42) -> gym.Env:
    """Create and seed LunarLander-v3."""
    env = gym.make("LunarLander-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def random_policy_baseline(n_episodes: int = 200, seed: int = 42) -> List[float]:
    """
    Part A: Evaluate a uniformly-random policy.
    Returns list of episode rewards.
    """
    env = make_env(seed)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        if (ep + 1) % 50 == 0:
            print(f"  [Random] Episode {ep+1}/{n_episodes}  "
                  f"Avg reward (last 50): {np.mean(rewards[-50:]):.1f}")
    env.close()
    return rewards


# ─────────────────────────────────────────────
# B. DQN Components
# ─────────────────────────────────────────────

# ── B-1. Replay Buffer ──────────────────────

class ReplayBuffer:
    """
    Uniform Experience Replay Buffer.
    Stores (s, a, r, s', done) tuples and samples random mini-batches.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.buffer: deque = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action: int, reward: float,
             next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(actions,                dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards,                dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,                  dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ── B-2. Q-Network ──────────────────────────

class QNetwork(nn.Module):
    """
    Fully-connected Q-network.
    Architecture: input → [hidden_dims] → output (n_actions)
    Uses ReLU activations; no activation on the output layer.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── B-3. Epsilon-greedy policy ──────────────

class EpsilonGreedy:
    """
    Linearly decays epsilon from `eps_start` to `eps_end`
    over `decay_steps` environment steps.
    """

    def __init__(self, eps_start: float, eps_end: float, decay_steps: int):
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.decay_steps = decay_steps
        self.step_count  = 0

    @property
    def epsilon(self) -> float:
        fraction = min(self.step_count / self.decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, q_values: torch.Tensor, n_actions: int) -> int:
        self.step_count += 1
        if random.random() < self.epsilon:
            return random.randrange(n_actions)
        return int(q_values.argmax(dim=-1).item())


# ─────────────────────────────────────────────
# Hyperparameter dataclass
# ─────────────────────────────────────────────

@dataclass
class HParams:
    # Environment
    seed: int = 42
    # Replay buffer
    buffer_capacity: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000     # steps before first gradient update
    # Network
    hidden_dims: Tuple[int, ...] = (256, 256)
    # Training
    total_timesteps: int = 500_000
    gamma: float = 0.99
    lr: float = 1e-3
    # Target network
    target_update_freq: int = 500    # hard update every N steps
    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000
    # Evaluation
    eval_freq: int = 10_000          # evaluate every N steps
    eval_episodes: int = 10
    # Saving
    save_dir: str = "checkpoints"
    run_name: str = "dqn_default"


# ─────────────────────────────────────────────
# C. DQN Agent
# ─────────────────────────────────────────────

class DQNAgent:
    """Full DQN agent (Mnih et al., 2015)."""

    def __init__(self, hp: HParams):
        self.hp = hp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        set_seed(hp.seed)

        # Environment (training)
        self.env = make_env(hp.seed)
        state_dim  = self.env.observation_space.shape[0]   # 8
        action_dim = self.env.action_space.n               # 4

        # Networks
        self.q_net      = QNetwork(state_dim, action_dim, hp.hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hp.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=hp.lr)

        # Replay buffer
        self.buffer = ReplayBuffer(hp.buffer_capacity, self.device)

        # Exploration schedule
        self.explorer = EpsilonGreedy(hp.eps_start, hp.eps_end, hp.eps_decay_steps)

        # Logging
        self.ep_rewards: List[float] = []
        self.eval_steps:   List[int]   = []
        self.eval_rewards: List[float] = []
        self.losses:       List[float] = []

    # ── Core methods ────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return self.explorer.select_action(q_values, self.env.action_space.n)

    def update(self) -> float:
        """Sample a mini-batch and perform one gradient step."""
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.hp.batch_size)

        # Current Q-values: Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ · max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.hp.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """Hard copy of online weights → target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Training loop ───────────────────────

    def train(self):
        hp = self.hp
        os.makedirs(hp.save_dir, exist_ok=True)

        obs, _ = self.env.reset(seed=hp.seed)
        ep_reward = 0.0
        best_eval = -np.inf
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  Training DQN on LunarLander-v3")
        print(f"  Run: {hp.run_name}  |  Device: {self.device}")
        print(f"  Total timesteps: {hp.total_timesteps:,}")
        print(f"{'='*60}\n")

        for step in range(1, hp.total_timesteps + 1):
            # ── Select & execute action ──
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.buffer.push(obs, action, reward, next_obs, float(terminated))
            obs = next_obs
            ep_reward += reward

            if done:
                self.ep_rewards.append(ep_reward)
                obs, _ = self.env.reset()
                ep_reward = 0.0

            # ── Learning update ──
            if (len(self.buffer) >= hp.learning_starts and
                    len(self.buffer) >= hp.batch_size):
                loss = self.update()
                self.losses.append(loss)

            # ── Target network sync ──
            if step % hp.target_update_freq == 0:
                self.sync_target()

            # ── Evaluation ──
            if step % hp.eval_freq == 0:
                mean_eval = self.evaluate(hp.eval_episodes)
                self.eval_steps.append(step)
                self.eval_rewards.append(mean_eval)

                recent_train = (np.mean(self.ep_rewards[-50:])
                                if len(self.ep_rewards) >= 50 else
                                np.mean(self.ep_rewards) if self.ep_rewards else 0.0)

                elapsed = time.time() - start_time
                print(f"Step {step:>7,} | ε={self.explorer.epsilon:.3f} | "
                      f"Train(50ep)={recent_train:6.1f} | "
                      f"Eval={mean_eval:6.1f} | "
                      f"Time={elapsed/60:.1f}min")

                if mean_eval > best_eval:
                    best_eval = mean_eval
                    self.save(os.path.join(hp.save_dir, f"{hp.run_name}_best.pt"))
                    print(f"  ↑ New best eval reward: {best_eval:.1f}  (model saved)")

        self.env.close()
        # Always save final weights
        self.save(os.path.join(hp.save_dir, f"{hp.run_name}_final.pt"))
        print(f"\nTraining complete. Best eval reward: {best_eval:.1f}")
        return self

    # ── Evaluation ──────────────────────────

    def evaluate(self, n_episodes: int = 20, render: bool = False) -> float:
        """Run n episodes with a greedy policy; return mean reward."""
        render_mode = "human" if render else None
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        total_rewards = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = int(self.q_net(state_t).argmax(dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                done = terminated or truncated
            total_rewards.append(ep_reward)

        env.close()
        return float(np.mean(total_rewards))

    # ── Save / Load ─────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hp": asdict(self.hp),
            "ep_rewards": self.ep_rewards,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
        }, path)

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hp_dict = checkpoint["hp"]
        hp_dict["hidden_dims"] = tuple(hp_dict["hidden_dims"])
        hp = HParams(**hp_dict)
        agent = cls(hp)
        agent.q_net.load_state_dict(checkpoint["q_net"])
        agent.target_net.load_state_dict(checkpoint["target_net"])
        agent.ep_rewards   = checkpoint.get("ep_rewards", [])
        agent.eval_steps   = checkpoint.get("eval_steps", [])
        agent.eval_rewards = checkpoint.get("eval_rewards", [])
        return agent


# ─────────────────────────────────────────────
# C. Plotting helpers
# ─────────────────────────────────────────────

def smooth(values: List[float], window: int = 50) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(agent: DQNAgent, save_path: str = "training_curves.png"):
    """
    Part C: Plot (1) smoothed episode rewards, (2) eval rewards, (3) TD loss.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"DQN Training — {agent.hp.run_name}", fontsize=14, fontweight="bold")

    # ── Panel 1: Training rewards ────────────
    ax = axes[0]
    ep_rewards = np.array(agent.ep_rewards)
    if len(ep_rewards) > 0:
        ax.plot(ep_rewards, alpha=0.25, color="steelblue", linewidth=0.8,
                label="Episode reward")
        if len(ep_rewards) >= 50:
            ax.plot(range(49, len(ep_rewards)),
                    smooth(ep_rewards, 50),
                    color="steelblue", linewidth=2.0, label="Moving avg (50)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.2,
               label="Solved threshold (200)")
    ax.axhline(0,   color="gray",  linestyle=":",  linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Episode Rewards")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Evaluation rewards ──────────
    ax = axes[1]
    if agent.eval_steps:
        ax.plot(agent.eval_steps, agent.eval_rewards,
                marker="o", markersize=4, color="darkorange",
                linewidth=1.8, label=f"Eval ({agent.hp.eval_episodes} eps)")
        ax.axhline(200, color="green", linestyle="--", linewidth=1.2,
                   label="Solved threshold (200)")
        ax.fill_between(agent.eval_steps, agent.eval_rewards,
                        alpha=0.15, color="darkorange")
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title("Evaluation Reward vs. Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: TD loss ─────────────────────
    ax = axes[2]
    if agent.losses:
        losses = np.array(agent.losses)
        ax.plot(losses, alpha=0.2, color="crimson", linewidth=0.5)
        if len(losses) >= 500:
            ax.plot(range(499, len(losses)),
                    smooth(losses, 500),
                    color="crimson", linewidth=2.0)
    ax.set_xlabel("Update Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("TD Loss (MSE)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_path}")
    plt.close()


def plot_random_vs_dqn(random_rewards: List[float],
                       dqn_rewards: List[float],
                       save_path: str = "comparison.png"):
    """Compare random baseline vs trained DQN evaluation rewards."""
    fig, ax = plt.subplots(figsize=(9, 5))

    def _violin_or_bar(data, pos, color, label):
        ax.violinplot(data, positions=[pos], widths=0.6,
                      showmedians=True, showextrema=True)
        # Overlay mean marker
        ax.scatter([pos], [np.mean(data)], color=color, zorder=5,
                   s=80, label=f"{label}  μ={np.mean(data):.1f}")

    _violin_or_bar(random_rewards, 1, "steelblue", "Random Policy")
    _violin_or_bar(dqn_rewards,    2, "darkorange", "DQN (trained)")
    ax.axhline(200, color="green", linestyle="--", linewidth=1.2,
               label="Solved threshold (200)")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Random Policy", "Trained DQN"], fontsize=11)
    ax.set_ylabel("Episode Reward")
    ax.set_title("Random Policy Baseline vs. Trained DQN")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_path}")
    plt.close()


def plot_epsilon_schedule(hp: HParams, save_path: str = "epsilon_schedule.png"):
    """Visualise the exploration schedule."""
    steps = np.arange(0, hp.total_timesteps + 1, 1000)
    eps_vals = np.clip(
        hp.eps_start + (hp.eps_end - hp.eps_start) * steps / hp.eps_decay_steps,
        hp.eps_end, hp.eps_start
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, eps_vals, color="purple", linewidth=2)
    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Epsilon")
    ax.set_title("Epsilon-Greedy Exploration Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# D. Hyperparameter sweep
# ─────────────────────────────────────────────

def hyperparameter_sweep():
    """
    Part D: Run a small grid of hyperparameter configurations and
    report final eval rewards. Adjust configs as needed.
    """
    configs = [
        # (run_name,              lr,    gamma, hidden_dims,    batch_size)
        ("dqn_lr1e-3_default",  1e-3,  0.99,  (256, 256),     64),
        ("dqn_lr5e-4",          5e-4,  0.99,  (256, 256),     64),
        ("dqn_lr1e-4",          1e-4,  0.99,  (256, 256),     64),
        ("dqn_gamma_095",       1e-3,  0.95,  (256, 256),     64),
        ("dqn_larger_net",      1e-3,  0.99,  (512, 512),     64),
        ("dqn_batch128",        1e-3,  0.99,  (256, 256),    128),
    ]

    results = {}
    SWEEP_STEPS = 200_000   # shorter runs for sweep; increase for final report

    for name, lr, gamma, hidden, batch in configs:
        print(f"\n{'─'*50}")
        print(f"  Sweep run: {name}")
        hp = HParams(
            run_name=name,
            lr=lr,
            gamma=gamma,
            hidden_dims=hidden,
            batch_size=batch,
            total_timesteps=SWEEP_STEPS,
            eval_freq=20_000,
            save_dir="checkpoints/sweep",
        )
        agent = DQNAgent(hp).train()
        final_eval = agent.evaluate(20)
        results[name] = final_eval
        print(f"  Final eval ({name}): {final_eval:.1f}")

    # ── Summary bar chart ────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(results.keys())
    values = [results[n] for n in names]
    colors = ["steelblue" if v < 200 else "green" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(200, color="green", linestyle="--", linewidth=1.4,
               label="Solved threshold (200)")
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlabel("Mean Eval Reward (20 episodes)")
    ax.set_title("Hyperparameter Sweep — Final Eval Rewards")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("checkpoints/sweep/sweep_results.png", dpi=150, bbox_inches="tight")
    print("\n[Sweep] Results chart saved.")

    # Save JSON
    with open("checkpoints/sweep/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Sweep] Results JSON saved.")
    return results


# ─────────────────────────────────────────────
# Video / GIF recording helper
# ─────────────────────────────────────────────

def record_episodes(agent: DQNAgent, n: int = 5,
                    out_dir: str = "videos"):
    """
    Record n greedy episodes using gymnasium's RecordVideo wrapper.
    Produces MP4 files in out_dir/.
    Requires: pip install moviepy
    """
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=out_dir,
        episode_trigger=lambda ep: True,   # record every episode
        name_prefix=agent.hp.run_name,
    )

    for ep in range(n):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32,
                                   device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action = int(agent.q_net(state_t).argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        print(f"  [Video] Episode {ep+1}: reward = {ep_reward:.1f}")

    env.close()
    print(f"[Video] {n} episode(s) saved to '{out_dir}/'")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN for LunarLander-v3")
    parser.add_argument("--mode", choices=["train", "eval", "sweep", "baseline"],
                        default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for eval/resume")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--run_name",  type=str,  default="dqn_default")
    args = parser.parse_args()

    # ── Baseline ──
    if args.mode == "baseline":
        print("Running random-policy baseline...")
        rewards = random_policy_baseline(n_episodes=200, seed=args.seed)
        print(f"\nRandom Policy — Mean: {np.mean(rewards):.1f} | "
              f"Std: {np.std(rewards):.1f} | "
              f"Min: {np.min(rewards):.1f} | Max: {np.max(rewards):.1f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rewards, alpha=0.6, color="steelblue")
        ax.axhline(np.mean(rewards), color="red", linestyle="--",
                   label=f"Mean: {np.mean(rewards):.1f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.set_title("Random Policy Baseline — LunarLander-v3")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("random_baseline.png", dpi=150)
        print("[Plot] Saved random_baseline.png")

    # ── Train ──
    elif args.mode == "train":
        hp = HParams(
            seed=args.seed,
            total_timesteps=args.timesteps,
            lr=args.lr,
            run_name=args.run_name,
        )
        agent = DQNAgent(hp).train()

        # Post-training analysis
        plot_training_curves(agent, save_path=f"checkpoints/{args.run_name}_curves.png")
        plot_epsilon_schedule(hp, save_path=f"checkpoints/{args.run_name}_epsilon.png")

        # Compare vs random baseline
        print("\nCollecting random baseline for comparison...")
        rand_rewards = random_policy_baseline(200)
        dqn_eval     = [agent.evaluate(1) for _ in range(100)]
        plot_random_vs_dqn(rand_rewards, dqn_eval,
                           save_path=f"checkpoints/{args.run_name}_comparison.png")

        # Record videos
        record_episodes(agent, n=5, out_dir=f"checkpoints/{args.run_name}_videos")

    # ── Eval only ──
    elif args.mode == "eval":
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for eval mode.")
        else:
            agent = DQNAgent.load(args.checkpoint)
            mean_r = agent.evaluate(20)
            print(f"Greedy eval over 20 episodes: {mean_r:.1f}")
            record_episodes(agent, n=5)

    # ── Sweep ──
    elif args.mode == "sweep":
        hyperparameter_sweep()
