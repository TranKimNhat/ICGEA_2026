"""
sim_MARL.py
===========
Training entry-point for MARL + GNN on CIGRE MV microgrid (Design-1 topology event).

Creates:
- MicrogridEnv (pandapower AC PF + SG ODE)
- SharedPolicy (GNN backbone + shared actor + centralized critic)
- MAPPO trainer

How to run (VS Code / PyCharm):
--------------------------------------------------------------
Option A (no terminal): Open sim_MARL.py and press F5 / Run.
  - Edit DEFAULT_MODE / DEFAULT_EPISODES at top of this file.

Option B (terminal):
1) Put these files in the same folder:
   - env.py
   - GNN.py
   - Magent.py
   - sim_MARL.py
   - cigre_mv_microgrid.py  (your network builder)

2) Install deps:
   pip install pandapower torch numpy

3) Train:
   python sim_MARL.py --episodes 500 --device cpu

4) Evaluate (optional):
   python sim_MARL.py --episodes 20 --eval --ckpt checkpoints/mappo_latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# Make relative paths (checkpoints/, logs/) work no matter where you press Run/F5 from
os.chdir(ROOT_DIR)

import time
from typing import Dict, Any

import numpy as np
import torch

from env_v2 import MicrogridEnv, SGParams
from Magent_v2 import SharedPolicy, PolicyConfig, MAPPO, RolloutBuffer

# Your microgrid builder (must be in same folder or in PYTHONPATH)
from cigre_mv_microgrid_v1 import create_cigre_mv_microgrid


# ============================================================
# Quick-run configuration (for VS Code / PyCharm Run or F5)
# ------------------------------------------------------------
# If you press F5 / Run without command-line arguments, this script will
# use the defaults below. You can change DEFAULT_MODE to 'eval' anytime.
# ============================================================
DEFAULT_MODE = 'train'   # 'train' or 'eval'
DEFAULT_EPISODES = 500
DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cpu'

# Training hyperparams (PPO)
DEFAULT_LR = 3e-4
DEFAULT_MINIBATCH = 256
DEFAULT_EPOCHS = 5

# Checkpoint path for eval (and resume training if you want)
DEFAULT_CKPT = 'checkpoints/mappo_latest.pt'

def _namespace_from_defaults() -> argparse.Namespace:
    """Build argparse-like args from constants above (no CLI needed)."""
    return argparse.Namespace(
        eval=(DEFAULT_MODE == 'eval'),
        episodes=DEFAULT_EPISODES,
        seed=DEFAULT_SEED,
        device=DEFAULT_DEVICE,
        lr=DEFAULT_LR,
        minibatch=DEFAULT_MINIBATCH,
        epochs=DEFAULT_EPOCHS,
        ckpt=DEFAULT_CKPT,
        log_every=10,
        ckpt_every=50,
    )


def make_env(seed: int = 0) -> MicrogridEnv:
    # --- Scenario window (seconds) ---
    scenario_cfg = {
        "t_start_s": 0.0,
        "t_end_s": 180.0,          # 3 minutes episode for fast training
        "pf_fail_streak_max": 3,

        # Scenario 2: line trip/restore (exogenous)
        "line_trip": {
            "line_id": 1,          # candidate critical line (Bus 2-3 in your analysis)
            "t_trip_s": 60.0,
            "t_restore_s": 120.0,  # set None for permanent trip
        },

        # Optional: add a load step to ensure df crosses deadband
        "load_step": {
            "t_step_s": 30.0,
            "duration_s": 10.0,
            "delta_kw": 1500.0,
        },
    }

    # --- EVCS config ---
    # You can provide explicit ratings per EVCS here. If omitted, env uses heuristics from microgrid_config.
    evcs_cfg = {
        "soc_init": 0.5,
        "include_bus_switch_edges": True,

        # Base exogenous injections (kW). Env will jitter by +/-exog_rand_frac every episode.
        "exogenous_base": {
            "P_load_kw": 5200.0,
            "P_ev_kw": 1200.0,
            "P_pv_kw": 1200.0,
            "P_wind_kw": 800.0,
        },
        "exog_rand_frac": 0.10,
    }

    # --- SG params ---
    sg_params = SGParams(
        f_nom_hz=50.0,
        s_base_mva=10.0,
        H_s=2.0,
        D_pu=1.0,
        R_percent=5.0,
        deadband_pri_hz=0.02,
        Tg_s=0.2,
        Tt_s=0.5,
        use_agc=True,
        Kp_agc=0.2,
        Ki_agc=0.1,
        Tf_agc=1.5,
        deadband_agc_hz=0.02,
        P_agc_max_pu=0.30,
        ramp_agc_pu_s=0.01,
        agc_leak=0.0005,
    )

    # --- Reward config (simple, aligned with metrics) ---
    reward_cfg = {
        "df_ref_hz": 0.2,
        "rocof_ref_hz_s": 0.5,
        "ploss_ref_kw": 100.0,
        "lam_rocof": 0.5,
        "lam_loss": 0.2,
        "lam_safety": 50.0,
        "lam_clip": 0.05,
    }

    create_net_fn = lambda: create_cigre_mv_microgrid(mesh_level=2)

    env = MicrogridEnv(
        create_net_fn=create_net_fn,
        scenario_cfg=scenario_cfg,
        evcs_cfg=evcs_cfg,
        sg_params=sg_params,
        reward_cfg=reward_cfg,
        dt_ctrl=0.1,
        dt_ode=0.01,
        s_base_mva=10.0,
        seed=seed,
    )
    return env


def save_ckpt(path: str, policy: SharedPolicy, cfg: PolicyConfig) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "cfg": cfg.__dict__,
        "agent_names": policy.agent_names,
    }
    torch.save(payload, path)


def load_ckpt(path: str, policy: SharedPolicy) -> None:
    payload = torch.load(path, map_location=policy.device)
    policy.load_state_dict(payload["state_dict"])


def train(args: argparse.Namespace) -> None:
    env = make_env(seed=args.seed)
    obs0 = env.reset(seed=args.seed)

    agent_names = env.agents
    node_dim = int(obs0["graph"]["node_x"].shape[1])
    edge_dim = int(obs0["graph"]["edge_attr"].shape[1]) if obs0["graph"]["edge_attr"].ndim == 2 else 0
    local_dim = int(len(obs0["local"][agent_names[0]]))

    cfg = PolicyConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        local_dim=local_dim,
        act_dim=4,
        gnn_hidden=128,
        gnn_layers=3,
        actor_hidden=256,
        critic_hidden=256,
        lr=args.lr,
        minibatch_size=args.minibatch,
        update_epochs=args.epochs,
    )

    policy = SharedPolicy(agent_names=agent_names, cfg=cfg, device=args.device)
    algo = MAPPO(policy, cfg)

    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        buf = RolloutBuffer()

        done = False
        ep_reward = 0.0
        freq_min = 1e9
        rocof_max = 0.0
        loss_sum_kwh = 0.0

        while not done:
            a_tensor, a_dict, logp, ent, value = policy.act(obs)
            next_obs, reward, done, info = env.step(a_dict)

            buf.add(obs, a_tensor, logp, value, reward, done)

            ep_reward += reward
            freq_min = min(freq_min, float(info["freq_hz"]))
            rocof_max = max(rocof_max, abs(float(info["rocof_hz_s"])))
            loss_sum_kwh += float(info["P_loss_kw"]) * env.dt_ctrl / 3600.0

            obs = next_obs

        last_value = 0.0
        losses = algo.update(buf, last_value=last_value)

        if ep % args.log_every == 0:
            print(
                f"[EP {ep:04d}] R={ep_reward:8.2f} | f_nadir={freq_min:6.3f} Hz | "
                f"maxRoCoF={rocof_max:6.3f} | Loss={loss_sum_kwh:7.3f} kWh | "
                f"pi={losses['loss_pi']:.4f} v={losses['loss_v']:.4f} ent={losses['entropy']:.3f} kl={losses['kl']:.4f}"
            )

        if ep % args.ckpt_every == 0:
            save_ckpt("checkpoints/mappo_latest.pt", policy, cfg)
            save_ckpt(f"checkpoints/mappo_ep{ep:04d}.pt", policy, cfg)

    save_ckpt("checkpoints/mappo_final.pt", policy, cfg)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    env = make_env(seed=args.seed)
    obs0 = env.reset(seed=args.seed)

    agent_names = env.agents
    node_dim = int(obs0["graph"]["node_x"].shape[1])
    edge_dim = int(obs0["graph"]["edge_attr"].shape[1]) if obs0["graph"]["edge_attr"].ndim == 2 else 0
    local_dim = int(len(obs0["local"][agent_names[0]]))

    cfg = PolicyConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        local_dim=local_dim,
        act_dim=4,
        lr=args.lr,
        minibatch_size=args.minibatch,
        update_epochs=args.epochs,
    )

    policy = SharedPolicy(agent_names=agent_names, cfg=cfg, device=args.device)
    load_ckpt(args.ckpt, policy)
    policy.eval()

    for ep in range(1, args.episodes + 1):
        obs = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        freq_min = 1e9
        rocof_max = 0.0
        loss_sum_kwh = 0.0

        while not done:
            # deterministic-ish: use act() but you can reduce exploration by setting log_std small after loading
            a_tensor, a_dict, logp, ent, value = policy.act(obs)
            obs, reward, done, info = env.step(a_dict)

            ep_reward += reward
            freq_min = min(freq_min, float(info["freq_hz"]))
            rocof_max = max(rocof_max, abs(float(info["rocof_hz_s"])))
            loss_sum_kwh += float(info["P_loss_kw"]) * env.dt_ctrl / 3600.0

        print(f"[EVAL {ep:03d}] R={ep_reward:8.2f} | f_nadir={freq_min:6.3f} Hz | maxRoCoF={rocof_max:6.3f} | Loss={loss_sum_kwh:7.3f} kWh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50)

    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ckpt", type=str, default="checkpoints/mappo_latest.pt")

    # If you just press F5 / Run, you typically won't pass CLI arguments.
    # In that case we use the DEFAULT_* constants above.
    if len(sys.argv) <= 1:
        args = _namespace_from_defaults()
    else:
        args = ap.parse_args()

    if getattr(args, 'eval', False):
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
