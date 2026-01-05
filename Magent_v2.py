"""
Magent.py
=========
Minimal MAPPO (Multi-Agent PPO) implementation with:
- Shared actor (parameter sharing across EVCS agents)
- Centralized critic
- GNN encoder backbone (from GNN.py)

This is designed for a small number of agents (e.g., 3 EVCS clusters) and short episodes
(conference-friendly). It avoids torch-geometric and heavy batching; PPO update loops
over samples.

Dependencies:
- torch
- numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GNN_v2 import MPNNEncoder, GNNConfig, MLP


def atanh_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * torch.log((1 + x) / (1 - x))


@dataclass
class PolicyConfig:
    # GNN
    node_dim: int
    edge_dim: int
    gnn_hidden: int = 128
    gnn_layers: int = 3

    # Actor/Critic
    local_dim: int = 10
    act_dim: int = 4
    id_dim: int = 16
    actor_hidden: int = 256
    critic_hidden: int = 256

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    update_epochs: int = 5
    minibatch_size: int = 64


class SharedPolicy(nn.Module):
    """
    Shared actor + centralized critic with GNN backbone.
    """
    def __init__(self, agent_names: List[str], cfg: PolicyConfig, device: str = "cpu"):
        super().__init__()
        self.agent_names = list(agent_names)
        self.n_agents = len(agent_names)
        self.cfg = cfg
        self.device = torch.device(device)

        # GNN encoder
        gcfg = GNNConfig(
            node_dim=cfg.node_dim,
            edge_dim=cfg.edge_dim,
            hidden_dim=cfg.gnn_hidden,
            msg_dim=cfg.gnn_hidden,
            n_layers=cfg.gnn_layers,
            dropout=0.0,
        )
        self.gnn = MPNNEncoder(gcfg)

        # Agent ID embedding
        self.id_emb = nn.Embedding(self.n_agents, cfg.id_dim)

        # Actor input: h_bus + local + [df, rocof] + id_emb
        self.actor_in_dim = cfg.gnn_hidden + cfg.local_dim + 2 + cfg.id_dim
        self.actor = MLP(self.actor_in_dim, cfg.act_dim, hidden_dim=cfg.actor_hidden, n_hidden=2, dropout=0.0)
        self.log_std = nn.Parameter(torch.zeros(cfg.act_dim))  # global log-std

        # Critic input: g_global + concat(local_all) + [df, rocof]
        self.critic_in_dim = cfg.gnn_hidden + (cfg.local_dim * self.n_agents) + 2
        self.critic = MLP(self.critic_in_dim, 1, hidden_dim=cfg.critic_hidden, n_hidden=2, dropout=0.0)

        self.to(self.device)

    # ----- GNN encode -----

    def encode_graph(self, graph: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        node_x = torch.nan_to_num(torch.as_tensor(graph["node_x"], device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        edge_index = torch.as_tensor(graph["edge_index"], device=self.device, dtype=torch.long)
        edge_attr = torch.nan_to_num(torch.as_tensor(graph["edge_attr"], device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        H, g = self.gnn(node_x, edge_index, edge_attr)
        return H, g

    # ----- Actor / Critic -----

    def _build_actor_inputs(self, H: torch.Tensor, obs_local: Dict[str, np.ndarray], df: float, rocof: float, evcs_pos: Dict[str, int]) -> torch.Tensor:
        """
        Returns X: [n_agents, actor_in_dim]
        """
        df_ro = torch.tensor([df, rocof], device=self.device, dtype=torch.float32)
        X = []
        for i, name in enumerate(self.agent_names):
            pos = evcs_pos[name]
            h = torch.nan_to_num(H[pos], nan=0.0, posinf=0.0, neginf=0.0)
            local = torch.nan_to_num(torch.as_tensor(obs_local[name], device=self.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            aid = self.id_emb(torch.tensor(i, device=self.device))
            x = torch.cat([h, local, df_ro, aid], dim=-1)
            X.append(x)
        return torch.stack(X, dim=0)

    def _build_critic_input(self, g: torch.Tensor, obs_local: Dict[str, np.ndarray], df: float, rocof: float) -> torch.Tensor:
        df_ro = torch.tensor([df, rocof], device=self.device, dtype=torch.float32)
        locals_flat = []
        for name in self.agent_names:
            locals_flat.append(torch.as_tensor(obs_local[name], device=self.device, dtype=torch.float32))
        locals_flat = torch.cat(locals_flat, dim=-1)
        x = torch.cat([g, locals_flat, df_ro], dim=-1)
        return x

    def _dist(self, mu: torch.Tensor) -> torch.distributions.Normal:
        # Numerical safety: prevent NaN/Inf and overly large std
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
        log_std = self.log_std.clamp(-5.0, 2.0)  # std in ~[0.0067, 7.39]
        std = torch.exp(log_std).unsqueeze(0).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @torch.no_grad()
    def act(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, float, float]:
        """
        Sample actions for all agents.

        Returns:
          actions_tensor: [n_agents, act_dim] (tanh-squashed, in [-1,1])
          action_dict: {name: np.array(act_dim)}
          logp_sum: float (sum over agents)
          entropy_sum: float
          value: float
        """
        graph = obs["graph"]
        local = obs["local"]
        df = float(graph.get("df", 0.0))
        rocof = float(graph.get("rocof", 0.0))
        if not np.isfinite(df):
            df = 0.0
        if not np.isfinite(rocof):
            rocof = 0.0
        evcs_pos = graph["evcs_pos"]

        H, g = self.encode_graph(graph)
        X = torch.nan_to_num(self._build_actor_inputs(H, local, df, rocof, evcs_pos), nan=0.0, posinf=0.0, neginf=0.0)  # [n, din]

        mu = self.actor(X)  # [n, act_dim]
        dist = self._dist(mu)

        u = dist.rsample()
        a = torch.tanh(u)

        # log prob with tanh correction
        logp = dist.log_prob(u).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        logp_sum = logp.sum().item()

        entropy = dist.entropy().sum(dim=-1).sum().item()

        # value
        v_in = self._build_critic_input(g, local, df, rocof)
        value = self.critic(v_in).squeeze(-1).item()

        a_np = a.cpu().numpy().astype(np.float32)
        action_dict = {name: a_np[i] for i, name in enumerate(self.agent_names)}
        return a_np, action_dict, float(logp_sum), float(entropy), float(value)

    def evaluate(self, obs: Dict[str, Any], actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate logp_sum, entropy_sum, value for given obs and actions.
        actions: [n_agents, act_dim] in [-1,1] (tanh output)

        Returns:
          logp_sum: scalar tensor
          entropy_sum: scalar tensor
          value: scalar tensor
        """
        graph = obs["graph"]
        local = obs["local"]
        df = float(graph.get("df", 0.0))
        rocof = float(graph.get("rocof", 0.0))
        if not np.isfinite(df):
            df = 0.0
        if not np.isfinite(rocof):
            rocof = 0.0
        evcs_pos = graph["evcs_pos"]

        H, g = self.encode_graph(graph)
        X = self._build_actor_inputs(H, local, df, rocof, evcs_pos)

        mu = self.actor(X)
        dist = self._dist(mu)

        a = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        u = atanh_torch(a)

        logp = dist.log_prob(u).sum(dim=-1) - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        logp_sum = logp.sum()

        entropy_sum = dist.entropy().sum()

        v_in = self._build_critic_input(g, local, df, rocof)
        value = self.critic(v_in).squeeze(-1)

        return logp_sum, entropy_sum, value

    @torch.no_grad()
    def value(self, obs: Dict[str, Any]) -> float:
        graph = obs["graph"]
        local = obs["local"]
        df = float(graph.get("df", 0.0))
        rocof = float(graph.get("rocof", 0.0))
        if not np.isfinite(df):
            df = 0.0
        if not np.isfinite(rocof):
            rocof = 0.0
        H, g = self.encode_graph(graph)
        v_in = self._build_critic_input(g, local, df, rocof)
        return float(self.critic(v_in).squeeze(-1).item())


class RolloutBuffer:
    def __init__(self):
        self.obs: List[Dict[str, Any]] = []
        self.actions: List[np.ndarray] = []
        self.logp: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, actions, logp, value, reward, done):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logp.append(float(logp))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)


class MAPPO:
    def __init__(self, policy: SharedPolicy, cfg: PolicyConfig):
        self.policy = policy
        self.cfg = cfg
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            v_next = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * v_next * mask - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * last_gae
            adv[t] = last_gae
            ret[t] = adv[t] + values[t]
        return adv, ret

    def update(self, buf: RolloutBuffer, last_value: float) -> Dict[str, float]:
        T = len(buf)
        assert T > 0

        adv, ret = self.compute_gae(buf.rewards, buf.values, buf.dones, last_value)

        # Normalize advantages
        adv_t = torch.as_tensor(adv, device=self.policy.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret, device=self.policy.device)

        logp_old = torch.as_tensor(np.array(buf.logp, dtype=np.float32), device=self.policy.device)
        actions_arr = buf.actions  # list of [n_agents, act_dim]
        obs_list = buf.obs

        idxs = np.arange(T)
        losses = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0}

        for _ in range(self.cfg.update_epochs):
            self.policy.train()
            np.random.shuffle(idxs)
            for start in range(0, T, self.cfg.minibatch_size):
                mb = idxs[start:start + self.cfg.minibatch_size]
                if len(mb) == 0:
                    continue

                logp_new_list = []
                ent_list = []
                v_list = []

                for j in mb:
                    lp, ent, v = self.policy.evaluate(obs_list[j], actions_arr[j])
                    logp_new_list.append(lp)
                    ent_list.append(ent)
                    v_list.append(v)

                logp_new = torch.stack(logp_new_list).squeeze(-1)
                entropy = torch.stack(ent_list).mean()
                v_pred = torch.stack(v_list).squeeze(-1)

                ratio = torch.exp(logp_new - logp_old[mb])

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_t[mb]
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = F.mse_loss(v_pred, ret_t[mb])

                loss = loss_pi + self.cfg.value_coef * loss_v - self.cfg.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (logp_old[mb] - logp_new).mean().abs().item()

                losses["loss_pi"] += float(loss_pi.item()) * len(mb)
                losses["loss_v"] += float(loss_v.item()) * len(mb)
                losses["entropy"] += float(entropy.item()) * len(mb)
                losses["kl"] += float(approx_kl) * len(mb)

        # average
        for k in losses:
            losses[k] /= max(T * self.cfg.update_epochs, 1)
        return losses
