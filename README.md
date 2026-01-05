2026 The 10th International Conference on Green Energy and Applications (ICGEA)

Main files
--------------

1) sim_MARL_F5_v2.py
   - Entry-point for MARL training/evaluation (MAPPO + GNN).
   - Creates the MicrogridEnv, initializes SharedPolicy, runs the train/eval loop.
   - Has default settings for a quick run via F5/Run (IDE).

2) env_v2.py
   - MicrogridEnv (MARL-ready) based on pandapower.
   - Simulates AC power flow and synchronous generator frequency dynamics (swing equation).
   - Supports line trip/restore scenarios, exogenous load/RES, and EVCS agents.

3) Magent_v2.py
   - Minimal MAPPO implementation:
     * Shared actor (parameter sharing) for EVCS agents.
     * Centralized critic.
     * Supports PPO update, GAE, and RolloutBuffer.

4) GNN_v2.py
   - Minimal MPNN encoder (no torch_geometric).
   - Takes a graph (node_x, edge_index, edge_attr) -> (node embeddings, global embedding).

5) cigre_mv_microgrid_v1.py
   - Function to create a CIGRE MV grid (pandapower network) and EVCS config.

6) sim_v1.py, sim_droop_v3.py
   - Independent / non-RL simulations to test droop / SG dynamics.

7) safe_line.py
   - Line safety-related logic/utilities (if used).

8) Magent_v2.py / GNN_v2.py
   - Algorithm/ML modules supporting MARL.

9) data/
   - Auxiliary data directory (if any).

Basic usage
-------------------
Requirements:
- Python 3.8+
- Libraries: numpy, torch, pandapower

Install libraries (suggested):
    pip install numpy torch pandapower

Run training (train):
    python sim_MARL_F5_v2.py --episodes 500 --device cpu

Run evaluation (eval) with a checkpoint:
    python sim_MARL_F5_v2.py --eval --episodes 20 --ckpt checkpoints/mappo_latest.pt

Quick configuration via IDE (VS Code / PyCharm):
- Open sim_MARL_F5_v2.py and press Run / F5.
- You can change DEFAULT_MODE / DEFAULT_EPISODES in the file to run quickly.

General workflow
-------------------------
1) sim_MARL_F5_v2.py calls make_env() to create the MicrogridEnv.
2) env_v2.py creates an observation consisting of:
   - graph: node_x, edge_index, edge_attr + system indices (df, rocof).
   - local: local state for each EVCS agent.
3) policy (Magent_v2.py) uses GNN_v2.py to encode the graph.
4) The actor generates actions for each agent; the environment steps and returns rewards.
5) MAPPO updates are performed based on the rollout.
