"""
env.py
=====
Microgrid RL Environment (MARL-ready) for CIGRE MV Microgrid
- Exogenous topology event: Line trip/restore (Scenario 2)  [Design 1]
- Agents control EVCS clusters with Option-B action:
    a_k = [P_bess, Q_bess, P_v2g, Q_v2g]  (normalized in [-1,1])

Plant simulation:
- pandapower AC power flow (Newton-Raphson)
- Synchronous generator frequency dynamics via swing equation (single-machine equivalent)
- Multi-rate: dt_ctrl (RL step) and dt_ode (SG ODE internal)

Dependencies (user environment):
- pandapower
- numpy

This file is intentionally self-contained and conference-friendly (minimal metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import pandapower as pp
except Exception as e:  # pragma: no cover
    pp = None
    _PANDAPOWER_IMPORT_ERROR = e


# -----------------------------
# Utility helpers
# -----------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.minimum(np.maximum(x, lo), hi))


def _atanh(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe atanh for tanh-squashed actions."""
    x = np.clip(x, -1 + eps, 1 - eps)
    return 0.5 * np.log((1 + x) / (1 - x))


# -----------------------------
# SG model (frequency dynamics)
# -----------------------------

@dataclass
class SGParams:
    f_nom_hz: float = 50.0
    s_base_mva: float = 10.0

    # Inertia & damping
    H_s: float = 2.0
    D_pu: float = 1.0

    # Primary droop + gov/turbine
    R_percent: float = 5.0
    deadband_pri_hz: float = 0.02
    Tg_s: float = 0.2
    Tt_s: float = 0.5

    # Secondary AGC (optional)
    use_agc: bool = True
    Kp_agc: float = 0.2
    Ki_agc: float = 0.1
    Tf_agc: float = 1.5
    deadband_agc_hz: float = 0.02
    P_agc_max_pu: float = 0.30
    ramp_agc_pu_s: float = 0.01
    agc_leak: float = 0.0005


class SGModel:
    """
    Single-machine equivalent SG frequency model consistent with ω in rad/s.

    States:
      omega: rad/s deviation from nominal (Δω)
      delta: rad (not used by PF; kept for completeness)
      P_ref: p.u. governor reference
      P_m: p.u. turbine mechanical
      P_agc: p.u. secondary
      f_filt, agc_int: AGC states

    Swing:
      M_rad * domega/dt = P_m + P_agc - P_e - D_rad*omega
      ddelta/dt = omega
      Δf = omega / (2π)
    """

    def __init__(self, P_nom_pu: float, params: SGParams):
        self.p = params
        self.omega0 = 2 * np.pi * self.p.f_nom_hz

        # Correct unit conversion for ω in rad/s
        self.M_rad = (2 * self.p.H_s) / self.omega0
        self.D_rad = self.p.D_pu / self.omega0

        # Dispatch
        self.P_nom = float(P_nom_pu)

        # States
        self.omega = 0.0
        self.delta = 0.0

        self.P_ref = float(P_nom_pu)
        self.P_m = float(P_nom_pu)

        # AGC
        self.f_filt = 0.0
        self.agc_int = 0.0
        self.P_agc = 0.0

        # For RoCoF computation
        self._last_domega_dt = 0.0

    def _primary_control(self, df_hz: float, dt: float) -> None:
        # deadband
        db = self.p.deadband_pri_hz
        if abs(df_hz) <= db:
            df_eff = 0.0
        else:
            df_eff = df_hz - np.sign(df_hz) * db

        # droop ΔP = -Δf / R_Hz, where R_Hz = (R%/100)*f_nom
        R_Hz = (self.p.R_percent / 100.0) * self.p.f_nom_hz
        P_droop = -df_eff / R_Hz if R_Hz > 0 else 0.0

        # governor reference target
        P_ref_tgt = self.P_nom + P_droop

        # simple saturation (can be expanded)
        # keep in [0, 2] pu to avoid crazy values; user can tune
        P_ref_tgt = _clip(P_ref_tgt, 0.0, 2.0)

        # governor lag
        Tg = max(self.p.Tg_s, 1e-3)
        self.P_ref += (P_ref_tgt - self.P_ref) * (dt / Tg)

        # turbine lag
        Tt = max(self.p.Tt_s, 1e-3)
        self.P_m += (self.P_ref - self.P_m) * (dt / Tt)

        # clip again
        self.P_m = _clip(self.P_m, 0.0, 2.0)

    def _secondary_control(self, df_hz: float, dt: float) -> None:
        if not self.p.use_agc:
            self.P_agc = 0.0
            self.agc_int = 0.0
            return

        # filter
        Tf = max(self.p.Tf_agc, 1e-3)
        alpha = dt / (Tf + dt)
        self.f_filt += alpha * (df_hz - self.f_filt)

        # deadband
        db = self.p.deadband_agc_hz
        if abs(self.f_filt) <= db:
            f_eff = 0.0
        else:
            f_eff = self.f_filt - np.sign(self.f_filt) * db

        # PI
        P_prop = -self.p.Kp_agc * f_eff
        self.agc_int += (-self.p.Ki_agc * f_eff) * dt
        self.agc_int -= self.p.agc_leak * self.agc_int * dt

        P_max = self.p.P_agc_max_pu
        self.agc_int = float(np.clip(self.agc_int, -P_max, P_max))
        P_tgt = float(np.clip(P_prop + self.agc_int, -P_max, P_max))

        # ramp limit
        ramp = self.p.ramp_agc_pu_s
        dP = P_tgt - self.P_agc
        dP = float(np.clip(dP, -ramp * dt, ramp * dt))
        self.P_agc += dP

    def step(self, P_e_pu: float, dt: float) -> Tuple[float, float]:
        """
        One ODE step for SG.

        Returns:
          df_hz: frequency deviation (Hz)
          rocof_hz_s: RoCoF (Hz/s)
        """
        df_hz = self.omega / (2 * np.pi)

        # controls
        self._primary_control(df_hz, dt)
        self._secondary_control(df_hz, dt)

        # swing
        domega_dt = (self.P_m + self.P_agc - P_e_pu - self.D_rad * self.omega) / max(self.M_rad, 1e-6)
        self._last_domega_dt = float(domega_dt)

        self.omega += domega_dt * dt
        self.delta += self.omega * dt

        df_hz = self.omega / (2 * np.pi)
        rocof = domega_dt / (2 * np.pi)
        return float(df_hz), float(rocof)

    def get_freq_hz(self) -> float:
        return float(self.p.f_nom_hz + self.omega / (2 * np.pi))

    def get_rocof_hz_s(self) -> float:
        return float(self._last_domega_dt / (2 * np.pi))


# -----------------------------
# EVCS device model
# -----------------------------

@dataclass
class DeviceRating:
    p_max_kw: float
    pf_min: float = 0.90
    e_kwh: float = 1000.0

    soc_min: float = 0.2
    soc_max: float = 0.9
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    ramp_p_kw_s: float = 2000.0
    ramp_q_kvar_s: float = 2000.0

    def s_rated_kva(self) -> float:
        return self.p_max_kw / max(self.pf_min, 1e-3)

    def q_max_kvar(self) -> float:
        S = self.s_rated_kva()
        return float(np.sqrt(max(S * S - self.p_max_kw * self.p_max_kw, 0.0)))


@dataclass
class EVCSClusterRating:
    bess: DeviceRating
    v2g: DeviceRating


# -----------------------------
# Environment
# -----------------------------

class MicrogridEnv:
    """
    A minimal pandapower-based microgrid environment for MARL + GNN.

    - Topology change (line trip/restore) is exogenous (scenario event).
    - Action is Option-B (Pb,Qb,Pv,Qv) per EVCS agent.
    - Reward is simple and aligned with reporting metrics (Δf, RoCoF, losses).

    Note: This environment is not Gymnasium-dependent (to keep dependencies minimal).
    """

    def __init__(
        self,
        create_net_fn: Callable[[], Any],
        scenario_cfg: Dict[str, Any],
        evcs_cfg: Dict[str, Any],
        sg_params: SGParams,
        reward_cfg: Dict[str, float],
        dt_ctrl: float = 0.1,
        dt_ode: float = 0.01,
        s_base_mva: float = 10.0,
        seed: int = 0,
    ):
        if pp is None:  # pragma: no cover
            raise ImportError(
                "pandapower is required to run MicrogridEnv. "
                f"Import error: {_PANDAPOWER_IMPORT_ERROR}"
            )

        self.create_net_fn = create_net_fn
        self.scenario_cfg = scenario_cfg
        self.evcs_cfg = evcs_cfg
        self.sg_params = sg_params
        self.reward_cfg = reward_cfg

        self.dt_ctrl = float(dt_ctrl)
        self.dt_ode = float(dt_ode)
        self.s_base_mva = float(s_base_mva)

        self.rng = np.random.default_rng(seed)

        # populated after reset()
        self.net = None
        self.t = 0.0
        self.sg: Optional[SGModel] = None

        # Agents
        self.agents: List[str] = []
        self.bus_of: Dict[str, int] = {}

        # Ratings per agent
        self.ratings: Dict[str, EVCSClusterRating] = {}

        # Element indices for fast update
        self.idx_loads: Dict[int, int] = {}
        self.idx_ev_loads: Dict[int, int] = {}
        self.idx_pv_sgens: Dict[int, int] = {}
        self.idx_wind_sgen: Optional[int] = None

        self.idx_bess_sgen: Dict[str, int] = {}
        self.idx_v2g_sgen: Dict[str, int] = {}

        self.sg_gen_idx: int = 0

        # EVCS states
        self.soc_bess: Dict[str, float] = {}
        self.soc_v2g: Dict[str, float] = {}
        self.last_cmd: Dict[str, np.ndarray] = {}  # [Pb,Qb,Pv,Qv] in kW/kvar

        # Exogenous injections
        self._exog = {}  # sampled per episode

        # Scenario state flags
        self._line_tripped = False

        # Simple safety tracking
        self._pf_fail_streak = 0

    # -------- public API --------

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.net = self.create_net_fn()

        # Islanding: disable HV bus0 and any trafo connected to hv_bus==0
        self._make_islanded()

        # Remove existing loads/sgens; keep gen (SG slack)
        self._clear_loads_and_sgens()

        # Read EVCS bus mapping from net.microgrid_config if available, else use evcs_cfg
        self._init_agents_and_buses()

        # Build/load EVCS ratings
        self._init_evcs_ratings()

        # Create exogenous elements (loads + RES) and controlled EVCS sgens
        self._create_elements()

        # Sample episode exogenous profile scalars
        self._sample_episode_exogenous()

        # Initialize time
        self.t = float(self.scenario_cfg.get("t_start_s", 0.0))
        self._line_tripped = False
        self._pf_fail_streak = 0

        # Set initial injections and solve PF
        self._apply_exogenous_injections(self.t)
        pf_ok = self._run_power_flow(init="auto")
        if not pf_ok:
            # Try a second attempt with different init
            pf_ok = self._run_power_flow(init="flat")
        if not pf_ok:  # pragma: no cover
            raise RuntimeError("Power flow did not converge at reset(). Check network and injections.")

        # Determine SG gen idx (slack gen)
        self.sg_gen_idx = self._find_slack_gen_idx()

        # Initialize SG model dispatch to current electrical output
        P_e_mw = self._get_sg_p_e_mw()
        P_nom_pu = float(P_e_mw / self.s_base_mva)
        self.sg = SGModel(P_nom_pu=P_nom_pu, params=self.sg_params)

        # Initialize EVCS SoC states and last commands
        for a in self.agents:
            self.soc_bess[a] = float(self.evcs_cfg.get("soc_init", 0.5))
            self.soc_v2g[a] = float(self.evcs_cfg.get("soc_init", 0.5))
            self.last_cmd[a] = np.zeros(4, dtype=np.float32)

        return self._build_obs(pf_ok=pf_ok)

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self.net is not None and self.sg is not None, "Call reset() first."

        # Apply scenario (line trip/restore) exogenously
        self._apply_scenario_events(self.t)

        # Update exogenous injections (loads, PV, wind, EV)
        self._apply_exogenous_injections(self.t)

        # Project RL actions -> feasible P/Q commands
        cmd, clip_pen = self._project_actions(action_dict)

        # Apply EVCS commands to net
        self._apply_evcs_commands_to_net(cmd)

        # Run AC power flow
        pf_ok = self._run_power_flow(init="results")
        if not pf_ok:
            # fall back to a more robust init
            pf_ok = self._run_power_flow(init="auto")

        if not pf_ok:
            self._pf_fail_streak += 1
        else:
            self._pf_fail_streak = 0

        # Get SG electrical power from PF
        P_e_mw = self._get_sg_p_e_mw() if pf_ok else self.s_base_mva * 1.0  # fallback
        P_e_pu = float(P_e_mw / self.s_base_mva)

        # Integrate SG ODE for dt_ctrl using dt_ode substeps
        n_sub = max(int(round(self.dt_ctrl / self.dt_ode)), 1)
        df_hz = 0.0
        rocof = 0.0
        for _ in range(n_sub):
            df_hz, rocof = self.sg.step(P_e_pu=P_e_pu, dt=self.dt_ode)

        freq_hz = self.sg.get_freq_hz()

        # Update SoC based on projected commands over dt_ctrl
        self._update_socs(cmd)

        # Compute losses
        P_loss_kw = self._get_losses_kw() if pf_ok else 5_000.0

        # Reward
        reward = self._compute_reward(df_hz=df_hz, rocof_hz_s=rocof, P_loss_kw=P_loss_kw, pf_ok=pf_ok, clip_penalty=clip_pen)

        # Advance time
        self.t += self.dt_ctrl

        # Done?
        done = self._is_done(pf_ok=pf_ok)

        obs = self._build_obs(pf_ok=pf_ok)

        info = {
            "t_s": self.t,
            "pf_ok": pf_ok,
            "freq_hz": freq_hz,
            "df_hz": df_hz,
            "rocof_hz_s": rocof,
            "P_e_mw": P_e_mw,
            "P_loss_kw": P_loss_kw,
            "clip_penalty": clip_pen,
            "line_tripped": self._line_tripped,
            "cmd": cmd,
            "soc_bess": dict(self.soc_bess),
            "soc_v2g": dict(self.soc_v2g),
        }

        return obs, float(reward), bool(done), info

    # -------- network initialization --------

    def _make_islanded(self) -> None:
        # Disable HV bus 0 and any transformer that connects HV side 0
        if 0 in self.net.bus.index:
            try:
                self.net.bus.at[0, "in_service"] = False
            except Exception:
                pass

        if len(self.net.trafo) > 0:
            for idx in self.net.trafo.index:
                try:
                    if int(self.net.trafo.at[idx, "hv_bus"]) == 0:
                        self.net.trafo.at[idx, "in_service"] = False
                except Exception:
                    continue

    def _clear_loads_and_sgens(self) -> None:
        if len(self.net.load) > 0:
            self.net.load.drop(self.net.load.index, inplace=True)
        if len(self.net.sgen) > 0:
            self.net.sgen.drop(self.net.sgen.index, inplace=True)

        # Ensure there is at least one gen acting as slack
        if len(self.net.gen) > 0:
            # make first gen slack; user network already has slack=True at SG bus
            try:
                # keep existing slack if exists
                if not bool(self.net.gen["slack"].any()):
                    self.net.gen.at[self.net.gen.index[0], "slack"] = True
            except Exception:
                pass
        else:  # pragma: no cover
            raise RuntimeError("Network has no 'gen' element. Need SG slack gen at Bus 1 for islanded PF.")

    def _init_agents_and_buses(self) -> None:
        bus_of = {}

        # Prefer net["microgrid_config"]["evcs_clusters"]
        try:
            cfg = self.net.get("microgrid_config", {})
            evcs = cfg.get("evcs_clusters", {})
            for name, d in evcs.items():
                bus_of[name] = int(d["bus"])
        except Exception:
            bus_of = {}

        # Fallback to evcs_cfg
        if not bus_of:
            bus_of = {k: int(v) for k, v in self.evcs_cfg["bus_of"].items()}

        self.bus_of = bus_of
        self.agents = list(bus_of.keys())

    def _init_evcs_ratings(self) -> None:
        # User can pass explicit ratings dict; else use heuristics
        ratings_cfg = self.evcs_cfg.get("ratings", {})

        for name in self.agents:
            if name in ratings_cfg:
                r = ratings_cfg[name]
                bess = DeviceRating(**r["bess"])
                v2g = DeviceRating(**r["v2g"])
            else:
                # heuristic: BESS ~ 0.8*EV load capacity, V2G ~ 0.6*EV load
                # If microgrid_config has p_ev_kw, use that
                p_ev_kw = None
                try:
                    p_ev_kw = float(self.net["microgrid_config"]["evcs_clusters"][name].get("p_ev_kw", None))
                except Exception:
                    p_ev_kw = None
                if p_ev_kw is None:
                    p_ev_kw = 600.0

                pb = 0.8 * p_ev_kw
                pv = 0.6 * p_ev_kw
                bess = DeviceRating(p_max_kw=pb, e_kwh=max(1.0, pb * 1.0))
                v2g = DeviceRating(p_max_kw=pv, e_kwh=max(1.0, pv * 2.0), ramp_p_kw_s=300.0, ramp_q_kvar_s=300.0)

            self.ratings[name] = EVCSClusterRating(bess=bess, v2g=v2g)

    def _create_elements(self) -> None:
        """
        Create static loads and RES sgens, plus controllable EVCS BESS/V2G sgens.
        """
        # Base load distribution over MV buses (conference-friendly)
        self.load_dist = self.evcs_cfg.get(
            "load_dist",
            {
                1: 0.0, 2: 0.05, 3: 0.10, 4: 0.05, 5: 0.10,
                6: 0.08, 7: 0.08, 8: 0.10, 9: 0.05, 10: 0.08,
                11: 0.10, 12: 0.08, 13: 0.05, 14: 0.08,
            },
        )

        # EV load distribution among EVCS buses using p_ev_kw weights from microgrid_config if possible
        ev_weights = {}
        try:
            for name, d in self.net["microgrid_config"]["evcs_clusters"].items():
                ev_weights[self.bus_of[name]] = float(d.get("p_ev_kw", 1.0))
        except Exception:
            for name in self.agents:
                ev_weights[self.bus_of[name]] = 1.0

        total_w = sum(ev_weights.values()) if ev_weights else 1.0
        self.ev_dist = {bus: w / total_w for bus, w in ev_weights.items()}

        # PV allocation: standalone + EVCS PV from config if present
        pv_alloc = {}
        try:
            # standalone res
            for key, d in self.net["microgrid_config"].get("standalone_res", {}).items():
                if "PV" in key:
                    pv_alloc[int(d["bus"])] = pv_alloc.get(int(d["bus"]), 0.0) + float(d.get("p_kw", 0.0))
            # evcs PV
            for name, d in self.net["microgrid_config"]["evcs_clusters"].items():
                pv_alloc[int(d["bus"])] = pv_alloc.get(int(d["bus"]), 0.0) + float(d.get("p_pv_kw", 0.0))
        except Exception:
            pv_alloc = {4: 500.0, 6: 350.0, 9: 350.0, 10: 350.0}

        self.pv_alloc_kw = pv_alloc
        self.pv_total_kw = sum(pv_alloc.values()) if pv_alloc else 1.0

        # Wind at bus from config if present
        self.wind_bus = None
        try:
            for key, d in self.net["microgrid_config"].get("standalone_res", {}).items():
                if "Wind" in key:
                    self.wind_bus = int(d["bus"])
        except Exception:
            self.wind_bus = 13

        # Create base loads
        for bus, frac in self.load_dist.items():
            if bus in self.net.bus.index and bool(self.net.bus.at[bus, "in_service"]):
                idx = pp.create_load(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"Load_{bus}")
                self.idx_loads[bus] = idx

        # Create EV loads at EVCS buses
        for bus in self.ev_dist.keys():
            if bus in self.net.bus.index and bool(self.net.bus.at[bus, "in_service"]):
                idx = pp.create_load(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"EV_Load_{bus}")
                self.idx_ev_loads[bus] = idx

        # Create PV sgens
        for bus, _kw in self.pv_alloc_kw.items():
            if bus in self.net.bus.index and bool(self.net.bus.at[bus, "in_service"]):
                idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"PV_{bus}", type="PV")
                self.idx_pv_sgens[bus] = idx

        # Create wind sgen
        if self.wind_bus is not None and self.wind_bus in self.net.bus.index and bool(self.net.bus.at[self.wind_bus, "in_service"]):
            self.idx_wind_sgen = pp.create_sgen(self.net, bus=self.wind_bus, p_mw=0.0, q_mvar=0.0, name=f"Wind_{self.wind_bus}", type="Wind")

        # Create controllable EVCS sgens: BESS + V2G per agent
        for name in self.agents:
            bus = self.bus_of[name]
            if bus not in self.net.bus.index or not bool(self.net.bus.at[bus, "in_service"]):
                raise RuntimeError(f"EVCS agent {name} bus {bus} not in-service or missing.")
            self.idx_bess_sgen[name] = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"{name}_BESS", type="BESS", controllable=True)
            self.idx_v2g_sgen[name] = pp.create_sgen(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"{name}_V2G", type="V2G", controllable=True)

    # -------- exogenous profiles & scenarios --------

    def _sample_episode_exogenous(self) -> None:
        # Base injection levels (kW) sampled per episode
        base = self.evcs_cfg.get("exogenous_base", {})
        load_kw = float(base.get("P_load_kw", 5000.0))
        ev_kw = float(base.get("P_ev_kw", 1200.0))
        pv_kw = float(base.get("P_pv_kw", 1200.0))
        wind_kw = float(base.get("P_wind_kw", 800.0))

        # Randomize +/-10% by default
        sigma = float(self.evcs_cfg.get("exog_rand_frac", 0.10))
        def jitter(x): return x * float(1.0 + self.rng.uniform(-sigma, sigma))

        self._exog = {
            "P_load_kw": jitter(load_kw),
            "P_ev_kw": jitter(ev_kw),
            "P_pv_kw": jitter(pv_kw),
            "P_wind_kw": jitter(wind_kw),
        }

    def _apply_scenario_events(self, t: float) -> None:
        # Line trip
        lt = self.scenario_cfg.get("line_trip", None)
        if lt is None:
            return

        line_id = int(lt["line_id"])
        t_trip = float(lt.get("t_trip_s", 60.0))
        t_restore = lt.get("t_restore_s", None)

        if (not self._line_tripped) and (t >= t_trip):
            if line_id in self.net.line.index:
                self.net.line.at[line_id, "in_service"] = False
            self._line_tripped = True

        if self._line_tripped and (t_restore is not None) and (t >= float(t_restore)):
            if line_id in self.net.line.index:
                self.net.line.at[line_id, "in_service"] = True

    def _apply_exogenous_injections(self, t: float) -> None:
        # Base values
        P_load = float(self._exog["P_load_kw"])
        P_ev = float(self._exog["P_ev_kw"])
        P_pv = float(self._exog["P_pv_kw"])
        P_w = float(self._exog["P_wind_kw"])

        # Optional load step
        ls = self.scenario_cfg.get("load_step", None)
        if ls is not None:
            t0 = float(ls.get("t_step_s", 20.0))
            dur = float(ls.get("duration_s", 10.0))
            dkw = float(ls.get("delta_kw", 0.0))
            if t0 <= t < (t0 + dur):
                P_load += dkw

        # Set loads (simple Q=0 for conference)
        for bus, idx in self.idx_loads.items():
            frac = float(self.load_dist.get(bus, 0.0))
            p_mw = (P_load * frac) / 1000.0
            self.net.load.at[idx, "p_mw"] = p_mw
            self.net.load.at[idx, "q_mvar"] = 0.0

        for bus, idx in self.idx_ev_loads.items():
            frac = float(self.ev_dist.get(bus, 0.0))
            p_mw = (P_ev * frac) / 1000.0
            self.net.load.at[idx, "p_mw"] = p_mw
            self.net.load.at[idx, "q_mvar"] = 0.0

        # PV generation
        for bus, idx in self.idx_pv_sgens.items():
            alloc = float(self.pv_alloc_kw.get(bus, 0.0))
            frac = alloc / max(self.pv_total_kw, 1e-6)
            p_mw = (P_pv * frac) / 1000.0
            self.net.sgen.at[idx, "p_mw"] = p_mw
            self.net.sgen.at[idx, "q_mvar"] = 0.0

        # Wind generation
        if self.idx_wind_sgen is not None:
            self.net.sgen.at[self.idx_wind_sgen, "p_mw"] = P_w / 1000.0
            self.net.sgen.at[self.idx_wind_sgen, "q_mvar"] = 0.0

    # -------- power flow & measurements --------

    def _run_power_flow(self, init: str = "results") -> bool:
        try:
            pp.runpp(self.net, algorithm="nr", init=init, enforce_q_lims=True, max_iteration=30, tolerance_mva=1e-6)
            return True
        except Exception:
            return False

    def _find_slack_gen_idx(self) -> int:
        try:
            slack_rows = self.net.gen[self.net.gen["slack"] == True]
            if len(slack_rows) > 0:
                return int(slack_rows.index[0])
        except Exception:
            pass
        return int(self.net.gen.index[0])

    def _get_sg_p_e_mw(self) -> float:
        # Electrical power from PF slack gen
        try:
            return float(self.net.res_gen.at[self.sg_gen_idx, "p_mw"])
        except Exception:
            # fallback: sum all gen
            try:
                return float(self.net.res_gen.p_mw.sum())
            except Exception:
                return float(self.s_base_mva * 1.0)

    def _get_losses_kw(self) -> float:
        pl_mw = 0.0
        try:
            if hasattr(self.net, "res_line") and len(self.net.res_line) > 0:
                pl_mw += float(self.net.res_line.pl_mw.sum())
        except Exception:
            pass
        try:
            if hasattr(self.net, "res_trafo") and len(self.net.res_trafo) > 0:
                pl_mw += float(self.net.res_trafo.pl_mw.sum())
        except Exception:
            pass
        return float(max(0.0, pl_mw * 1000.0))

    # -------- EVCS actions & constraints --------

    def _project_actions(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Map normalized actions -> feasible kW/kvar commands.
        Returns:
          cmd[name] = np.array([Pb,Qb,Pv,Qv]) in kW/kvar
          clip_penalty = sum |raw - projected| (normalized)
        """
        clip_pen = 0.0
        cmd_out: Dict[str, np.ndarray] = {}

        dt = self.dt_ctrl

        for name in self.agents:
            a = action_dict.get(name, None)
            if a is None:
                a = np.zeros(4, dtype=np.float32)
            a = np.asarray(a, dtype=np.float32).reshape(4)
            a = np.clip(a, -1.0, 1.0)

            r = self.ratings[name]
            Pb_max = r.bess.p_max_kw
            Qb_max = r.bess.q_max_kvar()
            Pv_max = r.v2g.p_max_kw
            Qv_max = r.v2g.q_max_kvar()

            raw = np.array([
                a[0] * Pb_max,
                a[1] * Qb_max,
                a[2] * Pv_max,
                a[3] * Qv_max,
            ], dtype=np.float32)

            prev = self.last_cmd.get(name, np.zeros(4, dtype=np.float32))

            # Ramp
            Pb = float(np.clip(raw[0], prev[0] - r.bess.ramp_p_kw_s * dt, prev[0] + r.bess.ramp_p_kw_s * dt))
            Qb = float(np.clip(raw[1], prev[1] - r.bess.ramp_q_kvar_s * dt, prev[1] + r.bess.ramp_q_kvar_s * dt))
            Pv = float(np.clip(raw[2], prev[2] - r.v2g.ramp_p_kw_s * dt, prev[2] + r.v2g.ramp_p_kw_s * dt))
            Qv = float(np.clip(raw[3], prev[3] - r.v2g.ramp_q_kvar_s * dt, prev[3] + r.v2g.ramp_q_kvar_s * dt))

            # SoC gating
            soc_b = self.soc_bess.get(name, 0.5)
            soc_v = self.soc_v2g.get(name, 0.5)

            # Prevent discharge when empty (P>0), prevent charge when full (P<0)
            if soc_b <= r.bess.soc_min:
                Pb = min(Pb, 0.0)
            if soc_b >= r.bess.soc_max:
                Pb = max(Pb, 0.0)

            if soc_v <= r.v2g.soc_min:
                Pv = min(Pv, 0.0)
            if soc_v >= r.v2g.soc_max:
                Pv = max(Pv, 0.0)

            # Capability circle
            S_b = r.bess.s_rated_kva()
            Qcap_b = float(np.sqrt(max(S_b * S_b - Pb * Pb, 0.0)))
            Qb = float(np.clip(Qb, -Qcap_b, Qcap_b))

            S_v = r.v2g.s_rated_kva()
            Qcap_v = float(np.sqrt(max(S_v * S_v - Pv * Pv, 0.0)))
            Qv = float(np.clip(Qv, -Qcap_v, Qcap_v))

            projected = np.array([Pb, Qb, Pv, Qv], dtype=np.float32)

            # Clip penalty normalized by ratings
            scale = np.array([Pb_max, max(Qb_max, 1.0), Pv_max, max(Qv_max, 1.0)], dtype=np.float32)
            clip_pen += float(np.sum(np.abs((raw - projected) / (scale + 1e-6))))

            cmd_out[name] = projected
            self.last_cmd[name] = projected

        return cmd_out, float(clip_pen)

    def _apply_evcs_commands_to_net(self, cmd: Dict[str, np.ndarray]) -> None:
        for name, vec in cmd.items():
            Pb, Qb, Pv, Qv = map(float, vec)
            # convert to MW/Mvar
            self.net.sgen.at[self.idx_bess_sgen[name], "p_mw"] = Pb / 1000.0
            self.net.sgen.at[self.idx_bess_sgen[name], "q_mvar"] = Qb / 1000.0
            self.net.sgen.at[self.idx_v2g_sgen[name], "p_mw"] = Pv / 1000.0
            self.net.sgen.at[self.idx_v2g_sgen[name], "q_mvar"] = Qv / 1000.0

    def _update_socs(self, cmd: Dict[str, np.ndarray]) -> None:
        dt_hr = self.dt_ctrl / 3600.0
        for name, vec in cmd.items():
            Pb, _, Pv, _ = map(float, vec)
            r = self.ratings[name]

            # BESS
            soc = self.soc_bess[name]
            P_dis = max(Pb, 0.0)
            P_ch = max(-Pb, 0.0)
            soc = soc - (P_dis * dt_hr) / max(r.bess.eta_dis * r.bess.e_kwh, 1e-6) + (r.bess.eta_ch * P_ch * dt_hr) / max(r.bess.e_kwh, 1e-6)
            self.soc_bess[name] = float(soc)

            # V2G
            soc = self.soc_v2g[name]
            P_dis = max(Pv, 0.0)
            P_ch = max(-Pv, 0.0)
            soc = soc - (P_dis * dt_hr) / max(r.v2g.eta_dis * r.v2g.e_kwh, 1e-6) + (r.v2g.eta_ch * P_ch * dt_hr) / max(r.v2g.e_kwh, 1e-6)
            self.soc_v2g[name] = float(soc)

    # -------- reward & termination --------

    def _compute_reward(self, df_hz: float, rocof_hz_s: float, P_loss_kw: float, pf_ok: bool, clip_penalty: float) -> float:
        df_ref = float(self.reward_cfg.get("df_ref_hz", 0.2))
        rocof_ref = float(self.reward_cfg.get("rocof_ref_hz_s", 0.5))
        ploss_ref = float(self.reward_cfg.get("ploss_ref_kw", 100.0))
        lam_r = float(self.reward_cfg.get("lam_rocof", 0.5))
        lam_l = float(self.reward_cfg.get("lam_loss", 0.2))
        lam_s = float(self.reward_cfg.get("lam_safety", 50.0))
        lam_c = float(self.reward_cfg.get("lam_clip", 0.1))

        # per-step loss energy normalized (kW*s)
        E_ref = ploss_ref * self.dt_ctrl
        E_loss = P_loss_kw * self.dt_ctrl

        # SoC violations
        soc_viol = 0.0
        for name in self.agents:
            r = self.ratings[name]
            sb = self.soc_bess[name]
            sv = self.soc_v2g[name]
            soc_viol += max(0.0, r.bess.soc_min - sb) + max(0.0, sb - r.bess.soc_max)
            soc_viol += max(0.0, r.v2g.soc_min - sv) + max(0.0, sv - r.v2g.soc_max)

        viol = (0.0 if pf_ok else 1.0) + soc_viol

        rwd = -(abs(df_hz) / df_ref) - lam_r * (abs(rocof_hz_s) / rocof_ref) - lam_l * (E_loss / max(E_ref, 1e-6)) - lam_s * viol - lam_c * clip_penalty
        return float(rwd)

    def _is_done(self, pf_ok: bool) -> bool:
        t_end = float(self.scenario_cfg.get("t_end_s", 180.0))
        if self.t >= t_end:
            return True
        # terminate on repeated PF failure
        if self._pf_fail_streak >= int(self.scenario_cfg.get("pf_fail_streak_max", 3)):
            return True
        return False

    # -------- observation building --------

    def _build_obs(self, pf_ok: bool) -> Dict[str, Any]:
        graph = self._build_graph(pf_ok=pf_ok)
        local = self._build_local()
        return {"graph": graph, "local": local}

    def _build_local(self) -> Dict[str, np.ndarray]:
        local = {}
        for name in self.agents:
            r = self.ratings[name]
            cmd = self.last_cmd.get(name, np.zeros(4, dtype=np.float32))
            Pb, Qb, Pv, Qv = cmd.astype(np.float32)

            # Normalize by ratings
            Pb_max = r.bess.p_max_kw
            Qb_max = r.bess.q_max_kvar()
            Pv_max = r.v2g.p_max_kw
            Qv_max = r.v2g.q_max_kvar()

            S_b = r.bess.s_rated_kva()
            Qcap_b = float(np.sqrt(max(S_b * S_b - float(Pb) * float(Pb), 0.0)))
            S_v = r.v2g.s_rated_kva()
            Qcap_v = float(np.sqrt(max(S_v * S_v - float(Pv) * float(Pv), 0.0)))

            vec = np.array([
                self.soc_bess[name],
                self.soc_v2g[name],
                Pb / max(Pb_max, 1e-6),
                Qb / max(Qb_max, 1e-6),
                Pv / max(Pv_max, 1e-6),
                Qv / max(Qv_max, 1e-6),
                (Pb_max - abs(Pb)) / max(Pb_max, 1e-6),
                Qcap_b / max(Qb_max, 1e-6),
                (Pv_max - abs(Pv)) / max(Pv_max, 1e-6),
                Qcap_v / max(Qv_max, 1e-6),
            ], dtype=np.float32)

            local[name] = vec
        return local

    def _build_graph(self, pf_ok: bool) -> Dict[str, Any]:
        """
        Build a dynamic graph view of the network for GNN.
        Nodes: in-service buses
        Edges: in-service lines + trafos (+ optionally closed bus-bus switches)
        """
        # Bus list (keep consistent ordering)
        buses = [int(b) for b in self.net.bus.index if bool(self.net.bus.at[b, "in_service"])]
        bus_pos = {b: i for i, b in enumerate(buses)}
        n = len(buses)

        # Node features
        node_x = np.zeros((n, 5), dtype=np.float32)  # [V, theta, P_pu, Q_pu, is_evcs]
        if pf_ok and hasattr(self.net, "res_bus") and len(self.net.res_bus) > 0:
            for b in buses:
                i = bus_pos[b]
                try:
                    V_raw = self.net.res_bus.at[b, "vm_pu"]
                    th_raw = self.net.res_bus.at[b, "va_degree"]
                    P_raw = self.net.res_bus.at[b, "p_mw"]
                    Q_raw = self.net.res_bus.at[b, "q_mvar"]

                    V = float(V_raw) if np.isfinite(V_raw) else 1.0
                    th_deg = float(th_raw) if np.isfinite(th_raw) else 0.0
                    P_mw = float(P_raw) if np.isfinite(P_raw) else 0.0
                    Q_mvar = float(Q_raw) if np.isfinite(Q_raw) else 0.0

                    th = th_deg * np.pi / 180.0
                    P = P_mw / self.s_base_mva
                    Q = Q_mvar / self.s_base_mva
                except Exception:
                    V, th, P, Q = 1.0, 0.0, 0.0, 0.0

                node_x[i, 0] = V
                node_x[i, 1] = th
                node_x[i, 2] = P
                node_x[i, 3] = Q
                node_x[i, 4] = 1.0 if b in self.bus_of.values() else 0.0
        else:
            node_x[:, 0] = 1.0

        # Edges
        edge_src = []
        edge_dst = []
        edge_attr = []  # [r_pu, x_pu, type_line, type_trafo, type_switch]

        # Base impedance conversion
        def zbase(v_kv: float) -> float:
            return (v_kv * v_kv) / max(self.s_base_mva, 1e-6)

        # Lines
        if len(self.net.line) > 0:
            for idx in self.net.line.index:
                try:
                    if not bool(self.net.line.at[idx, "in_service"]):
                        continue
                    fb = int(self.net.line.at[idx, "from_bus"])
                    tb = int(self.net.line.at[idx, "to_bus"])
                    if fb not in bus_pos or tb not in bus_pos:
                        continue

                    r_ohm = float(self.net.line.at[idx, "r_ohm_per_km"]) * float(self.net.line.at[idx, "length_km"])
                    x_ohm = float(self.net.line.at[idx, "x_ohm_per_km"]) * float(self.net.line.at[idx, "length_km"])
                    v_kv = float(self.net.bus.at[fb, "vn_kv"])
                    zb = zbase(v_kv)
                    r_pu = r_ohm / max(zb, 1e-6)
                    x_pu = x_ohm / max(zb, 1e-6)

                    i = bus_pos[fb]
                    j = bus_pos[tb]

                    # undirected -> add both directions
                    edge_src += [i, j]
                    edge_dst += [j, i]
                    edge_attr += [
                        [r_pu, x_pu, 1.0, 0.0, 0.0],
                        [r_pu, x_pu, 1.0, 0.0, 0.0],
                    ]
                except Exception:
                    continue

        # Transformers
        if len(self.net.trafo) > 0:
            for idx in self.net.trafo.index:
                try:
                    if not bool(self.net.trafo.at[idx, "in_service"]):
                        continue
                    hv = int(self.net.trafo.at[idx, "hv_bus"])
                    lv = int(self.net.trafo.at[idx, "lv_bus"])
                    if hv not in bus_pos or lv not in bus_pos:
                        continue

                    # Approximate pu from vk_percent and sn_mva
                    vk = float(self.net.trafo.at[idx, "vk_percent"]) / 100.0
                    sn = float(self.net.trafo.at[idx, "sn_mva"])
                    x_pu = vk * (self.s_base_mva / max(sn, 1e-6))
                    r_pu = 0.01 * x_pu  # rough, can refine via vkr_percent if available

                    i = bus_pos[hv]
                    j = bus_pos[lv]
                    edge_src += [i, j]
                    edge_dst += [j, i]
                    edge_attr += [
                        [r_pu, x_pu, 0.0, 1.0, 0.0],
                        [r_pu, x_pu, 0.0, 1.0, 0.0],
                    ]
                except Exception:
                    continue

        # Optional: closed bus-bus switches as edges with tiny impedance
        if self.evcs_cfg.get("include_bus_switch_edges", True) and len(self.net.switch) > 0:
            for idx in self.net.switch.index:
                try:
                    if str(self.net.switch.at[idx, "et"]) != "b":
                        continue
                    if not bool(self.net.switch.at[idx, "closed"]):
                        continue
                    b1 = int(self.net.switch.at[idx, "bus"])
                    b2 = int(self.net.switch.at[idx, "element"])
                    if b1 not in bus_pos or b2 not in bus_pos:
                        continue
                    i = bus_pos[b1]
                    j = bus_pos[b2]
                    # tiny impedance (avoid exact zero)
                    r_pu = 1e-4
                    x_pu = 1e-4
                    edge_src += [i, j]
                    edge_dst += [j, i]
                    edge_attr += [
                        [r_pu, x_pu, 0.0, 0.0, 1.0],
                        [r_pu, x_pu, 0.0, 0.0, 1.0],
                    ]
                except Exception:
                    continue

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32) if len(edge_attr) > 0 else np.zeros((0, 5), dtype=np.float32)

        # EVCS positions in this node array
        evcs_pos = {name: bus_pos[int(bus)] for name, bus in self.bus_of.items() if int(bus) in bus_pos}


        # Final sanitization: avoid NaN/Inf propagating into GNN/Policy
        if not np.isfinite(node_x).all():
            bad_v = ~np.isfinite(node_x[:, 0])
            node_x = np.nan_to_num(node_x, nan=0.0, posinf=0.0, neginf=0.0)
            if bad_v.any():
                node_x[bad_v, 0] = 1.0  # default voltage

        if edge_attr.size > 0 and (not np.isfinite(edge_attr).all()):
            edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        df = float(self.sg.omega / (2 * np.pi)) if (self.sg is not None and np.isfinite(self.sg.omega)) else 0.0
        rocof = float(self.sg.get_rocof_hz_s()) if (self.sg is not None and np.isfinite(self.sg.get_rocof_hz_s())) else 0.0

        return {
            "node_x": node_x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "evcs_pos": evcs_pos,
            "df": df,
            "rocof": rocof,
        }
