"""
sim_droop_v2_metrics.py
=======================
Dynamic Frequency Simulation with SG + EVCS Primary Droop Control
Enhanced with Comprehensive Metrics Tracking

Metrics Tracked:
- nadir (Hz), max |Δf| (mHz)
- max |RoCoF| (Hz/s)
- recovery time to ±200mHz / ±100mHz
- total energy BESS (kWh), total energy V2G (kWh)
- number of SoC violations / saturation events
- total losses (MW) per window

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime
import warnings
import sys
import os
from pathlib import Path
import json

warnings.filterwarnings('ignore')

# Import CIGRE MV microgrid (use same directory as this file)
_this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_this_dir))
from cigre_mv_microgrid_v1 import (
    create_cigre_mv_microgrid,
    get_network_topology,
    trip_line,
    restore_line,
    get_line_between_buses,
    set_switch_state,
)

try:
    import pandapower as pp
except ImportError:
    raise ImportError("pandapower is required. Install with: pip install pandapower")


# ============================================================================
# 1. SYSTEM CONSTANTS
# ============================================================================
F_NOM = 50.0                        # Nominal frequency [Hz]
OMEGA_0 = 2 * np.pi * F_NOM         # Nominal angular frequency [rad/s]
S_BASE_MVA = 10.0                   # System base [MVA]
S_BASE_KW = S_BASE_MVA * 1000       # System base [kW]

# Simulation timesteps
DT_DYN = 0.01                       # ODE integration step [s]
DT_PF_NORMAL = 0.2                  # Power flow step (normal) [s]
DT_PF_EVENT = 0.02                  # Power flow step (near event) [s]
EVENT_WINDOW = 60.0                 # ±60s around event for dense PF [s]

# SG Bus
SG_BUS = 1

# SG Limits
SG_P_MAX_PU = 1.2                   # 12 MW / 10 MVA
SG_P_MIN_PU = 0.05


# ============================================================================
# 2. SG PARAMETERS
# ============================================================================
H_SG = 2.0                          # Inertia constant [s]
D_PU = 1.0                          # Damping coefficient [p.u.]
M_RAD = (2 * H_SG) / OMEGA_0        # Inertia [s²/rad]
D_RAD = D_PU / OMEGA_0              # Damping [s/rad]

# Governor/Turbine
R_PERCENT = 5.0                     # Droop [%]
TG = 0.2                            # Governor time constant [s]
TT = 0.5                            # Turbine time constant [s]
DEADBAND_PRI_HZ = 0.02              # Primary deadband [Hz]

# AGC (Secondary control)
KP_AGC = 0.2                        # Proportional gain
KI_AGC = 0.1                        # Integral gain
TF_AGC = 1.5                        # Filter time constant [s]
DEADBAND_AGC_HZ = 0.02              # AGC deadband [Hz]
P_AGC_MAX_PU = 0.30                 # Max AGC contribution [p.u.]
RAMP_AGC_PU_S = 0.01                # AGC ramp rate [p.u./s]

# Dispatch
T_DISPATCH = 300.0                  # Dispatch update time constant [s]


# ============================================================================
# 3. SCENARIO CONFIGURATION
# ============================================================================
SCEN1_TIME_H = 11.0
SCEN1_TIME_S = SCEN1_TIME_H * 3600
SCEN1_LOAD_STEP_KW = 2000.0
SCEN1_BUS = 6
SCEN1_DURATION_S = 600.0
SCEN1_RAMP_ON_S = 5.0
SCEN1_RAMP_OFF_S = 5.0

SCEN2_ENABLED = False
SCEN2_TIME_H = 14.0
SCEN2_TIME_S = SCEN2_TIME_H * 3600
SCEN2_LINE_BUS1 = 14
SCEN2_LINE_BUS2 = 8
SCEN2_DURATION_S = 1800.0
SCEN2_FAULT_DURATION_S = 0.15

SIM_START_H = 10.0 + 55.0/60.0
SIM_END_H = 11.0 + 20.0/60.0
SIM_START_S = SIM_START_H * 3600
SIM_END_S = SIM_END_H * 3600


# ============================================================================
# 4. EVCS CONTROLLER WITH METRICS TRACKING
# ============================================================================

def volt_var_q(v_pu, qmax_kvar, V1=0.95, V2=0.98, V3=1.02, V4=1.05):
    """
    Volt-Var curve (IEEE 1547 style, piecewise-linear with deadband).
    
    Low V -> +Q (inject reactive power to support voltage)
    High V -> -Q (absorb reactive power to lower voltage)
    
    Parameters
    ----------
    v_pu : float
        Bus voltage in per-unit
    qmax_kvar : float
        Maximum reactive power capability [kvar]
    V1, V2, V3, V4 : float
        Voltage breakpoints [p.u.]
        V1-V2: ramp from +Qmax to 0
        V2-V3: deadband (Q=0)
        V3-V4: ramp from 0 to -Qmax
    
    Returns
    -------
    Q_kvar : float
        Reactive power command [kvar]
        Positive = inject (capacitive), Negative = absorb (inductive)
    """
    if v_pu <= V1:
        return +qmax_kvar
    if v_pu < V2:
        return +qmax_kvar * (V2 - v_pu) / max(V2 - V1, 1e-6)
    if v_pu <= V3:
        return 0.0
    if v_pu < V4:
        return -qmax_kvar * (v_pu - V3) / max(V4 - V3, 1e-6)
    return -qmax_kvar


def q_cap_from_s(S_kva, P_kw):
    """
    Calculate remaining Q capability given S rating and current P.
    
    Enforces: P² + Q² ≤ S²
    
    Parameters
    ----------
    S_kva : float
        Apparent power rating [kVA]
    P_kw : float
        Current active power [kW]
    
    Returns
    -------
    Q_cap_kvar : float
        Maximum Q magnitude available [kvar]
    """
    P = abs(P_kw)
    return np.sqrt(max(S_kva**2 - P**2, 0.0))


def get_evcs_bus_mapping(net):
    """Get EVCS bus mapping from network config."""
    config = net.get("microgrid_config", {})
    evcs_clusters = config.get("evcs_clusters", {})
    return {
        'EVCS1': int(evcs_clusters.get('EVCS1', {}).get('bus', 6)),
        'EVCS2': int(evcs_clusters.get('EVCS2', {}).get('bus', 10)),
        'EVCS3': int(evcs_clusters.get('EVCS3', {}).get('bus', 15)),
    }


class EVCSControllerInterface:
    """
    EVCS Primary Frequency Control with BESS + V2G Droop and Metrics Tracking.
    
    Tracks:
    - Energy delivered/absorbed by BESS and V2G
    - SoC violation counts (hitting min/max limits)
    - Saturation counts (power output at limits)
    """
    
    def __init__(self, net=None, evcs_config=None, enabled=True):
        self.enabled = enabled
        
        cfg = (net.get("microgrid_config", {}) if net is not None else {})
        clusters = cfg.get("evcs_clusters", {})
        
        # Build per-cluster parameters
        self.bus_of = {}
        self.p_bess_max = {}
        self.p_v2g_max = {}
        self.e_bess_kwh = {}
        self.e_v2g_kwh = {}
        
        for name, c in clusters.items():
            bus = int(c.get("bus", 0))
            self.bus_of[name] = bus
            
            e_kwh = float(c.get("bess_kwh", 0.0))
            self.e_bess_kwh[name] = e_kwh
            self.p_bess_max[name] = 0.5 * e_kwh
            
            p_ev_kw = float(c.get("p_ev_kw", 0.0))
            self.p_v2g_max[name] = 0.5 * p_ev_kw
            
            n_ev_equivalent = p_ev_kw / 50.0 if p_ev_kw > 0 else 0
            self.e_v2g_kwh[name] = n_ev_equivalent * 60 * 0.30
        
        # Fallback
        if not self.bus_of:
            self.bus_of = {'EVCS1': 6, 'EVCS2': 10, 'EVCS3': 15}
            self.e_bess_kwh = {'EVCS1': 1200, 'EVCS2': 1600, 'EVCS3': 600}
            self.p_bess_max = {'EVCS1': 600, 'EVCS2': 800, 'EVCS3': 300}
            self.p_v2g_max = {'EVCS1': 300, 'EVCS2': 400, 'EVCS3': 150}
            self.e_v2g_kwh = {'EVCS1': 216, 'EVCS2': 288, 'EVCS3': 108}
        
        # States
        self.soc_bess = {k: 0.5 for k in self.bus_of.keys()}
        self.soc_v2g = {k: 0.5 for k in self.bus_of.keys()}
        
        self.P_bess = {self.bus_of[k]: 0.0 for k in self.bus_of.keys()}
        self.P_v2g = {self.bus_of[k]: 0.0 for k in self.bus_of.keys()}
        
        # Droop settings
        self.DB = 0.02
        self.FULL = 0.20
        self.tau_f = 0.05
        self.df_lp = 0.0
        
        # Ramp limits
        self.ramp_bess = {k: 10.0 * self.p_bess_max[k] for k in self.bus_of.keys()}
        self.ramp_v2g = {k: 2.0 * self.p_v2g_max[k] for k in self.bus_of.keys()}
        
        # SoC limits
        self.soc_bess_min, self.soc_bess_max = 0.20, 0.90
        self.soc_v2g_min, self.soc_v2g_max = 0.30, 0.90
        
        # Efficiency
        self.eta_dis = 0.95
        self.eta_ch = 0.95
        
        # Hysteresis for V2G
        self.v2g_active = {k: False for k in self.bus_of.keys()}
        self.HYST_ON = 0.95
        self.HYST_OFF = 0.80
        
        # ================================================================
        # Q-V DROOP (Volt-Var) SETTINGS
        # ================================================================
        # Voltage breakpoints [p.u.] - IEEE 1547 style
        self.V1, self.V2, self.V3, self.V4 = 0.95, 0.98, 1.02, 1.05
        
        # Voltage filter time constant [s]
        self.tau_v = 0.10
        
        # Power factor for S rating (conservative baseline)
        self.pf_min = 0.90
        
        # S ratings [kVA] - based on Pmax/pf_min
        self.S_bess_kva = {k: self.p_bess_max[k] / self.pf_min for k in self.bus_of.keys()}
        self.S_v2g_kva = {k: self.p_v2g_max[k] / self.pf_min for k in self.bus_of.keys()}
        
        # Q ramp limits [kvar/s] - fast like BESS P ramp
        self.ramp_q_bess = {k: 10.0 * (self.S_bess_kva[k] * np.sqrt(1 - self.pf_min**2)) 
                           for k in self.bus_of.keys()}
        self.ramp_q_v2g = {k: 2.0 * (self.S_v2g_kva[k] * np.sqrt(1 - self.pf_min**2)) 
                          for k in self.bus_of.keys()}
        
        # Q states [kvar]
        self.Q_bess = {self.bus_of[k]: 0.0 for k in self.bus_of.keys()}
        self.Q_v2g = {self.bus_of[k]: 0.0 for k in self.bus_of.keys()}
        
        # Filtered voltage [p.u.]
        self.v_lp = {self.bus_of[k]: 1.0 for k in self.bus_of.keys()}
        
        # Q-V closed-loop delay tracking
        self._qv_last_update_t = 0.0
        self._qv_update_count = 0

        # Initialize metrics tracking (must exist before step())
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics containers used by step() and summaries."""
        # ================================================================
        # METRICS TRACKING
        # ================================================================
        self.metrics = {
            # Energy accounting [kWh]
            'energy_bess_discharge_kwh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_bess_charge_kwh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_v2g_discharge_kwh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_v2g_charge_kwh': {k: 0.0 for k in self.bus_of.keys()},
    
            # SoC violation counts
            'soc_bess_min_violations': {k: 0 for k in self.bus_of.keys()},
            'soc_bess_max_violations': {k: 0 for k in self.bus_of.keys()},
            'soc_v2g_min_violations': {k: 0 for k in self.bus_of.keys()},
            'soc_v2g_max_violations': {k: 0 for k in self.bus_of.keys()},
    
            # Saturation counts
            'bess_saturation_count': {k: 0 for k in self.bus_of.keys()},
            'v2g_saturation_count': {k: 0 for k in self.bus_of.keys()},
    
            # Q-V droop metrics
            'energy_q_bess_inject_kvarh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_q_bess_absorb_kvarh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_q_v2g_inject_kvarh': {k: 0.0 for k in self.bus_of.keys()},
            'energy_q_v2g_absorb_kvarh': {k: 0.0 for k in self.bus_of.keys()},
            'q_bess_saturation_count': {k: 0 for k in self.bus_of.keys()},
            'q_v2g_saturation_count': {k: 0 for k in self.bus_of.keys()},
    
            # Last state for edge detection
            '_bess_was_saturated': {k: False for k in self.bus_of.keys()},
            '_v2g_was_saturated': {k: False for k in self.bus_of.keys()},
            '_q_bess_was_saturated': {k: False for k in self.bus_of.keys()},
            '_q_v2g_was_saturated': {k: False for k in self.bus_of.keys()},
            '_bess_was_at_soc_min': {k: False for k in self.bus_of.keys()},
            '_bess_was_at_soc_max': {k: False for k in self.bus_of.keys()},
            '_v2g_was_at_soc_min': {k: False for k in self.bus_of.keys()},
            '_v2g_was_at_soc_max': {k: False for k in self.bus_of.keys()},
        }

    def tune_qv_droop(self, tau_v=None, v_deadband_width=None, q_ramp_factor=None):
        """
        Tune Q-V droop parameters to avoid oscillations.
        
        Call this if voltage/Q oscillations are observed.
        
        Parameters
        ----------
        tau_v : float, optional
            Voltage filter time constant [s]. Increase to smooth voltage.
            Default: 0.10, Range: [0.05, 0.5]
        v_deadband_width : float, optional
            Width of voltage deadband [p.u.]. 
            V2-V1 and V4-V3 will be set to this value.
            Wider deadband = less aggressive Q response.
            Default: 0.03, Range: [0.02, 0.10]
        q_ramp_factor : float, optional
            Multiplier for Q ramp rate. Lower = slower response.
            Default: 1.0, Range: [0.1, 2.0]
        
        Example
        -------
        # More conservative settings for high R/X network:
        evcs.tune_qv_droop(tau_v=0.2, v_deadband_width=0.05, q_ramp_factor=0.5)
        """
        if tau_v is not None:
            self.tau_v = float(np.clip(tau_v, 0.05, 0.5))
            print(f"  Q-V: tau_v = {self.tau_v:.2f} s")
        
        if v_deadband_width is not None:
            width = float(np.clip(v_deadband_width, 0.02, 0.10))
            # Symmetric adjustment
            v_center_low = (self.V1 + self.V2) / 2
            v_center_high = (self.V3 + self.V4) / 2
            self.V1 = v_center_low - width / 2
            self.V2 = v_center_low + width / 2
            self.V3 = v_center_high - width / 2
            self.V4 = v_center_high + width / 2
            print(f"  Q-V: V breakpoints = [{self.V1:.3f}, {self.V2:.3f}, {self.V3:.3f}, {self.V4:.3f}]")
        
        if q_ramp_factor is not None:
            factor = float(np.clip(q_ramp_factor, 0.1, 2.0))
            for k in self.bus_of.keys():
                base_qmax_bess = self.S_bess_kva[k] * np.sqrt(1 - self.pf_min**2)
                base_qmax_v2g = self.S_v2g_kva[k] * np.sqrt(1 - self.pf_min**2)
                self.ramp_q_bess[k] = 10.0 * base_qmax_bess * factor
                self.ramp_q_v2g[k] = 2.0 * base_qmax_v2g * factor
            print(f"  Q-V: ramp factor = {factor:.2f}")
        
        # Ensure metrics exist (don't reset existing counters)
        if not hasattr(self, 'metrics'):
            self._init_metrics()
    
    def _droop_command_kw(self, df_hz, pmax_kw):
        if abs(df_hz) <= self.DB:
            return 0.0
        df_eff = df_hz - (self.DB * np.sign(df_hz))
        K = pmax_kw / max(self.FULL - self.DB, 1e-6)
        P = -K * df_eff
        return float(np.clip(P, -pmax_kw, +pmax_kw))
    
    def update_voltage_measurements(self, net, dt_s):
        """
        Update filtered voltage measurements from power flow results.
        Call this after each successful pp.runpp().
        
        Parameters
        ----------
        net : pandapower network
            Network with converged power flow results
        dt_s : float
            Time since last voltage update [s]
        """
        if not self.enabled:
            return
        
        for name, bus in self.bus_of.items():
            try:
                v_meas = float(net.res_bus.at[bus, "vm_pu"])
            except (KeyError, IndexError):
                v_meas = 1.0  # Default if bus result not available
            
            # Low-pass filter
            alpha = dt_s / (self.tau_v + dt_s)
            self.v_lp[bus] += alpha * (v_meas - self.v_lp[bus])
    
    def step(self, freq_dev_hz, dt_s):
        if not self.enabled:
            return
        
        # LPF
        alpha = dt_s / (self.tau_f + dt_s)
        self.df_lp += alpha * (freq_dev_hz - self.df_lp)
        df = self.df_lp
        
        for name, bus in self.bus_of.items():
            pmax_b = self.p_bess_max[name]
            P_cmd_b = self._droop_command_kw(df, pmax_b)
            
            # SoC gating BESS
            soc = self.soc_bess[name]
            soc_blocked_min = False
            soc_blocked_max = False
            
            if P_cmd_b > 0 and soc <= self.soc_bess_min:
                P_cmd_b = 0.0
                soc_blocked_min = True
            if P_cmd_b < 0 and soc >= self.soc_bess_max:
                P_cmd_b = 0.0
                soc_blocked_max = True
            
            # Track SoC violations (edge detection)
            if soc_blocked_min and not self.metrics['_bess_was_at_soc_min'][name]:
                self.metrics['soc_bess_min_violations'][name] += 1
            if soc_blocked_max and not self.metrics['_bess_was_at_soc_max'][name]:
                self.metrics['soc_bess_max_violations'][name] += 1
            self.metrics['_bess_was_at_soc_min'][name] = soc_blocked_min
            self.metrics['_bess_was_at_soc_max'][name] = soc_blocked_max
            
            # Ramp limit BESS
            P_prev = self.P_bess[bus]
            dP_max = self.ramp_bess[name] * dt_s
            dP = np.clip(P_cmd_b - P_prev, -dP_max, dP_max)
            P_bess = P_prev + dP
            self.P_bess[bus] = P_bess
            
            # Track BESS saturation
            EPS = 1e-3
            bess_saturated = abs(P_bess) >= 0.98 * pmax_b
            if bess_saturated and not self.metrics['_bess_was_saturated'][name]:
                self.metrics['bess_saturation_count'][name] += 1
            self.metrics['_bess_was_saturated'][name] = bess_saturated
            
            # V2G Control
            pmax_v = self.p_v2g_max[name]
            P_bess_actual = self.P_bess[bus]
            
            need_discharge = (df < -self.DB)
            need_charge = (df > self.DB)
            
            if need_discharge and pmax_b > EPS:
                bess_util = P_bess_actual / pmax_b
            elif need_charge and pmax_b > EPS:
                bess_util = -P_bess_actual / pmax_b
            else:
                bess_util = 0.0
            
            bess_blocked = soc_blocked_min or soc_blocked_max
            
            if bess_util >= self.HYST_ON or bess_blocked:
                self.v2g_active[name] = True
            elif bess_util <= self.HYST_OFF and not bess_blocked:
                self.v2g_active[name] = False
            
            need_v2g = self.v2g_active[name] and pmax_v > EPS
            
            if need_v2g:
                P_cmd_v = self._droop_command_kw(df, pmax_v)
                
                socv = self.soc_v2g[name]
                v2g_soc_blocked_min = False
                v2g_soc_blocked_max = False
                
                if P_cmd_v > 0 and socv <= self.soc_v2g_min:
                    P_cmd_v = 0.0
                    v2g_soc_blocked_min = True
                if P_cmd_v < 0 and socv >= self.soc_v2g_max:
                    P_cmd_v = 0.0
                    v2g_soc_blocked_max = True
                
                # Track V2G SoC violations
                if v2g_soc_blocked_min and not self.metrics['_v2g_was_at_soc_min'][name]:
                    self.metrics['soc_v2g_min_violations'][name] += 1
                if v2g_soc_blocked_max and not self.metrics['_v2g_was_at_soc_max'][name]:
                    self.metrics['soc_v2g_max_violations'][name] += 1
                self.metrics['_v2g_was_at_soc_min'][name] = v2g_soc_blocked_min
                self.metrics['_v2g_was_at_soc_max'][name] = v2g_soc_blocked_max
                
                P_prev_v = self.P_v2g[bus]
                dP_max_v = self.ramp_v2g[name] * dt_s
                dP_v = np.clip(P_cmd_v - P_prev_v, -dP_max_v, dP_max_v)
                P_v2g = P_prev_v + dP_v
                self.P_v2g[bus] = P_v2g
            else:
                P_prev_v = self.P_v2g[bus]
                dP_max_v = self.ramp_v2g[name] * dt_s
                dP_v = np.clip(-P_prev_v, -dP_max_v, dP_max_v)
                self.P_v2g[bus] = P_prev_v + dP_v
            
            # Track V2G saturation
            v2g_saturated = abs(self.P_v2g[bus]) >= 0.98 * pmax_v if pmax_v > EPS else False
            if v2g_saturated and not self.metrics['_v2g_was_saturated'][name]:
                self.metrics['v2g_saturation_count'][name] += 1
            self.metrics['_v2g_was_saturated'][name] = v2g_saturated
            
            # ================================================================
            # Q-V DROOP (Volt-Var) CONTROL
            # ================================================================
            # Get filtered voltage
            vpu = self.v_lp.get(bus, 1.0)
            
            # S ratings and current P values
            S_bess = self.S_bess_kva[name]
            S_v2g = self.S_v2g_kva[name]
            P_bess_actual = self.P_bess[bus]
            P_v2g_actual = self.P_v2g[bus]
            
            # Compute Q capability (headroom after P)
            Qcap_b = q_cap_from_s(S_bess, P_bess_actual)
            Qcap_v = q_cap_from_s(S_v2g, P_v2g_actual)
            
            # Compute Q request from Volt-Var curve
            Qmax_total = Qcap_b + Qcap_v
            Qreq = volt_var_q(vpu, Qmax_total, self.V1, self.V2, self.V3, self.V4)
            
            # Allocate Q: BESS first, then V2G
            Qb_tgt = np.clip(Qreq, -Qcap_b, +Qcap_b)
            Qrem = Qreq - Qb_tgt
            Qv_tgt = np.clip(Qrem, -Qcap_v, +Qcap_v)
            
            # Ramp-limit Q
            Qb_prev = self.Q_bess.get(bus, 0.0)
            Qv_prev = self.Q_v2g.get(bus, 0.0)
            dQb_max = self.ramp_q_bess[name] * dt_s
            dQv_max = self.ramp_q_v2g[name] * dt_s
            
            self.Q_bess[bus] = Qb_prev + np.clip(Qb_tgt - Qb_prev, -dQb_max, +dQb_max)
            self.Q_v2g[bus] = Qv_prev + np.clip(Qv_tgt - Qv_prev, -dQv_max, +dQv_max)
            
            # Track Q saturation (when Q at capability limit)
            q_bess_saturated = abs(self.Q_bess[bus]) >= 0.98 * Qcap_b if Qcap_b > EPS else False
            if q_bess_saturated and not self.metrics['_q_bess_was_saturated'][name]:
                self.metrics['q_bess_saturation_count'][name] += 1
            self.metrics['_q_bess_was_saturated'][name] = q_bess_saturated
            
            q_v2g_saturated = abs(self.Q_v2g[bus]) >= 0.98 * Qcap_v if Qcap_v > EPS else False
            if q_v2g_saturated and not self.metrics['_q_v2g_was_saturated'][name]:
                self.metrics['q_v2g_saturation_count'][name] += 1
            self.metrics['_q_v2g_was_saturated'][name] = q_v2g_saturated
            
            # Energy accounting
            dt_h = dt_s / 3600.0
            
            # BESS energy
            Pm = self.P_bess[bus]
            if Pm > 0:  # Discharge
                self.metrics['energy_bess_discharge_kwh'][name] += Pm * dt_h
            else:  # Charge
                self.metrics['energy_bess_charge_kwh'][name] += abs(Pm) * dt_h
            
            # V2G energy
            Pv = self.P_v2g[bus]
            if Pv > 0:  # Discharge
                self.metrics['energy_v2g_discharge_kwh'][name] += Pv * dt_h
            else:  # Charge
                self.metrics['energy_v2g_charge_kwh'][name] += abs(Pv) * dt_h
            
            # Q energy accounting (kvarh)
            Qb = self.Q_bess[bus]
            if Qb > 0:  # Inject (capacitive)
                self.metrics['energy_q_bess_inject_kvarh'][name] += Qb * dt_h
            else:  # Absorb (inductive)
                self.metrics['energy_q_bess_absorb_kvarh'][name] += abs(Qb) * dt_h
            
            Qv = self.Q_v2g[bus]
            if Qv > 0:  # Inject
                self.metrics['energy_q_v2g_inject_kvarh'][name] += Qv * dt_h
            else:  # Absorb
                self.metrics['energy_q_v2g_absorb_kvarh'][name] += abs(Qv) * dt_h
            
            # SoC update
            E_bess = max(self.e_bess_kwh[name], 1e-6)
            if Pm >= 0:
                self.soc_bess[name] -= (Pm / self.eta_dis) * dt_h / E_bess
            else:
                self.soc_bess[name] -= (Pm * self.eta_ch) * dt_h / E_bess
            self.soc_bess[name] = float(np.clip(self.soc_bess[name], 0.0, 1.0))
            
            E_v2g = max(self.e_v2g_kwh[name], 1e-6)
            if Pv >= 0:
                self.soc_v2g[name] -= (Pv / self.eta_dis) * dt_h / E_v2g
            else:
                self.soc_v2g[name] -= (Pv * self.eta_ch) * dt_h / E_v2g
            self.soc_v2g[name] = float(np.clip(self.soc_v2g[name], 0.0, 1.0))
    
    def get_injections_kw(self):
        return self.P_bess.copy(), self.P_v2g.copy()
    
    def get_q_injections_kvar(self):
        """Get reactive power injections [kvar]."""
        return self.Q_bess.copy(), self.Q_v2g.copy()
    
    def get_total_response_kw(self):
        return sum(self.P_bess.values()) + sum(self.P_v2g.values())
    
    def get_total_q_response_kvar(self):
        """Get total reactive power response [kvar]."""
        return sum(self.Q_bess.values()) + sum(self.Q_v2g.values())
    
    def get_soc_status(self):
        return {'bess': self.soc_bess.copy(), 'v2g': self.soc_v2g.copy()}
    
    def get_voltage_status(self):
        """Get filtered voltage measurements [p.u.]."""
        return self.v_lp.copy()
    
    def detect_qv_oscillations(self, results_df, threshold_kvar=50, min_reversals=5):
        """
        Detect potential Q-V oscillations from simulation results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Simulation results with Q_bess_kvar, Q_v2g_kvar columns
        threshold_kvar : float
            Minimum Q change to count as a reversal [kvar]
        min_reversals : int
            Minimum reversals to flag as oscillation
        
        Returns
        -------
        oscillation_detected : bool
        details : dict
            {cluster: reversal_count, ...}
        """
        details = {}
        oscillation_detected = False
        
        for cluster in ['EVCS1', 'EVCS2', 'EVCS3']:
            q_col = f'Q_bess_{cluster}_kvar'
            if q_col in results_df.columns:
                q_data = results_df[q_col].values
                
                # Count sign reversals with significant magnitude
                dq = np.diff(q_data)
                sign_changes = np.diff(np.sign(dq))
                significant = np.abs(dq[1:]) > threshold_kvar
                reversals = int(np.sum((sign_changes != 0) & significant))
                
                details[cluster] = reversals
                if reversals >= min_reversals:
                    oscillation_detected = True
        
        return oscillation_detected, details
    
    def get_metrics_summary(self):
        """Get summary of all tracked metrics."""
        summary = {
            # Total P energy [kWh]
            'total_bess_discharge_kwh': sum(self.metrics['energy_bess_discharge_kwh'].values()),
            'total_bess_charge_kwh': sum(self.metrics['energy_bess_charge_kwh'].values()),
            'total_v2g_discharge_kwh': sum(self.metrics['energy_v2g_discharge_kwh'].values()),
            'total_v2g_charge_kwh': sum(self.metrics['energy_v2g_charge_kwh'].values()),
            
            # Net P energy (discharge - charge)
            'net_bess_energy_kwh': sum(self.metrics['energy_bess_discharge_kwh'].values()) - 
                                   sum(self.metrics['energy_bess_charge_kwh'].values()),
            'net_v2g_energy_kwh': sum(self.metrics['energy_v2g_discharge_kwh'].values()) - 
                                  sum(self.metrics['energy_v2g_charge_kwh'].values()),
            
            # Total Q energy [kvarh]
            'total_q_bess_inject_kvarh': sum(self.metrics['energy_q_bess_inject_kvarh'].values()),
            'total_q_bess_absorb_kvarh': sum(self.metrics['energy_q_bess_absorb_kvarh'].values()),
            'total_q_v2g_inject_kvarh': sum(self.metrics['energy_q_v2g_inject_kvarh'].values()),
            'total_q_v2g_absorb_kvarh': sum(self.metrics['energy_q_v2g_absorb_kvarh'].values()),
            
            # Net Q energy (inject - absorb)
            'net_q_bess_kvarh': sum(self.metrics['energy_q_bess_inject_kvarh'].values()) - 
                                sum(self.metrics['energy_q_bess_absorb_kvarh'].values()),
            'net_q_v2g_kvarh': sum(self.metrics['energy_q_v2g_inject_kvarh'].values()) - 
                               sum(self.metrics['energy_q_v2g_absorb_kvarh'].values()),
            
            # SoC violations
            'total_soc_bess_min_violations': sum(self.metrics['soc_bess_min_violations'].values()),
            'total_soc_bess_max_violations': sum(self.metrics['soc_bess_max_violations'].values()),
            'total_soc_v2g_min_violations': sum(self.metrics['soc_v2g_min_violations'].values()),
            'total_soc_v2g_max_violations': sum(self.metrics['soc_v2g_max_violations'].values()),
            
            # P saturation counts
            'total_bess_saturation_count': sum(self.metrics['bess_saturation_count'].values()),
            'total_v2g_saturation_count': sum(self.metrics['v2g_saturation_count'].values()),
            
            # Q saturation counts
            'total_q_bess_saturation_count': sum(self.metrics['q_bess_saturation_count'].values()),
            'total_q_v2g_saturation_count': sum(self.metrics['q_v2g_saturation_count'].values()),
            
            # Per-cluster details
            'per_cluster': {}
        }
        
        for name in self.bus_of.keys():
            summary['per_cluster'][name] = {
                # P energy
                'bess_discharge_kwh': self.metrics['energy_bess_discharge_kwh'][name],
                'bess_charge_kwh': self.metrics['energy_bess_charge_kwh'][name],
                'v2g_discharge_kwh': self.metrics['energy_v2g_discharge_kwh'][name],
                'v2g_charge_kwh': self.metrics['energy_v2g_charge_kwh'][name],
                # Q energy
                'q_bess_inject_kvarh': self.metrics['energy_q_bess_inject_kvarh'][name],
                'q_bess_absorb_kvarh': self.metrics['energy_q_bess_absorb_kvarh'][name],
                'q_v2g_inject_kvarh': self.metrics['energy_q_v2g_inject_kvarh'][name],
                'q_v2g_absorb_kvarh': self.metrics['energy_q_v2g_absorb_kvarh'][name],
                # SoC violations
                'soc_bess_min_violations': self.metrics['soc_bess_min_violations'][name],
                'soc_bess_max_violations': self.metrics['soc_bess_max_violations'][name],
                'soc_v2g_min_violations': self.metrics['soc_v2g_min_violations'][name],
                'soc_v2g_max_violations': self.metrics['soc_v2g_max_violations'][name],
                # Saturation counts
                'bess_saturation_count': self.metrics['bess_saturation_count'][name],
                'v2g_saturation_count': self.metrics['v2g_saturation_count'][name],
                'q_bess_saturation_count': self.metrics['q_bess_saturation_count'][name],
                'q_v2g_saturation_count': self.metrics['q_v2g_saturation_count'][name],
                # Final states
                'final_soc_bess': self.soc_bess[name],
                'final_soc_v2g': self.soc_v2g[name],
            }
        
        return summary


# ============================================================================
# 5. PROFILE GENERATORS (same as before)
# ============================================================================

def build_ev_distribution(net):
    buses = get_evcs_bus_mapping(net)
    return {buses['EVCS1']: 0.35, buses['EVCS2']: 0.47, buses['EVCS3']: 0.18}


def build_pv_allocation(net):
    buses = get_evcs_bus_mapping(net)
    pv_alloc = {4: 500, buses['EVCS1']: 350, 9: 350, buses['EVCS2']: 350, buses['EVCS3']: 150}
    return pv_alloc, sum(pv_alloc.values())


LOAD_DIST = {
    1: 0.00, 2: 0.05, 3: 0.10, 4: 0.05, 5: 0.10,
    6: 0.08, 7: 0.08, 8: 0.10, 9: 0.05, 10: 0.08,
    11: 0.10, 12: 0.08, 13: 0.05, 14: 0.08,
}

WIND_BUS = 13


def create_profiles_for_window(t_start_s, t_end_s, dt=1.0):
    times = np.arange(t_start_s, t_end_s + dt, dt)
    n = len(times)
    profiles = {'t_s': times, 'P_load_kW': np.zeros(n), 'P_pv_kW': np.zeros(n),
                'P_wind_kW': np.zeros(n), 'P_ev_kW': np.zeros(n)}
    for i, t in enumerate(times):
        t_h = t / 3600.0
        profiles['P_load_kW'][i] = 5500 + 200 * np.sin((t_h - 10.5) * np.pi / 2)
        profiles['P_pv_kW'][i] = 1700 * 0.6 * (1 + 0.05 * np.sin((t_h - 11.0) * np.pi * 2))
        profiles['P_wind_kW'][i] = 1000 * 0.5 * (1 + 0.1 * np.sin((t_h - 10.0) * np.pi))
        profiles['P_ev_kW'][i] = 1700 * 0.25 * (1 + 0.1 * (t_h - 10.5))
    return profiles


def get_profile_at_time(profiles, t_s):
    times = profiles['t_s']
    return {key: np.interp(t_s, times, values) for key, values in profiles.items() if key != 't_s'}


def get_scenario1_load_step(t_s):
    t_event = SCEN1_TIME_S
    t_end = t_event + SCEN1_DURATION_S
    if t_s < t_event:
        return 0.0
    elif t_s < t_event + SCEN1_RAMP_ON_S:
        return SCEN1_LOAD_STEP_KW * (t_s - t_event) / SCEN1_RAMP_ON_S
    elif t_s < t_end - SCEN1_RAMP_OFF_S:
        return SCEN1_LOAD_STEP_KW
    elif t_s < t_end:
        return SCEN1_LOAD_STEP_KW * (t_end - t_s) / SCEN1_RAMP_OFF_S
    return 0.0


# ============================================================================
# 6. NETWORK SETUP & POWER FLOW
# ============================================================================

def setup_network():
    net = create_cigre_mv_microgrid(mesh_level=2)
    
    for idx in net.trafo.index:
        if net.trafo.at[idx, 'hv_bus'] == 0:
            net.trafo.at[idx, 'in_service'] = False
    net.bus.at[0, 'in_service'] = False
    
    ev_dist = build_ev_distribution(net)
    pv_allocation, pv_total = build_pv_allocation(net)
    evcs_buses = get_evcs_bus_mapping(net)
    
    if len(net.load) > 0:
        net.load.drop(net.load.index, inplace=True)
    if len(net.sgen) > 0:
        net.sgen.drop(net.sgen.index, inplace=True)
    if len(net.storage) > 0:
        net.storage.drop(net.storage.index, inplace=True)
    
    element_indices = {
        'loads': {}, 'ev_loads': {}, 'pv_sgens': {}, 'wind_sgen': None, 'step_load': None,
        'evcs_bess_sgens': {}, 'evcs_v2g_sgens': {},
        'ev_dist': ev_dist, 'pv_allocation': pv_allocation, 'pv_total': pv_total, 'evcs_buses': evcs_buses,
    }
    
    n_bus = len(net.bus)
    
    for bus, frac in LOAD_DIST.items():
        if bus < n_bus and frac > 0:
            element_indices['loads'][bus] = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"Load_Bus{bus}")
    
    for bus, frac in ev_dist.items():
        if bus < n_bus:
            element_indices['ev_loads'][bus] = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"EV_Bus{bus}")
    
    for bus, p_kw in pv_allocation.items():
        if bus < n_bus:
            element_indices['pv_sgens'][bus] = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"PV_Bus{bus}", type="PV")
    
    if WIND_BUS < n_bus:
        element_indices['wind_sgen'] = pp.create_sgen(net, bus=WIND_BUS, p_mw=0.0, q_mvar=0.0, name="Wind_13", type="WP")
    
    if SCEN1_BUS < n_bus:
        element_indices['step_load'] = pp.create_load(net, bus=SCEN1_BUS, p_mw=0.0, q_mvar=0.0, name="LoadStep_Scen1")
    
    for name, bus in evcs_buses.items():
        if bus < n_bus:
            element_indices['evcs_bess_sgens'][bus] = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"{name}_BESS", type="BESS")
            element_indices['evcs_v2g_sgens'][bus] = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"{name}_V2G", type="V2G")
    
    if len(net.gen) > 0:
        net.gen.at[0, 'slack'] = True
    
    return net, element_indices


def update_network_injections(net, elem_idx, t_s, profiles_full, P_bess_kw=None, P_v2g_kw=None, Q_bess_kvar=None, Q_v2g_kvar=None):
    profiles = get_profile_at_time(profiles_full, t_s)
    P_load, P_ev, P_pv, P_wind = profiles['P_load_kW'], profiles['P_ev_kW'], profiles['P_pv_kW'], profiles['P_wind_kW']
    
    ev_dist = elem_idx.get('ev_dist', {})
    pv_allocation = elem_idx.get('pv_allocation', {})
    pv_total = elem_idx.get('pv_total', 1700)
    
    pf_load = 0.9
    tan_phi = np.tan(np.arccos(pf_load))
    
    for bus, idx in elem_idx['loads'].items():
        frac = LOAD_DIST.get(bus, 0)
        p_mw = P_load * frac / 1000.0
        net.load.at[idx, 'p_mw'] = p_mw
        net.load.at[idx, 'q_mvar'] = p_mw * tan_phi
    
    for bus, idx in elem_idx['ev_loads'].items():
        net.load.at[idx, 'p_mw'] = P_ev * ev_dist.get(bus, 0) / 1000.0
        net.load.at[idx, 'q_mvar'] = 0.0
    
    for bus, idx in elem_idx['pv_sgens'].items():
        p_alloc = pv_allocation.get(bus, 0)
        net.sgen.at[idx, 'p_mw'] = P_pv * (p_alloc / pv_total) / 1000.0 if pv_total > 0 else 0.0
    
    if elem_idx['wind_sgen'] is not None:
        net.sgen.at[elem_idx['wind_sgen'], 'p_mw'] = P_wind / 1000.0
    
    P_step = get_scenario1_load_step(t_s)
    if elem_idx['step_load'] is not None:
        net.load.at[elem_idx['step_load'], 'p_mw'] = P_step / 1000.0
        net.load.at[elem_idx['step_load'], 'q_mvar'] = P_step / 1000.0 * tan_phi
    
    P_bess_kw = P_bess_kw or {}
    P_v2g_kw = P_v2g_kw or {}
    Q_bess_kvar = Q_bess_kvar or {}
    Q_v2g_kvar = Q_v2g_kvar or {}
    
    # Set EVCS P and Q injections
    for bus, sgen_idx in elem_idx.get('evcs_bess_sgens', {}).items():
        net.sgen.at[sgen_idx, 'p_mw'] = P_bess_kw.get(bus, 0.0) / 1000.0
        net.sgen.at[sgen_idx, 'q_mvar'] = Q_bess_kvar.get(bus, 0.0) / 1000.0  # Q-V droop
    
    for bus, sgen_idx in elem_idx.get('evcs_v2g_sgens', {}).items():
        net.sgen.at[sgen_idx, 'p_mw'] = P_v2g_kw.get(bus, 0.0) / 1000.0
        net.sgen.at[sgen_idx, 'q_mvar'] = Q_v2g_kvar.get(bus, 0.0) / 1000.0  # Q-V droop
    
    return {'P_load_kW': P_load, 'P_ev_kW': P_ev, 'P_pv_kW': P_pv, 'P_wind_kW': P_wind, 
            'P_step_kW': P_step, 'P_bess_kW': sum(P_bess_kw.values()), 'P_v2g_kW': sum(P_v2g_kw.values()),
            'Q_bess_kvar': sum(Q_bess_kvar.values()), 'Q_v2g_kvar': sum(Q_v2g_kvar.values())}


def run_power_flow(net, init='results'):
    try:
        pp.runpp(net, algorithm='nr', init=init, max_iteration=30, tolerance_mva=1e-6)
        if net.converged:
            P_e_mw = net.res_gen.p_mw.sum() if len(net.res_gen) > 0 else (net.res_ext_grid.p_mw.sum() if len(net.res_ext_grid) > 0 else None)
            P_loss_mw = net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.res_trafo) > 0 else 0)
            return True, P_e_mw, P_loss_mw
    except:
        pass
    
    try:
        pp.runpp(net, algorithm='nr', init='flat', max_iteration=100, tolerance_mva=1e-5)
        if net.converged:
            P_e_mw = net.res_gen.p_mw.sum() if len(net.res_gen) > 0 else (net.res_ext_grid.p_mw.sum() if len(net.res_ext_grid) > 0 else None)
            P_loss_mw = net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.res_trafo) > 0 else 0)
            return True, P_e_mw, P_loss_mw
    except:
        pass
    
    return False, None, 0.0


# ============================================================================
# 7. SYNCHRONOUS GENERATOR MODEL
# ============================================================================

class SynchronousGenerator:
    def __init__(self):
        self.omega = 0.0
        self.delta = 0.0
        self.P_m = 0.5
        self.P_ref = 0.5
        self.P_droop = 0.0
        self.P_agc = 0.0
        self.agc_integral = 0.0
        self.f_filtered = 0.0
        self.P_nom = 0.5
    
    def update_dispatch(self, P_e_pu, dt):
        alpha = dt / (T_DISPATCH + dt)
        self.P_nom += alpha * (P_e_pu - self.P_nom)
        self.P_nom = np.clip(self.P_nom, SG_P_MIN_PU, SG_P_MAX_PU)
    
    def primary_control(self, freq_dev_hz, dt):
        df_eff = 0.0 if abs(freq_dev_hz) <= DEADBAND_PRI_HZ else freq_dev_hz - DEADBAND_PRI_HZ * np.sign(freq_dev_hz)
        R_pu = R_PERCENT / 100.0
        P_droop_cmd = -df_eff / (R_pu * F_NOM)
        
        alpha_g = dt / (TG + dt)
        P_target = np.clip(self.P_nom + P_droop_cmd + self.P_agc, SG_P_MIN_PU, SG_P_MAX_PU)
        self.P_ref += alpha_g * (P_target - self.P_ref)
        
        alpha_t = dt / (TT + dt)
        self.P_m += alpha_t * (self.P_ref - self.P_m)
        self.P_droop = P_droop_cmd
    
    def secondary_control(self, freq_dev_hz, dt):
        alpha_f = dt / (TF_AGC + dt)
        self.f_filtered += alpha_f * (freq_dev_hz - self.f_filtered)
        
        f_eff = 0.0 if abs(self.f_filtered) <= DEADBAND_AGC_HZ else self.f_filtered - DEADBAND_AGC_HZ * np.sign(self.f_filtered)
        
        P_prop = -KP_AGC * f_eff
        self.agc_integral += -KI_AGC * f_eff * dt
        self.agc_integral = np.clip(self.agc_integral, -P_AGC_MAX_PU, P_AGC_MAX_PU)
        
        P_agc_cmd = P_prop + self.agc_integral
        dP_max = RAMP_AGC_PU_S * dt
        dP = np.clip(P_agc_cmd - self.P_agc, -dP_max, dP_max)
        self.P_agc = np.clip(self.P_agc + dP, -P_AGC_MAX_PU, P_AGC_MAX_PU)
    
    def swing_equation(self, P_e_pu, dt):
        domega_dt = (self.P_m - P_e_pu - D_RAD * self.omega) / M_RAD
        self.omega += domega_dt * dt
        self.delta += self.omega * dt
    
    def get_frequency_hz(self):
        return F_NOM + self.omega / (2 * np.pi)
    
    def get_freq_dev_hz(self):
        return self.omega / (2 * np.pi)
    
    def get_rocof(self, P_e_pu):
        domega_dt = (self.P_m - P_e_pu - D_RAD * self.omega) / M_RAD
        return domega_dt / (2 * np.pi)


# ============================================================================
# 8. MAIN SIMULATION
# ============================================================================

def run_simulation(enable_evcs=True):
    print("=" * 70)
    print("SIMULATION: SG + EVCS with Comprehensive Metrics")
    print("=" * 70)
    print(f"EVCS Control: {'ENABLED' if enable_evcs else 'DISABLED (baseline)'}")
    print(f"Simulation: {SIM_START_H:.4f}h to {SIM_END_H:.4f}h")
    print(f"Scenario 1: Load step +{SCEN1_LOAD_STEP_KW:.0f} kW at {SCEN1_TIME_H:.2f}h")
    
    net, elem_idx = setup_network()
    evcs = EVCSControllerInterface(net=net, enabled=enable_evcs)
    
    print(f"EVCS clusters: {list(evcs.bus_of.keys())}")
    for name in evcs.bus_of.keys():
        print(f"  {name}: bus={evcs.bus_of[name]}, BESS={evcs.p_bess_max[name]:.0f}kW/{evcs.e_bess_kwh[name]:.0f}kWh, V2G={evcs.p_v2g_max[name]:.0f}kW")
    
    sg = SynchronousGenerator()
    profiles_full = create_profiles_for_window(SIM_START_S, SIM_END_S)
    
    P_bess_kw, P_v2g_kw = evcs.get_injections_kw()
    Q_bess_kvar, Q_v2g_kvar = evcs.get_q_injections_kvar()
    update_network_injections(net, elem_idx, SIM_START_S, profiles_full, P_bess_kw, P_v2g_kw, Q_bess_kvar, Q_v2g_kvar)
    converged, P_e_mw, P_loss_mw = run_power_flow(net, init='flat')
    
    if not converged:
        print("ERROR: Initial power flow failed!")
        return None, None
    
    # Initialize voltage measurements
    evcs.update_voltage_measurements(net, DT_PF_EVENT)
    
    P_e_pu = P_e_mw / S_BASE_MVA
    sg.P_nom = sg.P_ref = sg.P_m = P_e_pu
    
    print(f"Initial P_e = {P_e_mw:.3f} MW, Losses = {P_loss_mw*1000:.1f} kW")
    
    log_data = []
    t = SIM_START_S
    last_pf_t = t
    last_dispatch_t = t
    P_e_curr = P_e_pu
    P_loss_curr = P_loss_mw
    pf_count = 0
    total_loss_energy_kwh = 0.0
    
    scen2_line_idx = None
    scen2_tripped = False
    
    print("\nRunning simulation...")
    
    while t <= SIM_END_S:
        near_scen1 = abs(t - SCEN1_TIME_S) < EVENT_WINDOW or abs(t - (SCEN1_TIME_S + SCEN1_DURATION_S)) < EVENT_WINDOW
        near_event = near_scen1
        dt_pf = DT_PF_EVENT if near_event else DT_PF_NORMAL
        
        freq_dev_hz = sg.get_freq_dev_hz()
        
        evcs.step(freq_dev_hz, DT_DYN)
        P_bess_kw, P_v2g_kw = evcs.get_injections_kw()
        Q_bess_kvar, Q_v2g_kvar = evcs.get_q_injections_kvar()
        
        run_pf = (t - last_pf_t) >= dt_pf
        
        if run_pf:
            update_network_injections(net, elem_idx, t, profiles_full, P_bess_kw, P_v2g_kw, Q_bess_kvar, Q_v2g_kvar)
            converged, P_e_mw, P_loss_mw = run_power_flow(net)
            pf_count += 1
            
            if converged and P_e_mw is not None:
                P_e_curr = P_e_mw / S_BASE_MVA
                P_loss_curr = P_loss_mw
                # Update voltage measurements for Q-V droop
                evcs.update_voltage_measurements(net, dt_pf)
            
            last_pf_t = t
        
        total_loss_energy_kwh += P_loss_curr * 1000 * DT_DYN / 3600.0
        
        sg.primary_control(freq_dev_hz, DT_DYN)
        sg.secondary_control(freq_dev_hz, DT_DYN)
        sg.swing_equation(P_e_curr, DT_DYN)
        
        if (t - last_dispatch_t) >= 10.0:
            sg.update_dispatch(P_e_curr, t - last_dispatch_t)
            last_dispatch_t = t
        
        log_dt = 0.02 if near_event else 0.1
        if len(log_data) == 0 or (t - log_data[-1]['t_s']) >= log_dt:
            current_freq_dev_hz = sg.get_freq_dev_hz()
            soc_status = evcs.get_soc_status()
            v_status = evcs.get_voltage_status()
            
            log_data.append({
                't_s': t, 't_h': t / 3600.0,
                'freq_Hz': sg.get_frequency_hz(),
                'delta_f_mHz': current_freq_dev_hz * 1000,
                'RoCoF_Hz_s': sg.get_rocof(P_e_curr),
                'P_e_pu': P_e_curr, 'P_m_pu': sg.P_m,
                'P_droop_pu': sg.P_droop, 'P_agc_pu': sg.P_agc,
                'P_bess_kW': sum(P_bess_kw.values()),
                'P_v2g_kW': sum(P_v2g_kw.values()),
                'P_evcs_total_kW': evcs.get_total_response_kw(),
                # Q-V droop values
                'Q_bess_kvar': sum(Q_bess_kvar.values()),
                'Q_v2g_kvar': sum(Q_v2g_kvar.values()),
                'Q_evcs_total_kvar': evcs.get_total_q_response_kvar(),
                # SoC
                'SoC_BESS_avg': np.mean(list(soc_status['bess'].values())),
                'SoC_V2G_avg': np.mean(list(soc_status['v2g'].values())),
                'P_loss_kW': P_loss_curr * 1000,
                'near_event': near_event,
                # Per-cluster P
                'P_bess_EVCS1_kW': P_bess_kw.get(evcs.bus_of.get('EVCS1', 6), 0.0),
                'P_bess_EVCS2_kW': P_bess_kw.get(evcs.bus_of.get('EVCS2', 10), 0.0),
                'P_bess_EVCS3_kW': P_bess_kw.get(evcs.bus_of.get('EVCS3', 15), 0.0),
                'P_v2g_EVCS1_kW': P_v2g_kw.get(evcs.bus_of.get('EVCS1', 6), 0.0),
                'P_v2g_EVCS2_kW': P_v2g_kw.get(evcs.bus_of.get('EVCS2', 10), 0.0),
                'P_v2g_EVCS3_kW': P_v2g_kw.get(evcs.bus_of.get('EVCS3', 15), 0.0),
                # Per-cluster Q
                'Q_bess_EVCS1_kvar': Q_bess_kvar.get(evcs.bus_of.get('EVCS1', 6), 0.0),
                'Q_bess_EVCS2_kvar': Q_bess_kvar.get(evcs.bus_of.get('EVCS2', 10), 0.0),
                'Q_bess_EVCS3_kvar': Q_bess_kvar.get(evcs.bus_of.get('EVCS3', 15), 0.0),
                'Q_v2g_EVCS1_kvar': Q_v2g_kvar.get(evcs.bus_of.get('EVCS1', 6), 0.0),
                'Q_v2g_EVCS2_kvar': Q_v2g_kvar.get(evcs.bus_of.get('EVCS2', 10), 0.0),
                'Q_v2g_EVCS3_kvar': Q_v2g_kvar.get(evcs.bus_of.get('EVCS3', 15), 0.0),
                # Per-cluster voltage
                'V_EVCS1_pu': v_status.get(evcs.bus_of.get('EVCS1', 6), 1.0),
                'V_EVCS2_pu': v_status.get(evcs.bus_of.get('EVCS2', 10), 1.0),
                'V_EVCS3_pu': v_status.get(evcs.bus_of.get('EVCS3', 15), 1.0),
                # SoC per cluster
                'SoC_BESS_EVCS1': soc_status['bess'].get('EVCS1', 0.5),
                'SoC_BESS_EVCS2': soc_status['bess'].get('EVCS2', 0.5),
                'SoC_BESS_EVCS3': soc_status['bess'].get('EVCS3', 0.5),
            })
        
        t += DT_DYN
    
    print(f"\nSimulation complete: {pf_count} power flows")
    print(f"Total loss energy: {total_loss_energy_kwh:.2f} kWh")
    
    results = pd.DataFrame(log_data)
    results['total_loss_energy_kwh'] = total_loss_energy_kwh
    
    evcs_metrics = evcs.get_metrics_summary()
    
    return results, evcs_metrics


# ============================================================================
# 9. COMPREHENSIVE ANALYSIS
# ============================================================================

def analyze_results(results, evcs_metrics=None, enable_evcs=True):
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    t_event = SCEN1_TIME_S
    post_event = results[(results['t_s'] >= t_event) & (results['t_s'] <= t_event + 120)]
    
    metrics = {}
    
    # ================================================================
    # 1. Frequency Metrics
    # ================================================================
    if len(post_event) > 0:
        nadir_idx = post_event['freq_Hz'].idxmin()
        metrics['nadir_Hz'] = results.loc[nadir_idx, 'freq_Hz']
        metrics['nadir_mHz'] = results.loc[nadir_idx, 'delta_f_mHz']
        metrics['time_to_nadir_s'] = results.loc[nadir_idx, 't_s'] - t_event
        metrics['max_delta_f_mHz'] = abs(post_event['delta_f_mHz']).max()
    
    print(f"\n[Frequency Metrics]")
    print(f"  Nadir: {metrics.get('nadir_Hz', np.nan):.4f} Hz ({metrics.get('nadir_mHz', np.nan):.1f} mHz)")
    print(f"  Time to Nadir: {metrics.get('time_to_nadir_s', np.nan):.2f} s")
    print(f"  Max |Δf|: {metrics.get('max_delta_f_mHz', np.nan):.1f} mHz")
    
    # ================================================================
    # 2. RoCoF
    # ================================================================
    metrics['max_rocof_Hz_s'] = abs(post_event['RoCoF_Hz_s']).max() if len(post_event) > 0 else np.nan
    print(f"\n[RoCoF]")
    print(f"  Max |RoCoF|: {metrics.get('max_rocof_Hz_s', np.nan):.3f} Hz/s")
    
    # ================================================================
    # 3. Recovery Time
    # ================================================================
    nadir_time = t_event + metrics.get('time_to_nadir_s', 0)
    post_nadir = results[results['t_s'] >= nadir_time]
    
    for band_mHz, key in [(200, 'recovery_200mHz_s'), (100, 'recovery_100mHz_s')]:
        within_band = post_nadir[abs(post_nadir['delta_f_mHz']) <= band_mHz]
        metrics[key] = (within_band.iloc[0]['t_s'] - t_event) if len(within_band) > 0 else np.nan
    
    for band_mHz, key in [(200, 'settling_200mHz_s'), (100, 'settling_100mHz_s'), (50, 'settling_50mHz_s')]:
        settling_time = np.nan
        for i, row in post_nadir.iterrows():
            t_check = row['t_s']
            future = post_nadir[(post_nadir['t_s'] >= t_check) & (post_nadir['t_s'] <= t_check + 10)]
            if len(future) > 0 and (abs(future['delta_f_mHz']) <= band_mHz).all():
                settling_time = t_check - t_event
                break
        metrics[key] = settling_time
    
    print(f"\n[Recovery Time]")
    print(f"  To ±200 mHz: {metrics.get('recovery_200mHz_s', np.nan):.2f} s")
    print(f"  To ±100 mHz: {metrics.get('recovery_100mHz_s', np.nan):.2f} s")
    print(f"\n[Settling Time]")
    print(f"  ±200 mHz: {metrics.get('settling_200mHz_s', np.nan):.2f} s")
    print(f"  ±100 mHz: {metrics.get('settling_100mHz_s', np.nan):.2f} s")
    print(f"  ±50 mHz: {metrics.get('settling_50mHz_s', np.nan):.2f} s")
    
    # ================================================================
    # 4. Network Losses
    # ================================================================
    metrics['total_loss_energy_kwh'] = results['total_loss_energy_kwh'].iloc[0]
    metrics['avg_loss_kW'] = results['P_loss_kW'].mean()
    metrics['max_loss_kW'] = results['P_loss_kW'].max()
    
    print(f"\n[Network Losses]")
    print(f"  Total Energy: {metrics['total_loss_energy_kwh']:.2f} kWh")
    print(f"  Average: {metrics['avg_loss_kW']:.1f} kW")
    print(f"  Max: {metrics['max_loss_kW']:.1f} kW")
    
    # ================================================================
    # 5. EVCS Metrics
    # ================================================================
    if enable_evcs and evcs_metrics:
        print(f"\n[EVCS P Energy]")
        print(f"  BESS Discharge: {evcs_metrics['total_bess_discharge_kwh']:.2f} kWh")
        print(f"  BESS Charge: {evcs_metrics['total_bess_charge_kwh']:.2f} kWh")
        print(f"  BESS Net: {evcs_metrics['net_bess_energy_kwh']:.2f} kWh")
        print(f"  V2G Discharge: {evcs_metrics['total_v2g_discharge_kwh']:.2f} kWh")
        print(f"  V2G Charge: {evcs_metrics['total_v2g_charge_kwh']:.2f} kWh")
        print(f"  V2G Net: {evcs_metrics['net_v2g_energy_kwh']:.2f} kWh")
        
        print(f"\n[EVCS Q Energy (Volt-Var)]")
        print(f"  BESS Inject: {evcs_metrics['total_q_bess_inject_kvarh']:.2f} kvarh")
        print(f"  BESS Absorb: {evcs_metrics['total_q_bess_absorb_kvarh']:.2f} kvarh")
        print(f"  BESS Net: {evcs_metrics['net_q_bess_kvarh']:.2f} kvarh")
        print(f"  V2G Inject: {evcs_metrics['total_q_v2g_inject_kvarh']:.2f} kvarh")
        print(f"  V2G Absorb: {evcs_metrics['total_q_v2g_absorb_kvarh']:.2f} kvarh")
        print(f"  V2G Net: {evcs_metrics['net_q_v2g_kvarh']:.2f} kvarh")
        
        print(f"\n[EVCS SoC Violations]")
        print(f"  BESS Min (SoC<=20%): {evcs_metrics['total_soc_bess_min_violations']}")
        print(f"  BESS Max (SoC>=90%): {evcs_metrics['total_soc_bess_max_violations']}")
        print(f"  V2G Min (SoC<=30%): {evcs_metrics['total_soc_v2g_min_violations']}")
        print(f"  V2G Max (SoC>=90%): {evcs_metrics['total_soc_v2g_max_violations']}")
        
        print(f"\n[EVCS Saturation Events]")
        print(f"  P-f Droop: BESS={evcs_metrics['total_bess_saturation_count']}, V2G={evcs_metrics['total_v2g_saturation_count']}")
        print(f"  Q-V Droop: BESS={evcs_metrics['total_q_bess_saturation_count']}, V2G={evcs_metrics['total_q_v2g_saturation_count']}")
        
        print(f"\n[Per-Cluster Details]")
        for name, data in evcs_metrics['per_cluster'].items():
            print(f"  {name}:")
            print(f"    P: BESS {data['bess_discharge_kwh']:.2f}/{data['bess_charge_kwh']:.2f} kWh, V2G {data['v2g_discharge_kwh']:.2f}/{data['v2g_charge_kwh']:.2f} kWh")
            print(f"    Q: BESS {data['q_bess_inject_kvarh']:.2f}/{data['q_bess_absorb_kvarh']:.2f} kvarh, V2G {data['q_v2g_inject_kvarh']:.2f}/{data['q_v2g_absorb_kvarh']:.2f} kvarh")
            print(f"    Final SoC: BESS={data['final_soc_bess']*100:.1f}%, V2G={data['final_soc_v2g']*100:.1f}%")
            print(f"    Saturations: P(BESS={data['bess_saturation_count']},V2G={data['v2g_saturation_count']}), Q(BESS={data['q_bess_saturation_count']},V2G={data['q_v2g_saturation_count']})")
        
        metrics['evcs'] = evcs_metrics
    
    # ================================================================
    # 6. Compliance
    # ================================================================
    total_samples = len(results)
    metrics['compliance_200mHz_pct'] = (abs(results['delta_f_mHz']) <= 200).sum() / total_samples * 100
    metrics['compliance_500mHz_pct'] = (abs(results['delta_f_mHz']) <= 500).sum() / total_samples * 100
    
    print(f"\n[Frequency Compliance]")
    print(f"  Within ±200 mHz: {metrics['compliance_200mHz_pct']:.1f}%")
    print(f"  Within ±500 mHz: {metrics['compliance_500mHz_pct']:.1f}%")
    
    # ================================================================
    # 7. Voltage Compliance Metrics (Q-V Droop Evaluation)
    # ================================================================
    if enable_evcs:
        print(f"\n[Voltage Compliance (Q-V Droop)]")
        
        # Voltage limits
        V_MIN, V_MAX = 0.95, 1.05
        V_CRIT_MIN, V_CRIT_MAX = 0.90, 1.10
        
        voltage_metrics = {}
        
        for cluster in ['EVCS1', 'EVCS2', 'EVCS3']:
            v_col = f'V_{cluster}_pu'
            if v_col in results.columns:
                v_data = results[v_col]
                
                # Min/Max
                v_min = v_data.min()
                v_max = v_data.max()
                v_avg = v_data.mean()
                
                # Time outside normal band [0.95, 1.05]
                outside_normal = ((v_data < V_MIN) | (v_data > V_MAX)).sum()
                pct_outside_normal = outside_normal / total_samples * 100
                
                # Time outside critical band [0.90, 1.10]
                outside_critical = ((v_data < V_CRIT_MIN) | (v_data > V_CRIT_MAX)).sum()
                pct_outside_critical = outside_critical / total_samples * 100
                
                # Voltage violation events (edge detection)
                v_below_min = v_data < V_MIN
                v_above_max = v_data > V_MAX
                
                # Count transitions into violation
                undervoltage_events = (v_below_min.astype(int).diff() == 1).sum()
                overvoltage_events = (v_above_max.astype(int).diff() == 1).sum()
                
                voltage_metrics[cluster] = {
                    'v_min_pu': v_min,
                    'v_max_pu': v_max,
                    'v_avg_pu': v_avg,
                    'pct_outside_normal': pct_outside_normal,
                    'pct_outside_critical': pct_outside_critical,
                    'undervoltage_events': int(undervoltage_events),
                    'overvoltage_events': int(overvoltage_events),
                }
                
                print(f"  {cluster} (Bus {evcs_metrics['per_cluster'][cluster].get('bus', '?') if evcs_metrics else '?'}):")
                print(f"    V range: {v_min:.4f} - {v_max:.4f} pu (avg: {v_avg:.4f})")
                print(f"    Outside [0.95,1.05]: {pct_outside_normal:.1f}%")
                if pct_outside_critical > 0:
                    print(f"    Outside [0.90,1.10]: {pct_outside_critical:.1f}% (CRITICAL)")
                print(f"    Violations: {undervoltage_events} under, {overvoltage_events} over")
        
        # Overall voltage metrics
        all_v_cols = [f'V_{c}_pu' for c in ['EVCS1', 'EVCS2', 'EVCS3'] if f'V_{c}_pu' in results.columns]
        if all_v_cols:
            all_v = results[all_v_cols].values.flatten()
            metrics['v_global_min_pu'] = float(np.min(all_v))
            metrics['v_global_max_pu'] = float(np.max(all_v))
            metrics['v_global_avg_pu'] = float(np.mean(all_v))
            metrics['pct_outside_v_normal'] = float(((all_v < V_MIN) | (all_v > V_MAX)).sum() / len(all_v) * 100)
            
            print(f"\n  Global Voltage:")
            print(f"    Range: {metrics['v_global_min_pu']:.4f} - {metrics['v_global_max_pu']:.4f} pu")
            print(f"    Outside [0.95,1.05]: {metrics['pct_outside_v_normal']:.1f}%")
        
        # Q-V effectiveness analysis
        if evcs_metrics:
            total_q_energy = (evcs_metrics['total_q_bess_inject_kvarh'] + 
                             evcs_metrics['total_q_bess_absorb_kvarh'] +
                             evcs_metrics['total_q_v2g_inject_kvarh'] + 
                             evcs_metrics['total_q_v2g_absorb_kvarh'])
            
            if total_q_energy > 0.1:
                # Estimate voltage improvement from Q
                # Rough metric: if Q was used and voltage stayed in band, Q-V is working
                print(f"\n  Q-V Effectiveness:")
                print(f"    Total Q energy: {total_q_energy:.2f} kvarh")
                if metrics.get('pct_outside_v_normal', 0) < 5:
                    print(f"    Status: EFFECTIVE (voltage mostly in band)")
                else:
                    print(f"    Status: NEEDS TUNING (voltage excursions present)")
                
                # Check for oscillations in Q response
                q_cols = [c for c in results.columns if 'Q_bess' in c and '_kvar' in c]
                if q_cols:
                    # Simple oscillation detection: count sign reversals
                    for col in q_cols:
                        q_data = results[col].values
                        dq = np.diff(q_data)
                        sign_changes = np.sum(np.diff(np.sign(dq)) != 0)
                        significant_changes = np.sum(np.abs(dq) > 10)  # > 10 kvar
                        
                        if sign_changes > 20 and significant_changes > 10:
                            cluster = col.replace('Q_bess_', '').replace('_kvar', '')
                            print(f"    WARNING: Possible Q oscillation at {cluster}")
                            print(f"             Consider: evcs.tune_qv_droop(tau_v=0.2, q_ramp_factor=0.5)")
            else:
                print(f"\n  Q-V Effectiveness:")
                print(f"    Minimal Q injection - voltage was in deadband [0.98, 1.02]")
        
        metrics['voltage'] = voltage_metrics
    
    return metrics


def plot_results(results, enable_evcs=True, save_path=None):
    """Create comprehensive plots."""
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    t_event = SCEN1_TIME_S
    t_rel = results['t_s'] - t_event
    
    # Panel 1: Frequency
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_rel, results['delta_f_mHz'], 'b-', linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='±200 mHz')
    ax1.axhline(y=-200, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax1.set_ylabel('Δf [mHz]')
    ax1.set_title('Frequency Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: SG Power
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_rel, results['P_e_pu'] * S_BASE_KW, 'b-', label='P_e')
    ax2.plot(t_rel, results['P_m_pu'] * S_BASE_KW, 'r--', label='P_m')
    ax2.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Power [kW]')
    ax2.set_title('SG Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Control
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_rel, results['P_droop_pu'] * S_BASE_KW, 'b-', label='SG Droop')
    if enable_evcs:
        ax3.plot(t_rel, results['P_evcs_total_kW'], 'r-', linewidth=2, label='EVCS')
    ax3.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax3.set_ylabel('Power [kW]')
    ax3.set_title('Control Response')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: EVCS
    ax4 = fig.add_subplot(gs[2, 0])
    if enable_evcs:
        ax4.plot(t_rel, results['P_bess_kW'], 'b-', label='BESS')
        ax4.plot(t_rel, results['P_v2g_kW'], 'orange', label='V2G')
    ax4.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax4.set_ylabel('Power [kW]')
    ax4.set_title('EVCS Response')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: SoC
    ax5 = fig.add_subplot(gs[2, 1])
    if enable_evcs:
        ax5.plot(t_rel, results['SoC_BESS_avg'] * 100, 'b-', label='BESS')
        ax5.plot(t_rel, results['SoC_V2G_avg'] * 100, 'orange', label='V2G')
        ax5.axhline(y=20, color='r', linestyle='--', alpha=0.5)
        ax5.axhline(y=90, color='r', linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax5.set_ylabel('SoC [%]')
    ax5.set_title('State of Charge')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 100])
    
    # Panel 6: Losses
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(t_rel, results['P_loss_kW'], 'brown', linewidth=1)
    ax6.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax6.set_ylabel('Losses [kW]')
    ax6.set_title('Network Losses')
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: RoCoF
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.plot(t_rel, results['RoCoF_Hz_s'], 'purple', linewidth=1)
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax7.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax7.set_ylabel('RoCoF [Hz/s]')
    ax7.set_title('Rate of Change of Frequency')
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Frequency Zoom
    ax8 = fig.add_subplot(gs[4, 0])
    zoom_mask = (t_rel >= -10) & (t_rel <= 60)
    ax8.plot(t_rel[zoom_mask], results.loc[zoom_mask, 'delta_f_mHz'], 'b-', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax8.axhline(y=200, color='orange', linestyle='--', alpha=0.7)
    ax8.axhline(y=-200, color='orange', linestyle='--', alpha=0.7)
    ax8.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Δf [mHz]')
    ax8.set_title('Frequency Zoom')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([-10, 60])
    
    # Panel 9: Per-cluster BESS
    ax9 = fig.add_subplot(gs[4, 1])
    if enable_evcs:
        for cluster in ['EVCS1', 'EVCS2', 'EVCS3']:
            col = f'P_bess_{cluster}_kW'
            if col in results.columns:
                ax9.plot(t_rel, results[col], label=cluster)
    ax9.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Power [kW]')
    ax9.set_title('BESS Per Cluster')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    mode = "SG + EVCS" if enable_evcs else "SG Only"
    plt.suptitle(f'Frequency Response: {mode}\nLoad Step: +{SCEN1_LOAD_STEP_KW:.0f} kW', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.close()
    return fig


# ============================================================================
# 10. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-evcs', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.compare:
        print("\n" + "=" * 70)
        print("COMPARISON MODE")
        print("=" * 70)
        
        results_base, _ = run_simulation(enable_evcs=False)
        metrics_base = analyze_results(results_base, None, enable_evcs=False)
        
        results_evcs, evcs_met = run_simulation(enable_evcs=True)
        metrics_evcs = analyze_results(results_evcs, evcs_met, enable_evcs=True)
        
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<30} {'SG-only':<15} {'SG+EVCS':<15} {'Improve':<10}")
        print("-" * 70)
        
        nadir_imp = (abs(metrics_base['nadir_mHz']) - abs(metrics_evcs['nadir_mHz'])) / abs(metrics_base['nadir_mHz']) * 100
        rocof_imp = (metrics_base['max_rocof_Hz_s'] - metrics_evcs['max_rocof_Hz_s']) / metrics_base['max_rocof_Hz_s'] * 100
        
        print(f"{'Nadir [mHz]':<30} {metrics_base['nadir_mHz']:<15.1f} {metrics_evcs['nadir_mHz']:<15.1f} {nadir_imp:+.1f}%")
        print(f"{'Max RoCoF [Hz/s]':<30} {metrics_base['max_rocof_Hz_s']:<15.3f} {metrics_evcs['max_rocof_Hz_s']:<15.3f} {rocof_imp:+.1f}%")
        print(f"{'Recovery ±200mHz [s]':<30} {metrics_base.get('recovery_200mHz_s', np.nan):<15.2f} {metrics_evcs.get('recovery_200mHz_s', np.nan):<15.2f}")
        print(f"{'Settling ±100mHz [s]':<30} {metrics_base.get('settling_100mHz_s', np.nan):<15.2f} {metrics_evcs.get('settling_100mHz_s', np.nan):<15.2f}")
        
        results_base.to_csv(f"sim_droop_baseline_{timestamp}.csv", index=False)
        results_evcs.to_csv(f"sim_droop_evcs_{timestamp}.csv", index=False)
        
        with open(f"metrics_comparison_{timestamp}.json", 'w') as f:
            def clean(d):
                if isinstance(d, dict):
                    return {k: clean(v) for k, v in d.items() if not k.startswith('_')}
                elif isinstance(d, float) and np.isnan(d):
                    return None
                return d
            json.dump({'baseline': clean(metrics_base), 'evcs': clean(metrics_evcs)}, f, indent=2)
        
        plot_results(results_base, False, f"sim_droop_baseline_{timestamp}.png")
        plot_results(results_evcs, True, f"sim_droop_evcs_{timestamp}.png")
        
    else:
        enable_evcs = not args.no_evcs
        results, evcs_metrics = run_simulation(enable_evcs=enable_evcs)
        
        if results is not None:
            metrics = analyze_results(results, evcs_metrics, enable_evcs)
            
            mode = "evcs" if enable_evcs else "baseline"
            results.to_csv(f"sim_droop_{mode}_{timestamp}.csv", index=False)
            
            with open(f"sim_droop_{mode}_metrics_{timestamp}.json", 'w') as f:
                def clean(d):
                    if isinstance(d, dict):
                        return {k: clean(v) for k, v in d.items() if not k.startswith('_')}
                    elif isinstance(d, float) and np.isnan(d):
                        return None
                    return d
                json.dump(clean(metrics), f, indent=2)
            
            plot_results(results, enable_evcs, f"sim_droop_{mode}_{timestamp}.png")
    
    print("\nDone!")