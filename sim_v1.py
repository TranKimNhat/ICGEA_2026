"""
sim_v1_metrics.py
=================
Dynamic Frequency Simulation with AC Power Flow (Pandapower)
Enhanced with Comprehensive Metrics Tracking

Metrics Tracked:
- nadir (Hz), max |Δf| (mHz)
- max |RoCoF| (Hz/s)
- recovery time to ±200mHz / ±100mHz
- total losses (MW) per window
- settling time

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

warnings.filterwarnings('ignore')

# Import CIGRE MV microgrid (use same directory as this file)
from pathlib import Path
_this_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_this_dir))
from cigre_mv_microgrid_v1 import (
    create_cigre_mv_microgrid,
    get_network_topology,
    trip_line,
    restore_line,
    get_line_between_buses,
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
DT_PF_EVENT = 0.05                  # Power flow step (near event) [s]
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
# Scenario 1: Load step at 11:00
SCEN1_TIME_H = 11.0                 # Event time [hours]
SCEN1_TIME_S = SCEN1_TIME_H * 3600  # Event time [seconds]
SCEN1_LOAD_STEP_KW = 2000.0         # Load step magnitude [kW]
SCEN1_BUS = 6                       # Bus for load step
SCEN1_DURATION_S = 600.0            # Duration [s]
SCEN1_RAMP_ON_S = 5.0               # Ramp-on time [s]
SCEN1_RAMP_OFF_S = 5.0              # Ramp-off time [s]

# Simulation window
SIM_START_H = 10.0 + 55.0/60.0      # 10:55:00
SIM_END_H = 11.0 + 20.0/60.0        # 11:20:00
SIM_START_S = SIM_START_H * 3600
SIM_END_S = SIM_END_H * 3600


# ============================================================================
# 4. LOAD/GENERATION PROFILES
# ============================================================================

def get_evcs_bus_mapping(net):
    """Get EVCS bus mapping from network config."""
    config = net.get("microgrid_config", {})
    evcs_clusters = config.get("evcs_clusters", {})
    return {
        'EVCS1': int(evcs_clusters.get('EVCS1', {}).get('bus', 6)),
        'EVCS2': int(evcs_clusters.get('EVCS2', {}).get('bus', 10)),
        'EVCS3': int(evcs_clusters.get('EVCS3', {}).get('bus', 15)),
    }


def build_ev_distribution(net):
    """Build EV distribution dict from network config."""
    buses = get_evcs_bus_mapping(net)
    return {
        buses['EVCS1']: 0.35,
        buses['EVCS2']: 0.47,
        buses['EVCS3']: 0.18,
    }


def build_pv_allocation(net):
    """Build PV allocation dict from network config."""
    buses = get_evcs_bus_mapping(net)
    pv_alloc = {
        4: 500,
        buses['EVCS1']: 350,
        9: 350,
        buses['EVCS2']: 350,
        buses['EVCS3']: 150,
    }
    return pv_alloc, sum(pv_alloc.values())


# Load distribution by bus
LOAD_DIST = {
    1: 0.00, 2: 0.05, 3: 0.10, 4: 0.05, 5: 0.10,
    6: 0.08, 7: 0.08, 8: 0.10, 9: 0.05, 10: 0.08,
    11: 0.10, 12: 0.08, 13: 0.05, 14: 0.08,
}

WIND_BUS = 13


def create_profiles_for_window(t_start_s, t_end_s, dt=1.0):
    """Create synthetic profiles for simulation window."""
    times = np.arange(t_start_s, t_end_s + dt, dt)
    n = len(times)
    
    profiles = {
        't_s': times,
        'P_load_kW': np.zeros(n),
        'P_pv_kW': np.zeros(n),
        'P_wind_kW': np.zeros(n),
        'P_ev_kW': np.zeros(n),
    }
    
    for i, t in enumerate(times):
        t_h = t / 3600.0
        profiles['P_load_kW'][i] = 5500 + 200 * np.sin((t_h - 10.5) * np.pi / 2)
        profiles['P_pv_kW'][i] = 1700 * 0.6 * (1 + 0.05 * np.sin((t_h - 11.0) * np.pi * 2))
        profiles['P_wind_kW'][i] = 1000 * 0.5 * (1 + 0.1 * np.sin((t_h - 10.0) * np.pi))
        profiles['P_ev_kW'][i] = 1700 * 0.25 * (1 + 0.1 * (t_h - 10.5))
    
    return profiles


def get_profile_at_time(profiles, t_s):
    """Interpolate profiles to get values at specific time."""
    times = profiles['t_s']
    result = {}
    for key, values in profiles.items():
        if key == 't_s':
            continue
        result[key] = np.interp(t_s, times, values)
    return result


def get_scenario1_load_step(t_s):
    """Calculate Scenario 1 load step with ramp on/off."""
    t_event = SCEN1_TIME_S
    t_end = t_event + SCEN1_DURATION_S
    
    if t_s < t_event:
        return 0.0
    elif t_s < t_event + SCEN1_RAMP_ON_S:
        progress = (t_s - t_event) / SCEN1_RAMP_ON_S
        return SCEN1_LOAD_STEP_KW * progress
    elif t_s < t_end - SCEN1_RAMP_OFF_S:
        return SCEN1_LOAD_STEP_KW
    elif t_s < t_end:
        progress = (t_end - t_s) / SCEN1_RAMP_OFF_S
        return SCEN1_LOAD_STEP_KW * progress
    else:
        return 0.0


# ============================================================================
# 5. PANDAPOWER NETWORK SETUP
# ============================================================================

def setup_network():
    """Create and configure the pandapower network."""
    net = create_cigre_mv_microgrid(mesh_level=2)
    
    # Disable 110/20 kV transformers for islanded operation
    for idx in net.trafo.index:
        hv_bus = net.trafo.at[idx, 'hv_bus']
        if hv_bus == 0:
            net.trafo.at[idx, 'in_service'] = False
    
    net.bus.at[0, 'in_service'] = False
    
    # Build dynamic bus mappings
    ev_dist = build_ev_distribution(net)
    pv_allocation, pv_total = build_pv_allocation(net)
    evcs_buses = get_evcs_bus_mapping(net)
    
    # Clear ALL existing elements
    if len(net.load) > 0:
        net.load.drop(net.load.index, inplace=True)
    if len(net.sgen) > 0:
        net.sgen.drop(net.sgen.index, inplace=True)
    if len(net.storage) > 0:
        net.storage.drop(net.storage.index, inplace=True)
    
    element_indices = {
        'loads': {},
        'ev_loads': {},
        'pv_sgens': {},
        'wind_sgen': None,
        'step_load': None,
        'ev_dist': ev_dist,
        'pv_allocation': pv_allocation,
        'pv_total': pv_total,
        'evcs_buses': evcs_buses,
    }
    
    n_bus = len(net.bus)
    
    # Create base loads
    for bus, frac in LOAD_DIST.items():
        if bus < n_bus and frac > 0:
            idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"Load_Bus{bus}")
            element_indices['loads'][bus] = idx
    
    # Create EV loads
    for bus, frac in ev_dist.items():
        if bus < n_bus:
            idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"EV_Bus{bus}")
            element_indices['ev_loads'][bus] = idx
    
    # Create PV sgens
    for bus, p_kw in pv_allocation.items():
        if bus < n_bus:
            idx = pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"PV_Bus{bus}", type="PV")
            element_indices['pv_sgens'][bus] = idx
    
    # Create Wind sgen
    if WIND_BUS < n_bus:
        idx = pp.create_sgen(net, bus=WIND_BUS, p_mw=0.0, q_mvar=0.0, name="Wind_13", type="WP")
        element_indices['wind_sgen'] = idx
    
    # Create Scenario 1 step load
    if SCEN1_BUS < n_bus:
        idx = pp.create_load(net, bus=SCEN1_BUS, p_mw=0.0, q_mvar=0.0, name="LoadStep_Scen1")
        element_indices['step_load'] = idx
    
    if len(net.gen) > 0:
        net.gen.at[0, 'slack'] = True
    
    return net, element_indices


def update_network_injections(net, elem_idx, t_s, profiles_full):
    """Update all load/sgen values in pandapower network."""
    profiles = get_profile_at_time(profiles_full, t_s)
    
    P_load = profiles['P_load_kW']
    P_ev = profiles['P_ev_kW']
    P_pv = profiles['P_pv_kW']
    P_wind = profiles['P_wind_kW']
    
    ev_dist = elem_idx.get('ev_dist', {})
    pv_allocation = elem_idx.get('pv_allocation', {})
    pv_total = elem_idx.get('pv_total', 1700)
    
    pf_load = 0.9
    tan_phi = np.tan(np.arccos(pf_load))
    
    # Update base loads
    for bus, idx in elem_idx['loads'].items():
        frac = LOAD_DIST.get(bus, 0)
        p_mw = P_load * frac / 1000.0
        q_mvar = p_mw * tan_phi
        net.load.at[idx, 'p_mw'] = p_mw
        net.load.at[idx, 'q_mvar'] = q_mvar
    
    # Update EV loads
    for bus, idx in elem_idx['ev_loads'].items():
        frac = ev_dist.get(bus, 0)
        p_mw = P_ev * frac / 1000.0
        net.load.at[idx, 'p_mw'] = p_mw
        net.load.at[idx, 'q_mvar'] = 0.0
    
    # Update PV sgens
    for bus, idx in elem_idx['pv_sgens'].items():
        p_alloc = pv_allocation.get(bus, 0)
        p_mw = P_pv * (p_alloc / pv_total) / 1000.0 if pv_total > 0 else 0.0
        net.sgen.at[idx, 'p_mw'] = p_mw
        net.sgen.at[idx, 'q_mvar'] = 0.0
    
    # Update Wind sgen
    if elem_idx['wind_sgen'] is not None:
        net.sgen.at[elem_idx['wind_sgen'], 'p_mw'] = P_wind / 1000.0
        net.sgen.at[elem_idx['wind_sgen'], 'q_mvar'] = 0.0
    
    # Update Scenario 1 step load
    P_step = get_scenario1_load_step(t_s)
    if elem_idx['step_load'] is not None:
        net.load.at[elem_idx['step_load'], 'p_mw'] = P_step / 1000.0
        net.load.at[elem_idx['step_load'], 'q_mvar'] = P_step / 1000.0 * tan_phi
    
    return {
        'P_load_kW': P_load,
        'P_ev_kW': P_ev,
        'P_pv_kW': P_pv,
        'P_wind_kW': P_wind,
        'P_step_kW': P_step,
    }


def run_power_flow(net, init='results'):
    """Run AC power flow with fallback options."""
    # Primary attempt
    try:
        pp.runpp(net, algorithm='nr', init=init, max_iteration=30, tolerance_mva=1e-6)
        if net.converged:
            if len(net.res_gen) > 0:
                P_e_mw = net.res_gen.p_mw.sum()
            elif len(net.res_ext_grid) > 0:
                P_e_mw = net.res_ext_grid.p_mw.sum()
            else:
                P_e_mw = None
            # Get losses
            P_loss_mw = net.res_line.pl_mw.sum() if len(net.res_line) > 0 else 0.0
            P_loss_mw += net.res_trafo.pl_mw.sum() if len(net.res_trafo) > 0 else 0.0
            return True, P_e_mw, P_loss_mw
    except Exception:
        pass
    
    # Fallback: flat start
    try:
        pp.runpp(net, algorithm='nr', init='flat', max_iteration=100, tolerance_mva=1e-5)
        if net.converged:
            if len(net.res_gen) > 0:
                P_e_mw = net.res_gen.p_mw.sum()
            elif len(net.res_ext_grid) > 0:
                P_e_mw = net.res_ext_grid.p_mw.sum()
            else:
                P_e_mw = None
            P_loss_mw = net.res_line.pl_mw.sum() if len(net.res_line) > 0 else 0.0
            P_loss_mw += net.res_trafo.pl_mw.sum() if len(net.res_trafo) > 0 else 0.0
            return True, P_e_mw, P_loss_mw
    except Exception:
        pass
    
    return False, None, 0.0


# ============================================================================
# 6. SYNCHRONOUS GENERATOR MODEL
# ============================================================================

class SynchronousGenerator:
    """Synchronous Generator with swing equation, governor, and AGC."""
    
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
        if abs(freq_dev_hz) <= DEADBAND_PRI_HZ:
            df_eff = 0.0
        else:
            df_eff = freq_dev_hz - DEADBAND_PRI_HZ * np.sign(freq_dev_hz)
        
        R_pu = R_PERCENT / 100.0
        P_droop_cmd = -df_eff / (R_pu * F_NOM)
        
        alpha_g = dt / (TG + dt)
        P_target = self.P_nom + P_droop_cmd + self.P_agc
        P_target = np.clip(P_target, SG_P_MIN_PU, SG_P_MAX_PU)
        self.P_ref += alpha_g * (P_target - self.P_ref)
        
        alpha_t = dt / (TT + dt)
        self.P_m += alpha_t * (self.P_ref - self.P_m)
        
        self.P_droop = P_droop_cmd
    
    def secondary_control(self, freq_dev_hz, dt):
        alpha_f = dt / (TF_AGC + dt)
        self.f_filtered += alpha_f * (freq_dev_hz - self.f_filtered)
        
        if abs(self.f_filtered) <= DEADBAND_AGC_HZ:
            f_eff = 0.0
        else:
            f_eff = self.f_filtered - DEADBAND_AGC_HZ * np.sign(self.f_filtered)
        
        P_prop = -KP_AGC * f_eff
        self.agc_integral += -KI_AGC * f_eff * dt
        self.agc_integral = np.clip(self.agc_integral, -P_AGC_MAX_PU, P_AGC_MAX_PU)
        
        P_agc_cmd = P_prop + self.agc_integral
        
        dP_max = RAMP_AGC_PU_S * dt
        dP = np.clip(P_agc_cmd - self.P_agc, -dP_max, dP_max)
        self.P_agc += dP
        self.P_agc = np.clip(self.P_agc, -P_AGC_MAX_PU, P_AGC_MAX_PU)
    
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
# 7. MAIN SIMULATION
# ============================================================================

def run_simulation():
    """Run hybrid dynamic simulation (SG only, baseline)."""
    print("=" * 70)
    print("SIMULATION: SG Only (Baseline) with Comprehensive Metrics")
    print("=" * 70)
    print(f"Simulation: {SIM_START_H:.4f}h to {SIM_END_H:.4f}h")
    print(f"Scenario 1: Load step +{SCEN1_LOAD_STEP_KW:.0f} kW at {SCEN1_TIME_H:.2f}h")
    
    # Setup network
    print("\nSetting up network...")
    net, elem_idx = setup_network()
    
    # Initialize SG
    sg = SynchronousGenerator()
    
    # Precompute profiles
    profiles_full = create_profiles_for_window(SIM_START_S, SIM_END_S)
    
    # Initial power flow
    update_network_injections(net, elem_idx, SIM_START_S, profiles_full)
    converged, P_e_mw, P_loss_mw = run_power_flow(net, init='flat')
    
    if not converged:
        print("ERROR: Initial power flow failed!")
        return None
    
    P_e_pu = P_e_mw / S_BASE_MVA
    sg.P_nom = P_e_pu
    sg.P_ref = P_e_pu
    sg.P_m = P_e_pu
    
    print(f"Initial P_e = {P_e_mw:.3f} MW, Losses = {P_loss_mw*1000:.1f} kW")
    
    # Logging
    log_data = []
    
    # Simulation loop
    t = SIM_START_S
    last_pf_t = t
    last_dispatch_t = t
    P_e_curr = P_e_pu
    P_loss_curr = P_loss_mw
    pf_count = 0
    
    # Cumulative metrics
    total_loss_energy_kwh = 0.0
    
    print("\nRunning simulation...")
    
    while t <= SIM_END_S:
        # Determine PF timestep
        near_event = (abs(t - SCEN1_TIME_S) < EVENT_WINDOW or
                      abs(t - (SCEN1_TIME_S + SCEN1_DURATION_S)) < EVENT_WINDOW)
        dt_pf = DT_PF_EVENT if near_event else DT_PF_NORMAL
        
        # Get frequency deviation
        freq_dev_hz = sg.get_freq_dev_hz()
        
        # Run power flow if needed
        run_pf = (t - last_pf_t) >= dt_pf
        
        if run_pf:
            profiles = update_network_injections(net, elem_idx, t, profiles_full)
            converged, P_e_mw, P_loss_mw = run_power_flow(net)
            pf_count += 1
            
            if converged and P_e_mw is not None:
                P_e_curr = P_e_mw / S_BASE_MVA
                P_loss_curr = P_loss_mw
            
            last_pf_t = t
        
        # Accumulate loss energy
        total_loss_energy_kwh += P_loss_curr * 1000 * DT_DYN / 3600.0
        
        # SG control and swing equation
        sg.primary_control(freq_dev_hz, DT_DYN)
        sg.secondary_control(freq_dev_hz, DT_DYN)
        sg.swing_equation(P_e_curr, DT_DYN)
        
        # Dispatch update (slow)
        if (t - last_dispatch_t) >= 10.0:
            sg.update_dispatch(P_e_curr, t - last_dispatch_t)
            last_dispatch_t = t
        
        # Logging
        log_dt = 0.02 if near_event else 0.1
        if len(log_data) == 0 or (t - log_data[-1]['t_s']) >= log_dt:
            current_freq_dev_hz = sg.get_freq_dev_hz()
            
            log_data.append({
                't_s': t,
                't_h': t / 3600.0,
                'freq_Hz': sg.get_frequency_hz(),
                'delta_f_mHz': current_freq_dev_hz * 1000,
                'RoCoF_Hz_s': sg.get_rocof(P_e_curr),
                'P_e_pu': P_e_curr,
                'P_e_kW': P_e_curr * S_BASE_KW,
                'P_m_pu': sg.P_m,
                'P_m_kW': sg.P_m * S_BASE_KW,
                'P_droop_pu': sg.P_droop,
                'P_agc_pu': sg.P_agc,
                'P_loss_kW': P_loss_curr * 1000,
                'near_event': near_event,
            })
        
        t += DT_DYN
    
    print(f"\nSimulation complete: {pf_count} power flows")
    print(f"Total loss energy: {total_loss_energy_kwh:.2f} kWh")
    
    # Create DataFrame
    results = pd.DataFrame(log_data)
    results['total_loss_energy_kwh'] = total_loss_energy_kwh
    
    return results


# ============================================================================
# 8. COMPREHENSIVE ANALYSIS
# ============================================================================

def analyze_results(results):
    """Comprehensive analysis with all requested metrics."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    t_event = SCEN1_TIME_S
    
    # Post-event data (0-120s after event)
    post_event = results[(results['t_s'] >= t_event) & (results['t_s'] <= t_event + 120)]
    
    metrics = {}
    
    # ================================================================
    # 1. Frequency Nadir and Max |Δf|
    # ================================================================
    if len(post_event) > 0:
        nadir_idx = post_event['freq_Hz'].idxmin()
        metrics['nadir_Hz'] = results.loc[nadir_idx, 'freq_Hz']
        metrics['nadir_mHz'] = results.loc[nadir_idx, 'delta_f_mHz']
        metrics['time_to_nadir_s'] = results.loc[nadir_idx, 't_s'] - t_event
        metrics['max_delta_f_mHz'] = abs(post_event['delta_f_mHz']).max()
    else:
        metrics['nadir_Hz'] = np.nan
        metrics['nadir_mHz'] = np.nan
        metrics['time_to_nadir_s'] = np.nan
        metrics['max_delta_f_mHz'] = np.nan
    
    print(f"\n[Frequency Metrics]")
    print(f"  Nadir: {metrics['nadir_Hz']:.4f} Hz ({metrics['nadir_mHz']:.1f} mHz)")
    print(f"  Time to Nadir: {metrics['time_to_nadir_s']:.2f} s")
    print(f"  Max |Δf|: {metrics['max_delta_f_mHz']:.1f} mHz")
    
    # ================================================================
    # 2. Max |RoCoF|
    # ================================================================
    if len(post_event) > 0:
        metrics['max_rocof_Hz_s'] = abs(post_event['RoCoF_Hz_s']).max()
    else:
        metrics['max_rocof_Hz_s'] = np.nan
    
    print(f"\n[RoCoF]")
    print(f"  Max |RoCoF|: {metrics['max_rocof_Hz_s']:.3f} Hz/s")
    
    # ================================================================
    # 3. Recovery Time to ±200mHz and ±100mHz
    # ================================================================
    post_nadir = results[results['t_s'] >= t_event + metrics.get('time_to_nadir_s', 0)]
    
    # Recovery to ±200mHz
    within_200mHz = post_nadir[abs(post_nadir['delta_f_mHz']) <= 200]
    if len(within_200mHz) > 0:
        first_recovery_200 = within_200mHz.iloc[0]['t_s']
        metrics['recovery_200mHz_s'] = first_recovery_200 - t_event
    else:
        metrics['recovery_200mHz_s'] = np.nan
    
    # Recovery to ±100mHz
    within_100mHz = post_nadir[abs(post_nadir['delta_f_mHz']) <= 100]
    if len(within_100mHz) > 0:
        first_recovery_100 = within_100mHz.iloc[0]['t_s']
        metrics['recovery_100mHz_s'] = first_recovery_100 - t_event
    else:
        metrics['recovery_100mHz_s'] = np.nan
    
    # Settling time (stay within band for 10s)
    for band_mHz, key in [(200, 'settling_200mHz_s'), (100, 'settling_100mHz_s'), (50, 'settling_50mHz_s')]:
        settling_time = np.nan
        for i, row in post_nadir.iterrows():
            t_check = row['t_s']
            future_10s = post_nadir[(post_nadir['t_s'] >= t_check) & (post_nadir['t_s'] <= t_check + 10)]
            if len(future_10s) > 0 and (abs(future_10s['delta_f_mHz']) <= band_mHz).all():
                settling_time = t_check - t_event
                break
        metrics[key] = settling_time
    
    print(f"\n[Recovery Time]")
    print(f"  To ±200 mHz: {metrics['recovery_200mHz_s']:.2f} s")
    print(f"  To ±100 mHz: {metrics['recovery_100mHz_s']:.2f} s")
    print(f"\n[Settling Time (stay for 10s)]")
    print(f"  ±200 mHz: {metrics['settling_200mHz_s']:.2f} s")
    print(f"  ±100 mHz: {metrics['settling_100mHz_s']:.2f} s")
    print(f"  ±50 mHz: {metrics['settling_50mHz_s']:.2f} s")
    
    # ================================================================
    # 4. Total Losses
    # ================================================================
    if 'total_loss_energy_kwh' in results.columns:
        metrics['total_loss_energy_kwh'] = results['total_loss_energy_kwh'].iloc[0]
    else:
        # Calculate from P_loss_kW
        dt_log = np.diff(results['t_s'].values)
        P_loss = results['P_loss_kW'].values[:-1]
        metrics['total_loss_energy_kwh'] = (P_loss * dt_log).sum() / 3600.0
    
    metrics['avg_loss_kW'] = results['P_loss_kW'].mean()
    metrics['max_loss_kW'] = results['P_loss_kW'].max()
    
    print(f"\n[Network Losses]")
    print(f"  Total Energy Loss: {metrics['total_loss_energy_kwh']:.2f} kWh")
    print(f"  Average Loss: {metrics['avg_loss_kW']:.1f} kW")
    print(f"  Max Loss: {metrics['max_loss_kW']:.1f} kW")
    
    # ================================================================
    # 5. Frequency Compliance
    # ================================================================
    total_samples = len(results)
    metrics['compliance_200mHz_pct'] = (abs(results['delta_f_mHz']) <= 200).sum() / total_samples * 100
    metrics['compliance_500mHz_pct'] = (abs(results['delta_f_mHz']) <= 500).sum() / total_samples * 100
    
    print(f"\n[Frequency Compliance]")
    print(f"  Within ±200 mHz: {metrics['compliance_200mHz_pct']:.1f}%")
    print(f"  Within ±500 mHz: {metrics['compliance_500mHz_pct']:.1f}%")
    
    return metrics


def plot_results(results, save_path=None):
    """Create comprehensive result plots."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    t_event = SCEN1_TIME_S
    t_rel = (results['t_s'] - t_event)
    
    # Panel 1: Frequency (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_rel, results['delta_f_mHz'], 'b-', linewidth=1.5, label='Δf')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='±200 mHz')
    ax1.axhline(y=-200, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(y=500, color='r', linestyle='--', alpha=0.7, label='±500 mHz')
    ax1.axhline(y=-500, color='r', linestyle='--', alpha=0.7)
    ax1.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5, label='Event')
    ax1.set_xlabel('Time relative to event [s]')
    ax1.set_ylabel('Δf [mHz]')
    ax1.set_title('Frequency Deviation (SG Only Baseline)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: SG Power
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_rel, results['P_e_kW'], 'b-', label='P_e (electrical)')
    ax2.plot(t_rel, results['P_m_kW'], 'r--', label='P_m (mechanical)')
    ax2.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Time relative to event [s]')
    ax2.set_ylabel('Power [kW]')
    ax2.set_title('SG Power')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: RoCoF
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_rel, results['RoCoF_Hz_s'], 'purple', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Time relative to event [s]')
    ax3.set_ylabel('RoCoF [Hz/s]')
    ax3.set_title('Rate of Change of Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Losses
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t_rel, results['P_loss_kW'], 'brown', linewidth=1)
    ax4.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Time relative to event [s]')
    ax4.set_ylabel('Losses [kW]')
    ax4.set_title('Network Losses')
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Frequency Zoom
    ax5 = fig.add_subplot(gs[2, 1])
    zoom_mask = (t_rel >= -10) & (t_rel <= 60)
    ax5.plot(t_rel[zoom_mask], results.loc[zoom_mask, 'delta_f_mHz'], 'b-', linewidth=1.5)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.axhline(y=200, color='orange', linestyle='--', alpha=0.7)
    ax5.axhline(y=-200, color='orange', linestyle='--', alpha=0.7)
    ax5.axvline(x=0, color='g', linestyle='-', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Time relative to event [s]')
    ax5.set_ylabel('Δf [mHz]')
    ax5.set_title('Frequency Zoom (Event ±60s)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([-10, 60])
    
    plt.suptitle(f'Frequency Response: SG Only (Baseline)\n'
                 f'Load Step: +{SCEN1_LOAD_STEP_KW:.0f} kW at t=0',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.close()
    return fig


# ============================================================================
# 9. MAIN
# ============================================================================

if __name__ == "__main__":
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run simulation
    results = run_simulation()
    
    if results is not None:
        # Analyze
        metrics = analyze_results(results)
        
        # Save results
        csv_path = f"sim_v1_baseline_{timestamp}.csv"
        png_path = f"sim_v1_baseline_{timestamp}.png"
        metrics_path = f"sim_v1_metrics_{timestamp}.json"
        
        results.to_csv(csv_path, index=False)
        print(f"\nResults saved: {csv_path}")
        
        # Save metrics as JSON
        import json
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}, f, indent=2)
        print(f"Metrics saved: {metrics_path}")
        
        # Plot
        plot_results(results, save_path=png_path)
    
    print("\nDone!")