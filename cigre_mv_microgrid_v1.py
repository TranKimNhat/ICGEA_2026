"""
cigre_mv_microgrid.py
=====================
CIGRE MV Benchmark Network modified for Islanded Microgrid with EVCS

Configuration:
- Base: CIGRE MV (110/20 kV) → Modified to islanded 20kV microgrid
- Mesh level 2: S1, S2, S3 closed (inter-feeder ties)
- SG at Bus 1 (grid-forming master)
- 3 EVCS clusters (grid-forming VSG):
  - EVCS1 @ Bus 6 (MV 20kV): 600kW + PV 350kW
  - EVCS2 @ Bus 10 (MV 20kV): 800kW + PV 350kW  
  - EVCS3 @ Bus 14→LV14 (0.4kV via 630kVA trafo): 300kW + PV 150kW
- Standalone RES (grid-following):
  - PV @ Bus 4: 500kW
  - PV @ Bus 9: 350kW
  - Wind @ Bus 13: 1000kW

Reference: CIGRE Task Force C6.04.02
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np


def create_cigre_mv_microgrid(mesh_level=2):
    """
    Create CIGRE MV network modified for islanded microgrid operation.
    
    Parameters
    ----------
    mesh_level : int
        0 = radial (all switches open)
        1 = single loop (S1 closed only)
        2 = full mesh (S1, S2, S3 all closed) - RECOMMENDED
    
    Returns
    -------
    net : pandapower network
    """
    
    # Start with standard CIGRE MV
    net = pn.create_cigre_network_mv()
    
    # ================================================================
    # 1. REMOVE EXTERNAL GRID (Islanded operation)
    # ================================================================
    # Keep the 110kV bus but remove ext_grid
    # We'll add SG at Bus 1 instead
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    
    # ================================================================
    # 2. CONFIGURE MESH TOPOLOGY
    # ================================================================
    # CIGRE MV switches (verified from net.switch table):
    # S1: controls Line 14, connects Bus 14 ↔ Bus 8 (inter-feeder tie)
    # S2: controls Line 12, connects Bus 6 ↔ Bus 7 (feeder 1 loop)
    # S3: controls Line 13, connects Bus 11 ↔ Bus 4 (inter-feeder tie)
    
    if mesh_level >= 1:
        # S1: inter-feeder tie (Bus 14 - Bus 8)
        net.switch.loc[net.switch['name'] == 'S1', 'closed'] = True
    
    if mesh_level >= 2:
        # S2: feeder 1 loop (Bus 6 - Bus 7)
        net.switch.loc[net.switch['name'] == 'S2', 'closed'] = True
        # S3: inter-feeder tie (Bus 11 - Bus 4)
        net.switch.loc[net.switch['name'] == 'S3', 'closed'] = True
    
    # ================================================================
    # 3. ADD SG AT BUS 1 (Grid-forming master)
    # ================================================================
    # Bus 1 is 20kV MV bus (connected to 110kV via trafo)
    # For islanded operation, SG replaces external grid
    
    pp.create_gen(
        net,
        bus=1,
        p_mw=5.0,           # Initial dispatch (will be updated)
        vm_pu=1.0,
        name="SG_Bus1",
        type="SG",
        slack=True,         # Slack bus for power flow
        max_p_mw=12.0,      # Rated capacity
        min_p_mw=0.0,
        max_q_mvar=6.0,
        min_q_mvar=-6.0,
        controllable=True,
    )
    
    # ================================================================
    # 4. ADD LV BUS AND TRANSFORMER FOR EVCS3
    # ================================================================
    # Create LV bus at 0.4kV
    lv_bus = pp.create_bus(
        net,
        vn_kv=0.4,
        name="LV14",
        type="n",           # Load bus
        zone="EVCS3",
    )
    
    # 20/0.4 kV transformer (630 kVA)
    pp.create_transformer(
        net,
        hv_bus=14,          # MV side (Bus 14)
        lv_bus=lv_bus,      # LV side (new LV14 bus)
        std_type="0.63 MVA 20/0.4 kV",
        name="Trafo_14_LV14",
    )
    
    # ================================================================
    # 5. ADD EVCS CLUSTERS (as static generators for power flow)
    # ================================================================
    # Note: For dynamic simulation, these will be replaced by VSG models
    
    # EVCS1 @ Bus 6 (MV): 600kW charging load + 350kW PV + BESS
    pp.create_load(net, bus=6, p_mw=0.6, q_mvar=0.0, name="EVCS1_Load")
    pp.create_sgen(net, bus=6, p_mw=0.35, q_mvar=0.0, name="PV_6", type="PV")
    pp.create_storage(net, bus=6, p_mw=0.0, max_e_mwh=1.2, name="BESS1")
    
    # EVCS2 @ Bus 10 (MV): 800kW charging load + 350kW PV + BESS
    pp.create_load(net, bus=10, p_mw=0.8, q_mvar=0.0, name="EVCS2_Load")
    pp.create_sgen(net, bus=10, p_mw=0.35, q_mvar=0.0, name="PV_10", type="PV")
    pp.create_storage(net, bus=10, p_mw=0.0, max_e_mwh=1.6, name="BESS2")
    
    # EVCS3 @ LV14 (LV): 300kW charging load + 150kW PV + BESS
    pp.create_load(net, bus=lv_bus, p_mw=0.3, q_mvar=0.0, name="EVCS3_Load")
    pp.create_sgen(net, bus=lv_bus, p_mw=0.15, q_mvar=0.0, name="PV_14", type="PV")
    pp.create_storage(net, bus=lv_bus, p_mw=0.0, max_e_mwh=0.6, name="BESS3")
    
    # ================================================================
    # 6. ADD STANDALONE RES (Grid-following PQ)
    # ================================================================
    # PV @ Bus 4: 500kW
    pp.create_sgen(net, bus=4, p_mw=0.5, q_mvar=0.0, name="PV_4", type="PV")
    
    # PV @ Bus 9: 350kW
    pp.create_sgen(net, bus=9, p_mw=0.35, q_mvar=0.0, name="PV_9", type="PV")
    
    # Wind @ Bus 13: 1000kW
    pp.create_sgen(net, bus=13, p_mw=1.0, q_mvar=0.0, name="Wind_13", type="WP")
    
    # ================================================================
    # 7. ADD NETWORK METADATA
    # ================================================================
    net["microgrid_config"] = {
        "name": "CIGRE_MV_Microgrid",
        "base_mva": 10.0,           # Base MVA for per-unit
        "f_hz": 50.0,               # Nominal frequency
        "mesh_level": mesh_level,
        "voltage_levels": {
            "HV": 110.0,            # kV (not used in islanded)
            "MV": 20.0,             # kV
            "LV": 0.4,              # kV
        },
        "sg": {
            "bus": 1,
            "p_rated_mw": 12.0,
            "type": "grid-forming",
        },
        "evcs_clusters": {
            "EVCS1": {"bus": 6, "p_ev_kw": 600, "p_pv_kw": 350, "bess_kwh": 1200},
            "EVCS2": {"bus": 10, "p_ev_kw": 800, "p_pv_kw": 350, "bess_kwh": 1600},
            "EVCS3": {"bus": lv_bus, "p_ev_kw": 300, "p_pv_kw": 150, "bess_kwh": 600, "trafo_kva": 630},
        },
        "standalone_res": {
            "PV_4": {"bus": 4, "p_kw": 500},
            "PV_9": {"bus": 9, "p_kw": 350},
            "Wind_13": {"bus": 13, "p_kw": 1000},
        },
    }
    
    return net


def get_bus_mapping(net):
    """
    Get mapping of bus names to indices for easy reference.
    """
    return {row['name']: idx for idx, row in net.bus.iterrows()}


def get_network_topology(net, s_base_mva=10.0):
    """
    Extract network topology for GNN (adjacency matrix).
    
    This function properly respects switch states:
    - A line is considered connected only if in_service=True AND
      all switches on that line are closed (or no switches exist)
    
    Parameters
    ----------
    net : pandapower network
    s_base_mva : float
        System base MVA for per-unit calculations (default: 10.0)
    
    Returns
    -------
    adj_matrix : np.ndarray
        Adjacency matrix (N x N)
    admittance_matrix : np.ndarray
        B' matrix (susceptance) in per-unit on s_base_mva
    """
    n_bus = len(net.bus)
    adj = np.zeros((n_bus, n_bus))
    B = np.zeros((n_bus, n_bus))
    
    # Build set of lines that are blocked by open switches
    # Switch with et='l' controls a line; if closed=False, line is open
    blocked_lines = set()
    for idx in net.switch.index:
        if net.switch.at[idx, 'et'] == 'l':  # Line switch
            if not net.switch.at[idx, 'closed']:
                blocked_lines.add(net.switch.at[idx, 'element'])
    
    # Lines
    for idx in net.line.index:
        fb = int(net.line.at[idx, 'from_bus'])
        tb = int(net.line.at[idx, 'to_bus'])
        
        # Check if line is in service AND not blocked by switch
        in_service = net.line.at[idx, 'in_service']
        switch_open = (idx in blocked_lines)
        
        if in_service and not switch_open:
            adj[fb, tb] = 1
            adj[tb, fb] = 1
            
            # Calculate susceptance
            x_ohm = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km']
            v_kv = net.bus.at[fb, 'vn_kv']
            z_base = (v_kv ** 2) / s_base_mva
            x_pu = x_ohm / z_base if z_base > 0 else 0.01
            b = 1.0 / max(x_pu, 0.01)
            
            B[fb, fb] += b
            B[tb, tb] += b
            B[fb, tb] -= b
            B[tb, fb] -= b
    
    # Transformers (no switch handling needed for now)
    for idx in net.trafo.index:
        hv = int(net.trafo.at[idx, 'hv_bus'])
        lv = int(net.trafo.at[idx, 'lv_bus'])
        
        if net.trafo.at[idx, 'in_service']:
            adj[hv, lv] = 1
            adj[lv, hv] = 1
            
            vk = net.trafo.at[idx, 'vk_percent'] / 100.0
            sn = net.trafo.at[idx, 'sn_mva']
            x_pu = vk * s_base_mva / sn if sn > 0 else 0.06
            b = 1.0 / max(x_pu, 0.01)
            
            B[hv, hv] += b
            B[lv, lv] += b
            B[hv, lv] -= b
            B[lv, hv] -= b
    
    return adj, B


def print_network_summary(net):
    """Print summary of the network configuration."""
    print("=" * 70)
    print("CIGRE MV MICROGRID SUMMARY")
    print("=" * 70)
    
    print(f"\n[Topology]")
    print(f"  Buses: {len(net.bus)}")
    print(f"  Lines: {len(net.line)}")
    print(f"  Transformers: {len(net.trafo)}")
    print(f"  Switches: {len(net.switch)}")
    
    print(f"\n[Voltage Levels]")
    for vn in sorted(net.bus.vn_kv.unique(), reverse=True):
        count = len(net.bus[net.bus.vn_kv == vn])
        print(f"  {vn} kV: {count} buses")
    
    print(f"\n[Switches Status]")
    for idx, row in net.switch.iterrows():
        status = "CLOSED" if row['closed'] else "OPEN"
        name = row['name'] if row['name'] else f"Switch_{idx}"
        print(f"  {name}: {status}")
    
    print(f"\n[Generators]")
    for idx, row in net.gen.iterrows():
        print(f"  {row['name']}: Bus {row['bus']}, P={row['p_mw']} MW, "
              f"Max={row['max_p_mw']} MW")
    
    print(f"\n[Static Generators (PV/Wind)]")
    for idx, row in net.sgen.iterrows():
        print(f"  {row['name']}: Bus {row['bus']}, P={row['p_mw']} MW, "
              f"Type={row['type']}")
    
    print(f"\n[Storage (BESS)]")
    for idx, row in net.storage.iterrows():
        print(f"  {row['name']}: Bus {row['bus']}, "
              f"Max_E={row['max_e_mwh']} MWh")
    
    print(f"\n[Loads]")
    total_load = 0
    for idx, row in net.load.iterrows():
        print(f"  {row['name']}: Bus {row['bus']}, P={row['p_mw']*1000:.0f} kW")
        total_load += row['p_mw']
    print(f"  TOTAL: {total_load*1000:.0f} kW")
    
    # Run power flow to check
    print(f"\n[Power Flow Check]")
    try:
        pp.runpp(net)
        print(f"  ✅ Power flow converged")
        print(f"  SG output: {net.res_gen.p_mw.sum()*1000:.0f} kW")
        print(f"  Total load: {net.res_load.p_mw.sum()*1000:.0f} kW")
        print(f"  Losses: {net.res_line.pl_mw.sum()*1000:.1f} kW")
    except Exception as e:
        print(f"  ❌ Power flow failed: {e}")


# ================================================================
# TOPOLOGY MANIPULATION HELPERS
# ================================================================

def get_line_between_buses(net, bus1, bus2):
    """
    Find line index connecting two buses.
    
    Returns
    -------
    line_idx : int or None
        Line index if found, None otherwise
    """
    for idx in net.line.index:
        fb = int(net.line.at[idx, 'from_bus'])
        tb = int(net.line.at[idx, 'to_bus'])
        if (fb == bus1 and tb == bus2) or (fb == bus2 and tb == bus1):
            return idx
    return None


def trip_line(net, line_idx=None, bus1=None, bus2=None):
    """
    Trip a line (set in_service=False).
    
    Can specify either line_idx directly, or bus1+bus2 to find line.
    This affects both pandapower power flow AND get_network_topology().
    
    Parameters
    ----------
    net : pandapower network
    line_idx : int, optional
        Direct line index
    bus1, bus2 : int, optional
        Bus pair to find line between
    
    Returns
    -------
    line_idx : int
        Index of tripped line
    success : bool
        Whether trip was successful
    """
    if line_idx is None and bus1 is not None and bus2 is not None:
        line_idx = get_line_between_buses(net, bus1, bus2)
    
    if line_idx is None:
        return None, False
    
    if line_idx not in net.line.index:
        return line_idx, False
    
    net.line.at[line_idx, 'in_service'] = False
    return line_idx, True


def restore_line(net, line_idx):
    """
    Restore a tripped line (set in_service=True).
    
    Parameters
    ----------
    net : pandapower network
    line_idx : int
        Line index to restore
    
    Returns
    -------
    success : bool
    """
    if line_idx is None or line_idx not in net.line.index:
        return False
    
    net.line.at[line_idx, 'in_service'] = True
    return True


def set_switch_state(net, switch_name, closed):
    """
    Set switch state by name (S1, S2, S3, etc.)
    
    Parameters
    ----------
    net : pandapower network
    switch_name : str
        Switch name (e.g., 'S1', 'S2', 'S3')
    closed : bool
        True to close, False to open
    
    Returns
    -------
    success : bool
    """
    mask = net.switch['name'] == switch_name
    if mask.any():
        net.switch.loc[mask, 'closed'] = closed
        return True
    return False


def get_switch_controlled_line(net, switch_name):
    """
    Get the line index controlled by a named switch.
    
    Parameters
    ----------
    net : pandapower network
    switch_name : str
        Switch name (e.g., 'S1', 'S2', 'S3')
    
    Returns
    -------
    line_idx : int or None
        Line index if found and switch controls a line
    """
    mask = net.switch['name'] == switch_name
    if mask.any():
        idx = net.switch[mask].index[0]
        if net.switch.at[idx, 'et'] == 'l':  # Line switch
            return int(net.switch.at[idx, 'element'])
    return None


def get_critical_lines_for_evcs(net):
    """
    Get lines critical for EVCS connectivity.
    
    Returns lines that, if tripped, would affect EVCS buses.
    Reads EVCS bus IDs from microgrid_config (not hard-coded).
    
    Returns
    -------
    critical_lines : list of dict
        Each dict contains: line_idx, from_bus, to_bus, affected_evcs
    """
    # Read EVCS buses from network config (not hard-coded)
    config = net.get("microgrid_config", {})
    evcs_clusters = config.get("evcs_clusters", {})
    
    evcs_buses = set()
    evcs_bus_names = {}  # bus -> name mapping
    for name, cluster in evcs_clusters.items():
        bus = cluster.get('bus')
        if bus is not None:
            evcs_buses.add(bus)
            evcs_bus_names[bus] = name
    
    # Fallback if config not available
    if not evcs_buses:
        evcs_buses = {6, 10, 15}
        evcs_bus_names = {6: 'EVCS1', 10: 'EVCS2', 15: 'EVCS3'}
    
    critical = []
    
    for idx in net.line.index:
        fb = int(net.line.at[idx, 'from_bus'])
        tb = int(net.line.at[idx, 'to_bus'])
        
        # Check if line touches EVCS bus or is on path to EVCS
        affected = []
        if fb in evcs_buses:
            affected.append(f"{evcs_bus_names.get(fb, 'EVCS')}@Bus{fb}")
        if tb in evcs_buses:
            affected.append(f"{evcs_bus_names.get(tb, 'EVCS')}@Bus{tb}")
        
        # Also check if line is on critical path (buses 1-6, 1-10, 1-14)
        # This requires path analysis, simplified here
        if fb in {1, 2, 3, 4, 5} or tb in {1, 2, 3, 4, 5}:
            for evcs_bus, evcs_name in evcs_bus_names.items():
                if evcs_bus in {6}:  # EVCS1 on feeder 1
                    affected.append(f"path_to_{evcs_name}")
                    break
        
        if affected:
            critical.append({
                'line_idx': idx,
                'from_bus': fb,
                'to_bus': tb,
                'name': net.line.at[idx, 'name'],
                'affected_evcs': affected,
            })
    
    return critical


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    # Create network with mesh level 2 (full mesh)
    net = create_cigre_mv_microgrid(mesh_level=2)
    
    # Print summary
    print_network_summary(net)
    
    # Get topology for GNN
    adj, B = get_network_topology(net)
    print(f"\n[GNN Topology]")
    print(f"  Adjacency matrix shape: {adj.shape}")
    print(f"  B' matrix shape: {B.shape}")
    print(f"  Number of edges: {int(adj.sum() / 2)}")
    
    # Print bus table
    print(f"\n[Bus Table]")
    print(net.bus[['name', 'vn_kv', 'type']])