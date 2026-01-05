import pandapower.topology as top

SLACK_BUS = 1
TARGET_BUSES = [6, 10, 15, 13]  # EVCS1, EVCS2, EVCS3, Wind (tuỳ bạn)

def is_safe_trip(net, line_idx):
    # clone net nhẹ (hoặc net.deepcopy nếu bạn có)
    net2 = net.deepcopy()
    net2.line.at[line_idx, "in_service"] = False

    # tạo graph có xét trạng thái switch/line
    G = top.create_nxgraph(net2, include_trafos=True, respect_switches=True)

    # check các bus mục tiêu còn nối về slack không
    for b in TARGET_BUSES:
        if b in G and SLACK_BUS in G:
            if not top.nx.has_path(G, SLACK_BUS, b):
                return False
        else:
            return False
    return True

safe_lines = []
for idx in net.line.index:
    if not net.line.at[idx, "in_service"]:
        continue
    if is_safe_trip(net, idx):
        safe_lines.append(idx)

print("Safe line candidates:", safe_lines)
