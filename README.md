2026 The 10th International Conference on Green Energy and Applications (ICGEA)
Các file chính
--------------
1) sim_MARL_F5_v2.py
   - Entry-point huấn luyện/đánh giá MARL (MAPPO + GNN).
   - Tạo môi trường MicrogridEnv, khởi tạo SharedPolicy, chạy loop train/eval.
   - Có cấu hình mặc định để chạy nhanh bằng F5/Run (IDE).

2) env_v2.py
   - Môi trường MicrogridEnv (MARL-ready) dựa trên pandapower.
   - Mô phỏng power flow AC và động lực tần số SG (swing equation).
   - Hỗ trợ kịch bản line trip/restore, tải/RES ngoại sinh, và EVCS agents.

3) Magent_v2.py
   - Triển khai MAPPO tối giản:
     * Shared actor (parameter sharing) cho các EVCS agents.
     * Centralized critic.
     * Hỗ trợ PPO update, GAE, và RolloutBuffer.

4) GNN_v2.py
   - MPNN encoder tối giản (không dùng torch_geometric).
   - Nhận graph (node_x, edge_index, edge_attr) -> (node embeddings, global embedding).

5) cigre_mv_microgrid_v1.py
   - Hàm tạo lưới điện CIGRE MV (mạng pandapower) và config EVCS.

6) sim_v1.py, sim_droop_v3.py
   - Các mô phỏng độc lập/phi RL để kiểm tra droop/động lực SG.

7) safe_line.py
   - Logic/tiện ích liên quan đến an toàn đường dây (nếu có dùng).

8) Magent_v2.py / GNN_v2.py
   - Các module thuật toán/ML phục vụ MARL.

9) data/
   - Thư mục dữ liệu phụ trợ (nếu có).

Cách sử dụng cơ bản
-------------------
Yêu cầu:
- Python 3.8+
- Các thư viện: numpy, torch, pandapower

Cài đặt thư viện (gợi ý):
    pip install numpy torch pandapower

Chạy huấn luyện (train):
    python sim_MARL_F5_v2.py --episodes 500 --device cpu

Chạy đánh giá (eval) với checkpoint:
    python sim_MARL_F5_v2.py --eval --episodes 20 --ckpt checkpoints/mappo_latest.pt

Cấu hình nhanh bằng IDE (VS Code/PyCharm):
- Mở file sim_MARL_F5_v2.py và nhấn Run/F5.
- Có thể đổi DEFAULT_MODE / DEFAULT_EPISODES trong file để chạy nhanh.

Luồng hoạt động tổng quát
-------------------------
1) sim_MARL_F5_v2.py gọi make_env() để tạo MicrogridEnv.
2) env_v2.py tạo obs gồm:
   - graph: node_x, edge_index, edge_attr + các chỉ số hệ thống (df, rocof).
   - local: trạng thái cục bộ cho từng EVCS agent.
3) policy (Magent_v2.py) dùng GNN_v2.py để encode graph.
4) Actor sinh action cho từng agent; môi trường bước (step) và trả reward.
5) MAPPO update dựa trên rollout.
