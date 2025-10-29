#!/usr/bin/env python3
import re
import numpy as np
import os

EV_TO_KJ_PER_MOL = 96.485
R_kJmolK = 8.314462618e-3

# -------- 1) 能量读取 --------
PATTERNS = [
    r'\bEnergy:\s*([+-]?\d+(?:\.\d+)?)\s*eV',
    r'\bU\(pot\):\s*([+-]?\d+(?:\.\d+)?)\s*eV',
    r'\bPotential\s+Energy:\s*([+-]?\d+(?:\.\d+)?)\s*eV',
]

def read_energies(filename):
    patterns = [re.compile(p) for p in PATTERNS]
    vals = []
    with open(filename, 'r') as f:
        for line in f:
            for pat in patterns:
                m = pat.search(line)
                if m:
                    vals.append(float(m.group(1)))
                    break
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        raise ValueError(f"[解析失败] {filename} 没找到能量字段")
    return arr

# -------- 2) 平均能量 --------
def average_last_window(energies, window_ps, timestep_fs=1.0, stride=100):
    frame_interval_ps = timestep_fs * stride * 1e-3
    nframes_window = int(np.ceil(window_ps / frame_interval_ps))
    if len(energies) < nframes_window:
        raise ValueError(f"轨迹不足 {window_ps} ps，只有 {len(energies)} 帧")
    return energies[-nframes_window:].mean()

# -------- 3) 计算 ΔU, ΔH --------
def compute_vap(E_gas_avg_eV, E_liq_avg_eV, n_atoms_liq, n_atoms_gas=3, T_K=298.15):
    if n_atoms_liq % n_atoms_gas != 0:
        raise ValueError("n_atoms_liq 不是 n_atoms_gas 的整数倍")
    n_mol_liq = n_atoms_liq // n_atoms_gas
    E_liq_per_mol_eV = E_liq_avg_eV / n_mol_liq
    E_gas_per_mol_eV = E_gas_avg_eV
    delta_U_kJmol = (E_gas_per_mol_eV - E_liq_per_mol_eV) * EV_TO_KJ_PER_MOL
    delta_H_kJmol = delta_U_kJmol + R_kJmolK * T_K
    return delta_U_kJmol, delta_H_kJmol

# -------- 4) 主函数，多温度处理 --------
def run_vap_calc(single_dir, liq_dir, temps, n_atoms_liq,
                 timestep_fs=1.0, stride=100, window_ps=80.0):

    results = []
    for T in temps:
        gas_file = os.path.join(single_dir, f"run_{T}K.log")
        liq_file = os.path.join(liq_dir, f"run_{T}K.log")

        if not os.path.exists(gas_file):
            print(f"[警告] {gas_file} 不存在，跳过 {T}K")
            continue
        if not os.path.exists(liq_file):
            print(f"[警告] {liq_file} 不存在，跳过 {T}K")
            continue

        E_gas_all = read_energies(gas_file)
        E_liq_all = read_energies(liq_file)

        E_gas_avg = average_last_window(E_gas_all, window_ps, timestep_fs, stride)
        E_liq_avg = average_last_window(E_liq_all, window_ps, timestep_fs, stride)

        dU, dH = compute_vap(E_gas_avg, E_liq_avg, n_atoms_liq, n_atoms_gas=3, T_K=T)
        results.append((T, dU, dH))

    return results


if __name__ == "__main__":
    single_dir = "../single_run"   # 单分子气相目录
    liq_dir = "../run_data"        # 液相目录
    temps = [273, 277, 293, 298, 313, 323, 333]
    n_atoms_liq = 1593             # 液相盒子原子数

    results = run_vap_calc(single_dir, liq_dir, temps, n_atoms_liq)

    print("\n=== 蒸发焓计算结果 ===")
    print(" T (K)   ΔU (kJ/mol)   ΔH (kJ/mol)")
    print("----------------------------------")
    for T, dU, dH in results:
        print(f"{T:5d}   {dU:10.2f}   {dH:10.2f}")

