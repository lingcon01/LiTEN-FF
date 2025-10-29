import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union
import pickle
# from torch_nl import compute_neighborlist, ase2data

try:
    torch_compile = torch.compile  # PyTorch 2.x
except AttributeError:
    torch_compile = torch.jit.script  # fallback for older versions

from matscipy.neighbours import neighbour_list
sys.path.append('.')
current_dir = os.path.dirname(os.path.abspath(__file__))


# def pbc_neighbors_torchnl_single(atoms, cutoff=5.0, loop=True, device=torch.device("cuda")):
# 
#     pos, cell, pbc, batch, n_atoms = ase2data([atoms])
#     pos, cell, pbc, batch = pos.to(device), cell.to(device), pbc.to(device), batch.to(device)
# 
#     mapping, batch_mapping, shifts_idx = compute_neighborlist(
#         cutoff, pos, cell, pbc, batch, loop
#     )
# 
#     edge_index = mapping.contiguous().long()
#     edge_shift = shifts_idx.contiguous().float()
# 
#     return edge_index, edge_shift

@torch_compile
def pbc_neighbors(atoms, cutoff=5.0, loop=True):
    
    i, j, S = neighbour_list("ijS", atoms=atoms, cutoff=cutoff)

    mask = ~((i == j))
    i, j, S = i[mask], j[mask], S[mask]

    if loop:
        n = len(atoms)
        self_i = np.arange(n)
        self_j = self_i.copy()
        self_S = np.zeros((n, 3))
    
        i = np.concatenate([i, self_i])
        j = np.concatenate([j, self_j])
        S = np.concatenate([S, self_S])

    edge_index = torch.from_numpy(np.stack([i, j], axis=0)).long()
    edge_shift = torch.from_numpy(S).float()

    return edge_index, edge_shift

@torch_compile
def pbc_neighbors2(positions, cutoff, pbc, cell, loop=True):
    i, j, S = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )

    mask = ~((i == j))
    i, j, S = i[mask], j[mask], S[mask]

    if loop:
        n = len(positions)
        self_i = np.arange(n)
        self_j = self_i.copy()
        self_S = np.zeros((n, 3))

        i = np.concatenate([i, self_i])
        j = np.concatenate([j, self_j])
        S = np.concatenate([S, self_S])

    edge_index = torch.from_numpy(np.stack([i, j], axis=0)).long()
    edge_shift = torch.from_numpy(S).float()

    return edge_index, edge_shift


@torch_compile
class AtomicEnergiesBlock(nn.Module):
    def __init__(self, z_charge_energy: Dict[int, Dict[int, float]] = None):
        super().__init__()

        if z_charge_energy is None:
            file_path = os.path.join(current_dir, 'spice_energy.pkl')
            with open(file_path, 'rb') as f:
                z_charge_energy = pickle.load(f)
        self.z_max = max(z_charge_energy.keys())
        self.charge_min = min(min(qs) for qs in z_charge_energy.values())
        self.charge_max = max(max(qs) for qs in z_charge_energy.values())
        self.charge_offset = -self.charge_min

        energy_table = torch.full(
            (self.z_max + 1, self.charge_max - self.charge_min + 1),
            float('nan'),
            dtype=torch.get_default_dtype()
        )
        for z, charges in z_charge_energy.items():
            for q, e in charges.items():
                energy_table[z, q + self.charge_offset] = e
        self.register_buffer("energy_table", energy_table)

    def forward(self, z: torch.Tensor, charge: Union[torch.Tensor, None] = None) -> torch.Tensor:
        """
        Args:
            z: [N] atomic numbers
            charge: [N] atomic charges (optional, default to 0)
        Returns:
            energies: [N] atomic ground state energies
        """
        if charge is None:
            charge = torch.zeros_like(z)

        charge_idx = charge + self.charge_offset
        energies = self.energy_table[z, charge_idx]

        if torch.isnan(energies).any():
            raise ValueError("Some Z/charge combinations are not in the energy table.")

        return energies
