import os
import sys
import time
import torch
import argparse
import numpy as np
from ase import Atoms
from typing import List, Tuple, Optional
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.units import fs, Hartree
from torch_cluster import radius_graph
from torch_scatter import scatter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.LiTEN_FF import LiTEN
from util.data_pbc import *
from NNPOps.neighbors import getNeighborPairs

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Nab_path = os.path.join(parent_dir, 'checkpoints', 'LiTEN_nablaDFT.model')
SPICE_path = os.path.join(parent_dir, 'checkpoints', 'LiTEN_SPICE.model')
PBC_path = os.path.join(parent_dir, 'checkpoints', 'LiTEN_PBC.model')

model_checkpoints = {
    "LiTEN_Nab": Nab_path,
    "LiTEN_SPICE": SPICE_path,
    "LiTEN_PBC": PBC_path
}


class NabEnergyTable:
    def __init__(self):
        self.energy_table = torch.tensor([
            0.00000000e+00, -5.02493258e-01, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -3.77980993e+01, -5.45224148e+01,
            -7.49766781e+01, -9.96144695e+01, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -3.41139419e+02,
            -3.97971928e+02, -4.59988650e+02, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.57385717e+03,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, -2.97756429e+02
        ])

    def get_energy(self, Z):
        try:
            Z_tensor = torch.as_tensor(Z, dtype=torch.long)
            if torch.any(Z_tensor >= len(self.energy_table)) or torch.any(Z_tensor < 0):
                raise ValueError(f"Invalid atomic number Z: {Z}")
            return self.energy_table[Z_tensor]
        except Exception as e:
            print(f"[EnergyTable] Error: {e}")
            return None


def get_force(energy, pos):
    # calculate_forces:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]

    grads = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[pos],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=False,  # Make sure the graph is not destroyed during training
        allow_unused=True,  # For complete dissociation turn to true
        create_graph=False,  # Create graph for second derivative
    )[0]  # [n_nodes, 3]

    if grads is None:
        forces = torch.zeros_like(pos)
    else:
        forces = -grads

    return forces

def initial_model(model_name, is_pbc, device):

    model_name = model_name.lower()

    if model_name == 'nabladft' and is_pbc is not True:
        model_path = model_checkpoints['LiTEN_Nab']
    elif model_name == 'spice' and is_pbc is not True:
        model_path = model_checkpoints['LiTEN_SPICE']
    elif model_name == 'spice' and is_pbc:
        model_path = model_checkpoints['LiTEN_PBC']
    else:
        raise ValueError("nablaDFT is not supportted pbc")

    model = torch.load(model_path, weights_only=False, map_location=device)
    model = torch.compile(model, fullgraph=True)

    return model


class LiTENCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, model_name, is_pbc, is_batch, neighbor_type='matscipy', atoms=None, data=None, device=None, dtype=None, unit_energy=None,
                 unit_force=None, cutoff=None, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.dtype = dtype
        self.unit_energy = unit_energy
        self.unit_force = unit_force
        self.is_pbc = is_pbc
        self.is_batch = is_batch
        self.model_name = model_name.lower()
        self.cutoff = cutoff
        self.neighbor_type = neighbor_type

        self.model = model.eval()

        if atoms is not None:
            self.z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            self.batch = torch.tensor([0] * len(self.z))
        elif self.is_batch and data is not None:
            self.z = data.z
            self.batch = data.batch
        else:
            raise ValueError("[LumiForceCalculator] Atomic numbers (z) must be provided.")

        if self.is_pbc:
            self.cell = torch.from_numpy(atoms.cell.array).float().to(self.device)
            self.pbc = tuple(atoms.pbc)

        if self.model_name == 'nabladft':
            self.energy_table = NabEnergyTable()
            self.single_energy = self.energy_table.get_energy(self.z).to(torch.float64)
        else:
            self.energy_table = AtomicEnergiesBlock()
            self.single_energy = self.energy_table(self.z).to(torch.float64)

        self.base_energy = scatter(self.single_energy, index=self.batch, dim=0, reduce="sum",
                                   dim_size=self.batch[-1].item() + 1).to(self.dtype)
        self.z = self.z.to(self.device)
        self.batch = self.batch.to(self.device)

    def _getNeighborPairs(
                self, positions: torch.Tensor, cell: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Get the shifts and edge indices.

                Notes
                -----
                This method calculates the shifts and edge indices by determining neighbor pairs (``neighbors``)
                and respective wrapped distances (``wrappedDeltas``) using ``NNPOps.neighbors.getNeighborPairs``.
                After obtaining the ``neighbors`` and ``wrappedDeltas``, the pairs with negative indices (r>cutoff)
                are filtered out, and the edge indices and shifts are finally calculated.

                Parameters
                ----------
                positions : torch.Tensor
                    The positions of the atoms.
                cell : torch.Tensor
                    The cell vectors.

                Returns
                -------
                edgeIndex : torch.Tensor
                    The edge indices.
                shifts : torch.Tensor
                    The shifts.
                """
                # Get the neighbor pairs, shifts and edge indices.

                neighbors, wrappedDeltas, _, _ = getNeighborPairs(
                    positions, 5.0, -1, cell
                )
                mask = neighbors >= 0
                neighbors = neighbors[mask].view(2, -1)
                wrappedDeltas = wrappedDeltas[mask[0], :]

                edgeIndex = torch.hstack((neighbors, neighbors.flip(0))).to(torch.int64)

                if cell is not None:

                    deltas = positions[edgeIndex[0]] - positions[edgeIndex[1]]
                    wrappedDeltas = torch.vstack((wrappedDeltas, -wrappedDeltas))
                    shiftsIdx = torch.mm(deltas - wrappedDeltas, torch.linalg.inv(cell))

                    i, j = edgeIndex[0], edgeIndex[1]

                    n = positions.size(0)
                    self_i = torch.arange(n, device=positions.device)
                    self_j = self_i.clone()
                    self_S = torch.zeros((n, 3), device=positions.device)

                    i = torch.cat([i, self_i])
                    j = torch.cat([j, self_j])
                    S = torch.cat([shiftsIdx, self_S])

                    shifts = torch.mm(S, cell)
                    edgeIndex = torch.stack([i, j], dim=0)
                else:
                    shifts = torch.zeros((edgeIndex.shape[1], 3), dtype=self.dtype, device=positions.device)

                return edgeIndex, shifts


    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        pos = torch.from_numpy(atoms.get_positions()).to(dtype=self.dtype, device=self.device)
        pos.requires_grad_(True)
        if self.is_pbc:
            if self.neighbor_type == 'matscipy':
                edge_index, edge_shift = pbc_neighbors(atoms, cutoff=self.cutoff, loop=True)
                shift = edge_shift.to(self.device) @ self.cell
            elif self.neighbor_type == 'nnpops':
                edge_index, shift = self._getNeighborPairs(pos, self.cell)
            else:
                print('Please choose neighbor type in [matscipy, nnpops]')
        else:
            edge_index = radius_graph(pos, r=self.cutoff, loop=True, batch=self.batch, max_num_neighbors=1000)
            shift = None

        energy = self.model(z=self.z, pos=pos, batch=self.batch, edge_index=edge_index.to(self.device),
                            n_graph=self.batch[-1].item() + 1, shift=shift)

        forces = get_force(energy, pos)

        if self.is_batch:
            true_energy = (energy.detach().cpu().numpy() + self.base_energy.numpy()) * self.unit_energy
            true_forces = (forces * self.unit_force).cpu().numpy()
            self.results['true_energy'] = true_energy
            self.results['true_forces'] = true_forces

            self.results['energy'] = (energy.detach().cpu().numpy().sum() + self.base_energy.numpy().sum()) * self.unit_energy
        else:
            self.results['energy'] = (energy.item() + self.base_energy.item()) * self.unit_energy

        self.results['forces'] = (forces * self.unit_force).cpu().numpy()
