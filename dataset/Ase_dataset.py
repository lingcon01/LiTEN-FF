import numpy as np
from rdkit import Chem
import os
import sys
import tempfile
import shutil
from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
import logging
import warnings
import pickle
from ase.db import connect
import torch
from ase.io import read, write
from tqdm import tqdm
import lmdb
from ase import Atoms
import numpy as np
from ase.data import chemical_symbols


class ASEDataset(Dataset):
    def __init__(self, datapath, dataset_name=None):
        super(ASEDataset, self).__init__()

        self.datapath = datapath

        if dataset_name is None:
            self.dataset_name = os.listdir(datapath)

        print(f'data length: {len(self.dataset_name)}')

    def len(self):
        return len(self.dataset_name)

    def get(self, idx):

        abs_path = os.path.join(self.datapath, self.dataset_name[idx])

        atoms = read(abs_path)

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers()).long()
        positions = torch.Tensor(atoms.get_positions()).float()
        if positions.dim() == 3:
            natoms = positions.shape[0]
        else:
            natoms = 1

        # put the minimum data in torch geometric data object
        data = Data(pos=positions, z=atomic_numbers, natoms=natoms)

        return data


def build_atoms_from_data(data):
    # 将张量转为 numpy（如果还不是）
    pos = data.pos.cpu().numpy() if hasattr(data.pos, 'cpu') else np.asarray(data.pos)
    z = data.z.cpu().numpy() if hasattr(data.z, 'cpu') else np.asarray(data.z)

    symbols = [chemical_symbols[Z] for Z in z]  # 根据原子序数转换为元素符号

    atoms = Atoms(
        symbols=symbols,
        positions=pos,
    )

    return atoms
