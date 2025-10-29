import os
import sys
from typing import Optional
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.data import chemical_symbols

# Allowed element sets
nab_allowed_elements = {"H", "C", "N", "O", "S", "Cl", "F", "Br"}
spice_allowed_elements = {
    "H", "Li", "B", "C", "N", "O", "F",
    "Na", "Mg", "Si", "P", "S", "Cl",
    "K", "Ca", "Br", "I"
}


def is_valid_atomic_numbers(Z_list, model_name: str) -> bool:
    model_name = model_name.lower()
    if model_name == 'nabladft':
        allowed = nab_allowed_elements
    elif model_name == 'spice':
        allowed = spice_allowed_elements
    else:
        raise ValueError(f"[is_valid_atomic_numbers] Unknown model name: {model_name}")

    for Z in Z_list:
        symbol = chemical_symbols[Z]
        if symbol not in allowed:
            raise ValueError(f"Element '{symbol}' (Z={Z}) is not allowed for model '{model_name}'.")
    return True


def convert_rdkit_mol_to_ase(mol) -> Optional[Atoms]:
    """Convert RDKit Mol to ASE Atoms"""
    if mol is None:
        return None
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    conf = mol.GetConformer()

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [conf.GetAtomPosition(i) for i in range(len(symbols))]
    positions = [(pos.x, pos.y, pos.z) for pos in positions]

    return Atoms(symbols=symbols, positions=positions)


def read_structure(file_path: str) -> Optional[Atoms]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.xyz', '.pdb']:
        try:
            atoms = read(file_path)
            return atoms
        except Exception as e:
            print(f"[ASE] Error reading {file_path}: {e}")
            return None

    elif ext in ['.mol', '.sdf', '.mol2']:
        try:
            suppl = Chem.SDMolSupplier(file_path) if ext == '.sdf' else [
                Chem.MolFromMol2File(file_path, removeHs=False)]
            mol = suppl[0] if suppl else None
            if mol is None:
                raise ValueError("RDKit failed to parse molecule.")
            return convert_rdkit_mol_to_ase(mol)
        except Exception as e:
            print(f"[RDKit] Error reading {file_path}: {e}")
            return None

    else:
        print(f"[Unsupported] File extension {ext} not supported.")
        return None


# Example usage:
if __name__ == "__main__":
    file = sys.argv[1]  # e.g., python convert_to_atoms.py mol.sdf
    atoms = read_structure(file)
    if atoms:
        print(f"[Success] Loaded {len(atoms)} atoms from {file}")
        print(atoms)
    else:
        print("[Failure] Could not convert file to ASE Atoms.")
