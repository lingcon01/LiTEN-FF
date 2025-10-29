import os
import sys
import time
import torch
import argparse
import numpy as np
import ase
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase import units
from ase.units import Hartree
from torch_cluster import radius_graph
from collections import defaultdict
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from ase.optimize import BFGS, FIRE
from tqdm import tqdm
from ase.data import chemical_symbols
import glob
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.read_strucuture import *
from LITCalculator.LiTEN_Calculator import *
from dataset.Ase_dataset import *
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compute_rmsd(a, b, align=True):
    """Calculate RMSD between two ASE Atoms objects with optional alignment"""
    pos_a = a.get_positions()
    pos_b = b.get_positions()

    # Ensure atomic order matches
    assert np.all(a.get_atomic_numbers() == b.get_atomic_numbers()), "Atomic types/order mismatch"

    # Center coordinates
    pos_a_centered = pos_a - np.mean(pos_a, axis=0)
    pos_b_centered = pos_b - np.mean(pos_b, axis=0)

    if align:
        # Kabsch algorithm for optimal rotation
        H = pos_a_centered.T @ pos_b_centered
        U, S, Vt = np.linalg.svd(H)
        rotation = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = Vt.T @ U.T

        pos_b_centered = pos_b_centered @ rotation

    return np.sqrt(np.mean(np.sum((pos_a_centered - pos_b_centered) ** 2, axis=1)))


def split_atoms_by_batch(atoms, batch):
    """Split a batch of atoms into individual molecules based on batch indices"""
    if hasattr(batch, 'cpu'):
        batch = batch.cpu().numpy()
    else:
        batch = np.asarray(batch)

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    num_molecules = batch.max() + 1

    molecules = [
        Atoms(
            symbols=[chemical_symbols[Z] for Z in numbers[batch == i]],
            positions=positions[batch == i]
        )
        for i in range(num_molecules)
    ]

    return molecules


def process_molecule_conformations(batch_dir, rmsd_threshold):
    """Process and save all conformations for each molecule in the batch"""
    mol_output_dir = os.path.join(batch_dir, "per_molecule_conformations")
    os.makedirs(mol_output_dir, exist_ok=True)

    molecule_conformations = defaultdict(list)

    # Find all cycle directories
    cycle_dirs = sorted([d for d in os.listdir(batch_dir)
                         if d.startswith("cycle_") and os.path.isdir(os.path.join(batch_dir, d))],
                        key=lambda x: int(x.split('_')[1]))

    # Collect all conformations
    for cycle_dir in cycle_dirs:
        current_dir = os.path.join(batch_dir, cycle_dir)

        mol_dirs = [d for d in os.listdir(current_dir)
                    if d.startswith("mol_") and os.path.isdir(os.path.join(current_dir, d))]

        for mol_dir in mol_dirs:
            mol_idx = int(mol_dir.split('_')[1])
            mol_path = os.path.join(current_dir, mol_dir)

            for conf_file in glob.glob(os.path.join(mol_path, "*.xyz")):
                try:
                    energy = float(conf_file.split('_')[-1].replace('.xyz', ''))
                    conf = read(conf_file)
                    conf.info["energy"] = energy
                    conf.info["mol_idx"] = mol_idx
                    conf.info["cycle"] = int(cycle_dir.split('_')[1])
                    molecule_conformations[mol_idx].append(conf)
                except Exception as e:
                    print(f"Error processing {conf_file}: {str(e)}")

    # Save conformations per molecule
    for mol_idx, conformations in molecule_conformations.items():
        if not conformations:
            continue

        # Sort by energy
        conformations.sort(key=lambda x: x.info["energy"])

        # Filter unique conformations by RMSD
        unique_confs = []
        for conf in conformations:
            is_unique = True
            for saved in unique_confs:
                rmsd = compute_rmsd(saved, conf, align=True)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_confs.append(conf)
                if len(unique_confs) > 1:
                    conf.info["min_rmsd_to_lower"] = min(
                        compute_rmsd(conf, saved) for saved in unique_confs[:-1]
                    )
                else:
                    conf.info["min_rmsd_to_lower"] = 0.0

        # Save results
        mol_file = os.path.join(mol_output_dir, f"mol_{mol_idx:03d}_all_conformations.xyz")
        try:
            write(mol_file, unique_confs)
            print(f"Saved {len(unique_confs)} unique conformations for molecule {mol_idx} to {mol_file}")

            sorted_file = os.path.join(mol_output_dir, f"mol_{mol_idx:03d}_sorted_by_energy.xyz")
            write(sorted_file, sorted(unique_confs, key=lambda x: x.info["energy"]))

        except Exception as e:
            print(f"Failed to save {mol_file}: {str(e)}")
            for conf in unique_confs:
                conf.calc = None
            write(mol_file, unique_confs)

    return molecule_conformations


# ----------------------------
# 1. High-Temperature Molecular Dynamics (MD)
# ----------------------------
def run_high_temp_md(atoms, temp=1000, steps=1000, timestep=1.0, trajectory_file="md.traj", save_interval=10):
    """Run high-temperature MD simulation and save trajectory"""
    MaxwellBoltzmannDistribution(atoms, temp * units.kB)
    dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=temp, friction=0.01)

    dyn.attach(lambda: print(
        f"Step: {dyn.nsteps}, Energy: {atoms.get_potential_energy():.3f} eV, Temperature: {atoms.get_temperature():.2f} K"
    ), interval=args.log_interval)

    traj = Trajectory(trajectory_file, 'w', atoms)
    dyn.attach(traj.write, interval=save_interval)

    print(f"Running MD at {temp} K...")
    dyn.run(steps)

    return trajectory_file


# ----------------------------
# 2. Simulated Annealing
# ----------------------------
def simulated_annealing(atoms, initial_temp=1000, final_temp=300, timestep=1.0, steps=1000, save_interval=10):
    """Perform linear cooling simulated annealing"""
    dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=initial_temp, friction=0.01)

    dyn.attach(lambda: print(
        f"Step: {dyn.nsteps}, Energy: {atoms.get_potential_energy():.3f} eV, Temperature: {atoms.get_temperature():.2f} K"
    ), interval=args.log_interval)

    print(f"Annealing from {initial_temp} K to {final_temp} K...")
    for step in tqdm(range(steps), desc="Annealing Progress"):
        temp = initial_temp - (initial_temp - final_temp) * step / steps
        dyn = Langevin(atoms, timestep=timestep * units.fs, temperature_K=temp, friction=0.01)
        dyn.attach(lambda: print(
            f"Step: {dyn.nsteps}, Energy: {atoms.get_potential_energy():.3f} eV, Temperature: {atoms.get_temperature():.2f} K"
        ), interval=args.log_interval)
        dyn.run(10)

    # Quench to local minimum
    opt = FIRE(atoms)
    opt.run(fmax=0.05, steps=200)
    return atoms


# ----------------------------
# 3. Optimization
# ----------------------------
def optimize_and_save_batch(annealed, batch, output_dir, maxstep, molecule_conformations):
    """Optimize and save batch of molecules with energy tracking"""
    total_energy = annealed.get_potential_energy()
    molecule_energies = annealed.calc.results['true_energy']  # shape: (num_molecules,)

    current_molecules = split_atoms_by_batch(annealed, batch.batch)

    for i, mol in enumerate(current_molecules):
        mol_energy = molecule_energies[i]

        mol_dir = os.path.join(output_dir, f"mol_{i:03d}")
        os.makedirs(mol_dir, exist_ok=True)

        mol_path = os.path.join(mol_dir, f"conf_energy_{mol_energy:.3f}.xyz")
        write(mol_path, mol)

        mol.info['energy'] = mol_energy
        mol.info['mol_idx'] = i

        molecule_conformations[i].append(mol)

    batch_path = os.path.join(output_dir, f"batch_energy_{total_energy:.3f}.xyz")
    write(batch_path, annealed)

    return total_energy, molecule_energies, current_molecules


class BatchSamplingConvergenceChecker:
    """Convergence checker for batch sampling based on RMSD analysis"""

    def __init__(self, max_no_new_minima=4, rmsd_threshold=0.2,
                 max_minima_cache=100, cluster_stability_window=3, verbose=True):
        self.max_no_new_minima = max_no_new_minima
        self.rmsd_threshold = rmsd_threshold
        self.max_minima_cache = max_minima_cache
        self.cluster_stability_window = cluster_stability_window
        self.verbose = verbose

        self.reset()

    def _compute_rmsd_cached(self, a, b):
        """Compute RMSD with caching to avoid redundant calculations"""
        key = (id(a), id(b))
        if key not in self._rmsd_cache:
            if len(self._rmsd_cache) > 1000:
                self._rmsd_cache.pop(next(iter(self._rmsd_cache)))
            self._rmsd_cache[key] = compute_rmsd(a, b)
        return self._rmsd_cache[key]

    def _are_struct_sets_similar(self, structs1, structs2):
        """Check similarity between two sets of structures"""
        if len(structs1) == 0 or len(structs2) == 0:
            return False

        if len(structs1) != len(structs2):
            return False

        match_count = 0
        for s1, s2 in zip(structs1, structs2):
            rmsd = self._compute_rmsd_cached(s1, s2)
            if rmsd < self.rmsd_threshold:
                match_count += 1

        match_ratio = match_count / len(structs1)
        return match_ratio > 0.9

    def is_new_minimum(self, new_structs):
        """Check if any structure in the batch represents a new minimum"""
        batch_size = len(new_structs)
        is_new_list = []
        closest_idx_list = []
        closest_rmsd_list = []

        for i in range(batch_size):
            new_struct = new_structs[i]
            closest_rmsd = float('inf')
            closest_idx = -1
            is_new = True

            for j, old_structs in enumerate(self.minima_structures):
                if i < len(old_structs):
                    rmsd_val = self._compute_rmsd_cached(new_struct, old_structs[i])

                    if rmsd_val < closest_rmsd:
                        closest_rmsd = rmsd_val
                        closest_idx = j

                    if rmsd_val < self.rmsd_threshold:
                        is_new = False
                        if self.verbose:
                            print(f"Molecule {i}: Found similar structure - RMSD={rmsd_val:.3f} Å")
                        break

            is_new_list.append(is_new)
            closest_idx_list.append(closest_idx)
            closest_rmsd_list.append(closest_rmsd)

        return is_new_list, closest_idx_list, closest_rmsd_list

    def update(self, new_structs_batch):
        """Update convergence status with new batch of structures"""
        if self.converged:
            if self.verbose:
                print("Warning: Convergence already reached, ignoring update")
            return True

        is_new_list, closest_idx_list, closest_rmsd_list = self.is_new_minimum(new_structs_batch)

        new_minima_found = any(is_new_list)
        added_count = sum(is_new_list)

        if new_minima_found:
            self.minima_structures.append(new_structs_batch)
            self.no_new_minima_count = 0
            self._last_minima_count = len(self.minima_structures)

            if self.verbose:
                for i, is_new in enumerate(is_new_list):
                    if is_new:
                        print(f"New structure found for molecule {i}. "
                              f"Closest existing: idx={closest_idx_list[i]}, RMSD={closest_rmsd_list[i]:.3f} Å")

            if len(self.minima_structures) > self.max_minima_cache:
                self.minima_structures.pop(0)
        else:
            self.no_new_minima_count += 1

        self.last_cluster_sets.append(new_structs_batch)
        if len(self.last_cluster_sets) > self.cluster_stability_window:
            self.last_cluster_sets.pop(0)

        structure_sets_stable = False
        if len(self.last_cluster_sets) >= self.cluster_stability_window:
            reference = self.last_cluster_sets[-1]
            structure_sets_stable = all(
                self._are_struct_sets_similar(reference, prev)
                for prev in self.last_cluster_sets[:-1]
            )

        self.history.append({
            'cycle': len(self.history),
            'new_minima': added_count,
            'total_minima': len(self.minima_structures),
            'consecutive_no_new': self.no_new_minima_count,
            'structure_sets_stable': structure_sets_stable
        })

        if self.no_new_minima_count >= self.max_no_new_minima and structure_sets_stable:
            self.converged = True
            if self.verbose:
                print(f"\n=== Convergence Reached ===\n"
                      f"Total structures found: {len(self.minima_structures)}\n"
                      f"No new structures for {self.no_new_minima_count} cycles\n"
                      f"Stable structure sets across {self.cluster_stability_window} cycles")
            return True

        if self.verbose:
            print(f"Convergence status: {self.no_new_minima_count}/{self.max_no_new_minima} "
                  f"cycles without new structures; "
                  f"structure sets stable: {structure_sets_stable}")

        return False

    def get_minima_info(self):
        """Return list of structure batches"""
        return self.minima_structures

    def reset(self):
        """Reset all internal state"""
        self.no_new_minima_count = 0
        self.minima_structures = []
        self.converged = False
        self.history = []
        self._rmsd_cache = {}
        self._last_minima_count = 0
        self.last_cluster_sets = []

def main(args):
    """Main execution function for conformational sampling workflow"""
    start = time.time()
    torch.set_float32_matmul_precision('high')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(args.dtype, torch.float32)

    unit_energy = Hartree if args.unit_energy == 'Hartree' else 1.0
    unit_force = Hartree if args.unit_force == 'Hartree' else 1.0

    print("\n=== Loading dataset ===")
    dataset = ASEDataset(datapath=args.input_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    config = {"device": device,
              "dtype": dtype,
              "unit_energy": unit_energy,
              "unit_force": unit_force,
              "cutoff": args.cutoff if hasattr(args, 'cutoff') else 5.0}

    print(f"\n=== Loading model: {args.model_name} ===")
    model = initial_model(args.model_name, False, device)

    for batch_idx, batch in enumerate(dataloader):
        print(f"\n=== Processing batch {batch_idx} ===")
        batch_dir = os.path.join(args.output_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)

        # Create ASE Atoms object for the entire batch
        positions = batch.pos.cpu().numpy()
        atomic_numbers = batch.z.cpu().numpy()
        atoms = ase.Atoms(
            numbers=atomic_numbers,
            positions=positions,
        )

        calculator = LiTENCalculator(model=model, model_name=args.model_name, is_pbc=False, is_batch=True, data=batch, **config)
        atoms.calc = calculator

        # Initial geometry optimization
        if args.optimize:
            print("\n=== Running initial geometry optimization ===")
            opt = BFGS(atoms)
            opt.run(fmax=0.05, steps=args.maxstep)
            print("Geometry optimization completed")

        # Initialize convergence checker
        convergence_checker = BatchSamplingConvergenceChecker(
            max_no_new_minima=args.max_no_new_minima,
            rmsd_threshold=args.rmsd_threshold,
            cluster_stability_window=2,
            verbose=True
        )

        best_energy = float('inf')
        best_structure = None
        molecule_conformations = {i: [] for i in range(len(torch.unique(batch.batch)))}

        for cycle in range(args.max_cycles):
            print(f"\n=== Cycle {cycle + 1}/{args.max_cycles} ===")
            cycle_output_dir = os.path.join(batch_dir, f"cycle_{cycle}")
            os.makedirs(cycle_output_dir, exist_ok=True)

            # Step 1: High-Temperature MD
            print(f"Running high-T MD (T={args.initial_temp}K) for batch")
            md_traj_file = os.path.join(cycle_output_dir, "batch_high_temp.traj")

            md_traj = run_high_temp_md(
                atoms,
                temp=args.initial_temp,
                steps=args.high_steps,
                timestep=args.timestep,
                trajectory_file=md_traj_file,
                save_interval=args.save_interval
            )

            # Step 2: Simulated Annealing
            atoms = read(md_traj, index=-1)
            atoms.calc = calculator

            atoms_an = atoms.copy()
            atoms_an.calc = calculator

            annealed = simulated_annealing(
                atoms_an,
                initial_temp=args.initial_temp,
                final_temp=args.final_temp,
                timestep=args.timestep,
                steps=args.anneal_steps,
                save_interval=args.save_interval
            )

            # Step 3: Optimization
            current_energy, current_energies, current_molecules = optimize_and_save_batch(
                annealed, batch, cycle_output_dir, args.maxstep, molecule_conformations
            )

            # Step 4: Check convergence
            converged = convergence_checker.update(current_molecules)

            if current_energy < best_energy:
                best_energy = current_energy
                best_structure = annealed.copy()
                best_structure.calc = calculator

            if converged:
                print("Batch sampling has converged!")
                break

        # Save final best structure
        if best_structure is not None:
            final_path = os.path.join(batch_dir, "batch_final_best.xyz")
            write(final_path, best_structure)
            print(f"Saved final best batch structure (energy={best_energy:.3f}eV)")

        # Post-processing: Save all conformations per molecule
        print("\n=== Processing batch files ===")
        process_molecule_conformations(batch_dir, args.rmsd_threshold)

    print("\n=== All batches processed ===")
    end = time.time()
    print(f"Total running time: {end - start:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MD with LumiForce model using ASE.")

    # I/O parameters
    parser.add_argument('--model_name', type=str, default='nablaDFT', help='Path to the saved model')
    parser.add_argument('--input_dir', type=str, default=root_dir + '/example/under_25', help='Input structure file')
    parser.add_argument('--final_output', type=str, default='final_structure.pdb', help='Final structure output file')
    parser.add_argument('--high_trajectory', type=str, default='high_trajectory.traj', help='High-temp trajectory file')
    parser.add_argument('--output_dir', type=str, default='./output_conf/drug/10', help='Output directory')

    # Simulation parameters
    parser.add_argument('--initial_temp', type=int, default=600, help='Initial temperature (K)')
    parser.add_argument('--high_steps', type=int, default=100, help='Number of high-T MD steps')
    parser.add_argument('--final_temp', type=int, default=300, help='Final temperature (K)')
    parser.add_argument('--anneal_steps', type=int, default=10, help='Number of annealing steps')
    parser.add_argument('--timestep', type=float, default=1.0, help='MD timestep (fs)')
    parser.add_argument('--friction', type=float, default=0.01, help='Friction coefficient')

    # Control parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Print interval for MD')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval for trajectories')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32',
                        help='Floating point precision')
    parser.add_argument('--unit_energy', type=str, default='Hartree', help='Energy unit')
    parser.add_argument('--unit_force', type=str, default='Hartree', help='Force unit')
    parser.add_argument('--max_cycles', type=int, default=250, help='Max sampling cycles')
    parser.add_argument('--max_no_new_minima', type=int, default=5, help='Max cycles without new minima')
    parser.add_argument('--rmsd_threshold', type=float, default=0.4, help='RMSD threshold for convergence')
    parser.add_argument('--batch_size', type=float, default=100, help='Batch size of molecules for simulation')

    # Optimization parameters
    parser.add_argument('--optimize', type=bool, default=False, help='Run initial geometry optimization')
    parser.add_argument('--maxstep', type=int, default=250, help='Max optimization steps')

    args = parser.parse_args()
    main(args)
