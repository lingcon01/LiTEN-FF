import os
# os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"

import sys
import time
import torch
import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.units import fs, Hartree, Bohr
from torch_cluster import radius_graph
from ase.calculators.plumed import Plumed
from ase import units

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.data_pbc import *
from util.read_strucuture import *
from LITCalculator.LiTEN_Calculator import *

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(args.dtype, torch.float32)

    unit_energy = Hartree if args.unit_energy == 'Hartree' else 1.0
    unit_force = Hartree if args.unit_force == 'Hartree' else 1.0

    atoms = read_structure(args.input_file)  # 或 .gro, .cif 等
    if any(atoms.pbc):
        is_pbc = True
    else:
        is_pbc = False

    is_valid_atomic_numbers(torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long), args.model_name)

    config = {"device": device,
              "dtype": dtype,
              "unit_energy": unit_energy,
              "unit_force": unit_force,
              "cutoff": args.cutoff if hasattr(args, 'cutoff') else 5.0}

    print(f"\n=== Loading model: {args.model_name} ===")
    model = initial_model(args.model_name, is_pbc, device)

    atoms.calc = LiTENCalculator(model=model, model_name=args.model_name, is_pbc=is_pbc, is_batch=False, neighbor_type=args.neighbor_type, atoms=atoms, **config)

    if args.optimize:
        print("Running geometry optimization before MD...")
        print("Please wait a moment, model is on compiling...")
        opt = BFGS(atoms)
        opt.run(fmax=0.05, steps=args.maxstep)
        print("Geometry optimization complete. Starting MD...")

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    dyn = Langevin(atoms, timestep=args.timestep * fs, temperature_K=args.temperature, friction=args.friction / fs)

    dyn.attach(lambda: print(
        f"Step: {dyn.nsteps}, Energy: {float(atoms.get_potential_energy()):.3f} eV, Temperature: {atoms.get_temperature():.2f} K"
    ), interval=args.log_interval)

    traj = Trajectory(args.traj_output, 'w', atoms, properties=["energy", "forces"])
    dyn.attach(traj.write, interval=args.save_interval)

    dyn.run(args.steps)

    write(args.final_output, atoms)
    print(f'Total simulation finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MD with LiTEN-FF model using ASE.")

    parser.add_argument('--model_name', type=str, default='SPICE', help='Path to the saved model')
    parser.add_argument('--input_file', type=str, default=root_dir + '/example/H2O.pdb', help='Input structure file')
    parser.add_argument('--final_output', type=str, default='final.pdb', help='Final structure output file')
    parser.add_argument('--traj_output', type=str, default='trajectory.traj', help='Trajectory file output')
    parser.add_argument('--steps', type=int, default=1000000, help='Number of MD steps')
    parser.add_argument('--timestep', type=float, default=1.0, help='MD timestep in fs')
    parser.add_argument('--temperature', type=float, default=300, help='Temperature in Kelvin')
    parser.add_argument('--friction', type=float, default=0.01, help='Friction coefficient for Langevin thermostat')

    parser.add_argument('--log_interval', type=int, default=100, help='Print energy every N steps')
    parser.add_argument('--save_interval', type=int, default=100, help='Save trajectory every N steps')

    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32',
                        help='Torch float dtype')
    parser.add_argument('--neighbor_type', type=str, choices=['matscipy', 'nnpops'], default='matscipy',
                        help='Neighbor dtype, nnpops provide faster calculation by PyTorch')
    parser.add_argument('--unit_energy', type=str, default='Hartree', help='Energy unit (Hartree or 1.0)')
    parser.add_argument('--unit_force', type=str, default='Hartree', help='Force unit (Hartree or 1.0)')

    parser.add_argument('--optimize', type=bool, default=True, help='Run geometry optimization before MD')
    parser.add_argument('--maxstep', type=int, default=10, help='Run geometry optimization before MD')

    args = parser.parse_args()
    main(args)
