import os
import sys
from rdkit.Chem import rdmolfiles
from ase.io import read, write
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.data_pbc import *
from util.read_strucuture import *
from LITCalculator.LiTEN_Calculator import *
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dtype_map = {'float32': torch.float32, 'float64': torch.float64}
    dtype = dtype_map.get(args.dtype, torch.float32)

    unit_energy = Hartree if args.unit_energy == 'Hartree' else 1.0
    unit_force = Hartree if args.unit_force == 'Hartree' else 1.0

    atoms = read_structure(args.input_file)
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

    atoms.calc = LiTENCalculator(model=model, model_name=args.model_name, is_pbc=is_pbc, is_batch=False, atoms=atoms, **config)

    print("For processing multiple molecules, we recommend using a loop, "
          "as the model's compilation and initialization are relatively slow.")

    print("Please wait a moment, model is on compiling...")
    opt = BFGS(atoms, maxstep=0.3)
    converged = opt.run(fmax=0.05, steps=500)

    if converged:
        write(args.output_file, atoms)
        print(f"[✓] Optimized molecule {os.path.basename(args.input_file)}")
    else:
        print(f"[X] Molecule did not converge in {os.path.basename(args.input_file)}")

# === 命令行参数 ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Opt with LiTEN-FF model using ASE.")

    parser.add_argument('--model_name', type=str, default='nablaDFT', help='Path to the saved model')
    parser.add_argument('--input_file', type=str, default=root_dir + '/example/dipe.xyz',
                        help='Input structure file')
    parser.add_argument('--output_file', type=str, default=root_dir + '/example/dipe_opt.xyz',
                        help='Output structure file')
    parser.add_argument('--dtype', type=str, choices=['float32', 'float64'], default='float32',
                        help='Torch float dtype')
    parser.add_argument('--unit_energy', type=str, default='Hartree', help='Energy unit (Hartree or 1.0)')
    parser.add_argument('--unit_force', type=str, default='Hartree', help='Force unit (Hartree or 1.0)')
    parser.add_argument('--maxstep', type=int, default=500, help='Run geometry optimization before MD')

    args = parser.parse_args()
    main(args)
