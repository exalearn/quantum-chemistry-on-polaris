"""Compute the redox potentials for all example molecules"""
from argparse import ArgumentParser
from io import StringIO
from shutil import rmtree
from time import perf_counter

from ase.calculators.cp2k import CP2K
from ase.optimize import LBFGS
from ase.db import connect
from ase.io import read
from ase import units
from ase import Atoms
from tqdm import tqdm
import pandas as pd


def buffer_cell(atoms, buffer_size: float = 3.):
    """How to buffer the cell such that it has a vacuum layer around the side

    Args:
        atoms: Atoms to be centered
        buffer_size: Size of the buffer to place around the atoms
    """

    atoms.positions -= atoms.positions.min(axis=0)
    atoms.cell = [atoms.positions.max() + buffer_size * 2] * 3
    atoms.positions += atoms.cell.max(axis=0) / 2 - atoms.positions.mean(axis=0)


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--basis-set', default='AUG-DZVP-GTH')
    parser.add_argument('--cutoff', default=500, type=int, help='Cutoff for the gird')
    parser.add_argument('--buffer', default=6, type=float, help='Amount of vacuum around the molecule')
    parser.add_argument('--xc', choices=['BLYP'], default='BLYP', help='XC functional')

    args = parser.parse_args()

    # Make the calculator
    pp = {
        'BLYP': 'GTH-BLYP'
    }[args.xc]

    cp2k_opts = dict(
        xc=None,
        inp=f"""&FORCE_EVAL
&DFT
  &XC
     &XC_FUNCTIONAL {args.xc}
     &END XC_FUNCTIONAL
  &END XC
  &POISSON
     PERIODIC NONE
     PSOLVER WAVELET
  &END POISSON
  &SCF
    &OUTER_SCF
     MAX_SCF 5
    &END OUTER_SCF
    &OT T
      PRECONDITIONER FULL_ALL
    &END OT
  &END SCF
&END DFT
&SUBSYS
  &TOPOLOGY
    &CENTER_COORDINATES
    &END
  &END
&END FORCE_EVAL
""",
        basis_set_file='GTH_BASIS_SETS',
        basis_set=args.basis_set,
        pseudo_potential=pp,
        poisson_solver=None,
        stress_tensor=False
    )
    rmtree('run', ignore_errors=True)
    calc = CP2K(cutoff=args.cutoff * units.Ry, directory='run', max_scf=10, **cp2k_opts)

    # Read in the example data
    data = pd.read_csv('../../data/example_molecules.csv')
    data['atoms'] = data['xyz_neutral'].apply(StringIO).apply(lambda x: read(x, format='xyz'))

    # Run the relaxations for both the neutral and the charged geometry
    settings = args.__dict__.copy()
    tqdm = tqdm(total=len(data) * 3)
    for _, row in data.iterrows():  # Loop over each state of each molecule
        for state, charge in zip(['neutral', 'oxidized', 'reduced'], [0, 1, -1]):
            tqdm.set_description(f'inchi={row["inchi_key"]} state={state}')

            # Check if it has been done
            with connect('data.db') as db:
                if db.count(inchi_key=row['inchi_key'], state=state, **settings) > 0:
                    tqdm.update(1)
                    continue

            # If not, run an optimization
            start_time = perf_counter()
            calc.set(charge=charge, uks=charge != 0)
            atoms: Atoms = row['atoms'].copy()
            buffer_cell(atoms, buffer_size=args.buffer)
            atoms.set_calculator(calc)
            dyn = LBFGS(atoms)
            dyn.run(fmax=0.04)

            # Store the single point
            calc.get_forces()
            with connect('data.db') as db:
                db.write(atoms, inchi_key=row['inchi_key'], state=state,
                         runtime=perf_counter() - start_time,  **settings)
            tqdm.update(1)
