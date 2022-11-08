"""Compute the redox potentials for all example molecules"""
from argparse import ArgumentParser
from io import StringIO
from time import perf_counter

from ase.calculators.nwchem import NWChem
from ase.optimize import LBFGS
from ase.db import connect
from ase.io import read
from ase import Atoms
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--basis-set', default='aug-cc-pvdz')
    parser.add_argument('--xc', choices=['B3LYP'], default='B3LYP', help='XC functional')

    args = parser.parse_args()

    # Make the calculator
    calc = NWChem(
        command='mpirun -n 12 nwchem PREFIX.nwi > PREFIX.nwo 2> /dev/null',
        xc=args.xc, basis=args.basis_set
    )

    # Read in the example data
    data = pd.read_csv('../data/example_molecules.csv')
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
            calc.set()
            calc.set(charge=charge, dft={'mult': 1 if charge == 0 else 2, 'iterations': 100})
            atoms: Atoms = row['atoms'].copy()
            atoms.set_calculator(calc)
            dyn = LBFGS(atoms)
            dyn.run(fmax=0.04)

            # Store the single point
            calc.get_forces()
            with connect('data.db') as db:
                db.write(atoms, inchi_key=row['inchi_key'], state=state,
                         runtime=perf_counter() - start_time, **settings)
            tqdm.update(1)
