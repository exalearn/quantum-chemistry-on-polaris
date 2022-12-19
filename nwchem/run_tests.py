"""Compute the redox potentials for all example molecules"""
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path
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
    parser.add_argument('--pre-basis-set', default='none', help='Basis set used to get initial guess')
    parser.add_argument('--pre-relax-threshold', type=float, default='none', help='Force threshold to "pre-relax" with pre-basis set')
    parser.add_argument('--xc', choices=['B3LYP'], default='B3LYP', help='XC functional')

    args = parser.parse_args()

    # Make the calculator
    calc = NWChem(
        command='mpirun -n 12 nwchem PREFIX.nwi > PREFIX.nwo 2> /dev/null',
        xc=args.xc, basis=args.basis_set, set={'lindep:n_dep': 0}, dft={'iterations': 1000}
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

            # Get the atoms
            start_time = perf_counter()
            wfc_file = Path() / 'wfc.guess.movecs'
            atoms: Atoms = row['atoms'].copy()
            atoms.set_calculator(calc)

            # Update the charges
            calc.parameters['dft']['vectors'] = {'output': str(wfc_file.absolute())}
            calc.parameters['charge'] = charge
            mult = 1 if charge == 0 else 2
            calc.parameters['dft']['mult'] = mult

            # If there are pre-basis-steps, relax with them first
            guess_start_time = perf_counter()
            if args.pre_relax_threshold != "none":
                # Run an energy convergence with the small basis set
                calc.parameters['basis'] = args.pre_basis_set
                atoms.get_potential_energy()

                # Let it use that wavefunction as a starting guess
                calc.parameters['dft']['vectors']['input'] = str(wfc_file.absolute())

                # Run the expensive relaxation
                dyn = LBFGS(atoms)
                dyn.run(fmax=float(args.pre_relax_threshold))
                calc.parameters['basis'] = args.basis_set
                calc.reset()

            # Run an energy computation to converge the wfcs
            if args.pre_basis_set != "none":
                calc.parameters['pretasks'] = [
                    # Start with a small basis set
                    {'theory': 'dft', 'basis': args.pre_basis_set, 'dft': {'mult': mult, 'iterations': 1000}},
                ]

            atoms.set_calculator(calc)
            atoms.get_potential_energy()
            guess_runtime = perf_counter() - guess_start_time

            # Now that the wavefunction's converged, run the optimization using it as a starting guess
            calc.parameters.pop('pretasks', None)  # Only use the large basis set now that we're converged
            calc.parameters['dft']['vectors']['input'] = str(wfc_file.absolute())
            dyn = LBFGS(atoms)
            dyn.run(fmax=0.04)

            # Store the single point
            calc.get_forces()
            with connect('data.db') as db:
                db.write(atoms, inchi_key=row['inchi_key'], state=state,
                         runtime=perf_counter() - start_time,
                         guess_runtime=guess_runtime,
                         **settings)
            tqdm.update(1)
