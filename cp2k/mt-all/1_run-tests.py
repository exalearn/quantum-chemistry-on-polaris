"""Compute the redox potentials for all example molecules"""
from argparse import ArgumentParser
from time import perf_counter
from shutil import rmtree
from pathlib import Path
from io import StringIO

from ase.calculators.cp2k import CP2K
from ase.optimize import LBFGS
from ase.db import connect
from ase.io import read
from ase import units
from ase import Atoms
from tqdm import tqdm
import pandas as pd

_hfx_fraction = {
    'HYB_GGA_XC_B3LYP': 0.2,
    'HYB_GGA_XC_WB97X_D3': 1.0
}


def buffer_cell(atoms, buffer_size: float = 3.):
    """How to buffer the cell such that it has a vacuum layer around the side

    Args:
        atoms: Atoms to be centered
        buffer_size: Size of the buffer to place around the atoms
    """

    atoms.positions -= atoms.positions.min(axis=0)
    atoms.cell = atoms.positions.max(axis=0) + buffer_size * 2
    atoms.positions += atoms.cell.max(axis=0) / 2 - atoms.positions.mean(axis=0)



if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--basis-set', default='def2-TZVPD')
    parser.add_argument('--basis-set-file', default='../basis-sets/DEF2_BASIS_SETS')
    parser.add_argument('--cutoff', default=350, type=int, help='Cutoff for the grid')
    parser.add_argument('--rel-cutoff', default=60, type=int, help='Relative cutoff for multigrid.')
    parser.add_argument('--buffer', default=6, type=float, help='Amount of vacuum around the molecule')
    parser.add_argument('--xc', default='HYB_GGA_XC_B3LYP', help='XC functional')

    args = parser.parse_args()

    # Make the calculator
    b3lyp_other = """&HF
    &SCREENING
        EPS_SCHWARZ 1.0E-10 
    &END
    &MEMORY
        MAX_MEMORY  1024
    &END
    FRACTION 0.2
&END HF
"""
    wb97_v_other = """ &HF
    FRACTION 1.000
    &INTERACTION_POTENTIAL
        OMEGA 0.30
        POTENTIAL_TYPE MIX_CL
        SCALE_COULOMB 0.167
        SCALE_LONGRANGE 0.833
    &END INTERACTION_POTENTIAL
    &MEMORY
        MAX_MEMORY 1024
    &END MEMORY
    &SCREENING
        EPS_SCHWARZ 1.0E-10
    &END SCREENING
&END HF
&VDW_POTENTIAL
    DISPERSION_FUNCTIONAL NON_LOCAL
        &NON_LOCAL
        CUTOFF 40
        KERNEL_FILE_NAME rVV10_kernel_table.dat
        PARAMETERS 6.3 0.0093
        SCALE 1.0
        TYPE RVV10
        VERBOSE_OUTPUT
    &END NON_LOCAL
&END VDW_POTENTIAL
"""
    wb97_d3_other = """ &HF
    FRACTION 1.000
    &INTERACTION_POTENTIAL
        OMEGA 0.25
        POTENTIAL_TYPE MIX_CL
        SCALE_COULOMB 0.195728
        SCALE_LONGRANGE 0.804272
    &END INTERACTION_POTENTIAL
    &MEMORY
        MAX_MEMORY 1024
    &END MEMORY
    &SCREENING
        EPS_SCHWARZ 1.0E-10
    &END SCREENING
&END HF
&VDW_POTENTIAL
    DISPERSION_FUNCTIONAL PAIR_POTENTIAL
    &PAIR_POTENTIAL
        TYPE DFTD3
        PARAMETER_FILE_NAME ./dftd3.dat
        REFERENCE_FUNCTIONAL B97-D
    &END PAIR_POTENTIAL
&END VDW_POTENTIAL"""
    other_xc_parts = {
        'HYB_GGA_XC_B3LYP': b3lyp_other,
        'HYB_GGA_XC_WB97X_V': wb97_v_other,
        'HYB_GGA_XC_WB97X_D3': wb97_d3_other,
    }[args.xc]
    #   OT and an outer SCF loop seemed to be the key for getting this to converge properly
    cp2k_opts = dict(
        inp=f"""&FORCE_EVAL
&DFT
  &XC
    {other_xc_parts}
    &XC_FUNCTIONAL
        &{args.xc}
        &END {args.xc}
    &END XC_FUNCTIONAL
  &END XC
  &POISSON
     PERIODIC NONE
     PSOLVER MT
  &END POISSON
  &SCF
    IGNORE_CONVERGENCE_FAILURE
    &OUTER_SCF
      MAX_SCF 5
    &END OUTER_SCF
    &OT T
      PRECONDITIONER FULL_ALL
    &END OT
  &END SCF
  &QS
     METHOD GAPW
  &END QS
  &MGRID
    REL_CUTOFF [Ry] {args.rel_cutoff}
    COMMENSURATE TRUE
    NGRIDS 5
  &END MGRID
&END DFT
&END FORCE_EVAL
""",
        basis_set_file=args.basis_set_file,
        basis_set=args.basis_set,
        pseudo_potential='ALL',
        potential_file='ALL_POTENTIALS',
        xc=None,
        poisson_solver=None,
        stress_tensor=False,
    )
    rmtree('run', ignore_errors=True)
    calc = CP2K(
        cutoff=args.cutoff * units.Ry,
        max_scf=64,
        directory='run',
        command='/home/lward/Software/cp2k-2024.1/exe/local_cuda/cp2k_shell.ssmp',
        **cp2k_opts
    )

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
            atoms.calc = calc
            with calc:
                dyn = LBFGS(atoms)
                dyn.run(fmax=0.04)

                # Store the single point
                calc.get_forces()
                with connect('data.db') as db:
                    db.write(atoms, inchi_key=row['inchi_key'], state=state,
                             runtime=perf_counter() - start_time,  **settings)
            tqdm.update(1)
