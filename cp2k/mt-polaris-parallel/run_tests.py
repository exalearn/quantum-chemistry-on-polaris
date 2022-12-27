"""Compute the redox potentials for all example molecules"""
from concurrent.futures import as_completed
from argparse import ArgumentParser
from time import perf_counter
from shutil import rmtree
from io import StringIO
import os

from ase.db import connect
from ase.io import read
from ase import units
from ase import Atoms
from tqdm import tqdm
import pandas as pd

from parsl.addresses import address_by_hostname
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher, SimpleLauncher
from parsl.providers import PBSProProvider
from parsl.config import Config
from parsl import python_app
import parsl


def buffer_cell(atoms, buffer_size: float = 3.):
    """How to buffer the cell such that it has a vacuum layer around the side

    Args:
        atoms: Atoms to be centered
        buffer_size: Size of the buffer to place around the atoms
    """

    atoms.positions -= atoms.positions.min(axis=0)
    atoms.cell = atoms.positions.max(axis=0) + buffer_size * 2
    atoms.positions += atoms.cell.max(axis=0) / 2 - atoms.positions.mean(axis=0)
    
    
@python_app
def run_cp2k(atoms, xc, basis_set, cutoff, charge, num_nodes):
    """Run a CP2k relaxation"""
    from ase.calculators.cp2k import CP2K
    from ase.optimize import LBFGS
    from ase import units
    from time import perf_counter
    from shutil import rmtree
    from io import StringIO
    import os
    
    # Build the calculator. Has to happen on-node as it starts CP2kx3005c0s37b1n0
        # Make the calculator
    pp = {
        'BLYP': 'GTH-BLYP'
    }[xc]
    cp2k_opts = dict(
        xc=None,
        inp=f"""&FORCE_EVAL
&DFT
  &XC
     &XC_FUNCTIONAL {xc}
     &END XC_FUNCTIONAL
  &END XC
  &POISSON
     PERIODIC NONE
     PSOLVER MT
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
        basis_set=basis_set,
        pseudo_potential=pp,
        poisson_solver=None,
    )
    run_dir = f'run-{os.environ["PARSL_WORKER_POOL_ID"]}-{os.environ["PARSL_WORKER_RANK"]}'
    rmtree(run_dir, ignore_errors=True)
    os.mkdir(run_dir)
    calc = CP2K(cutoff=cutoff * units.Ry, max_scf=10,
                directory=run_dir,
                command=f'mpiexec -n {num_nodes * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                        '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                        '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
                **cp2k_opts)
    
    # If not, run an optimization
    start_time = perf_counter()
    calc.set(charge=charge, uks=charge != 0)
    atoms.set_calculator(calc)
    dyn = LBFGS(atoms, logfile=f'{run_dir}/opt.log')
    dyn.run(fmax=0.04)

    # Store the single point
    calc.get_forces()
    
    # Convert it to JSON for tranmission
    out = StringIO()
    atoms.write(out, 'json')
    atoms_msg = out.getvalue()
    
    # Force the calculator to die
    calc.__del__()
    
    return perf_counter() - start_time, atoms_msg


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--basis-set', default='AUG-DZVP-GTH')
    parser.add_argument('--cutoff', default=350, type=int, help='Cutoff for the gird')
    parser.add_argument('--buffer', default=6, type=float, help='Amount of vacuum around the molecule')
    parser.add_argument('--xc', choices=['BLYP'], default='BLYP', help='XC functional')
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes per calculation')
    

    args = parser.parse_args()
    
    # Make the parsl configuration
    config = Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                #available_accelerators=4,  # Ensures one worker per accelerator
                address=address_by_hostname(),
                #cpu_affinity="alternating",  # Prevents thread contention
                prefetch_capacity=0,  # Increase if you have many more tasks than workers
                start_method="fork",  # Needed to avoid interactions between MPI and os.fork
                max_workers=1,
                provider=PBSProProvider(
                    account="CSC249ADCD08",
                    worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd {os.getcwd()}
hostname
pwd
conda activate /lus/grand/projects/CSC249ADCD08/quantum-chemistry-on-polaris/env""",
                    walltime="1:00:00",
                    queue="debug",
                    scheduler_options="#PBS -l filesystems=home:eagle:grand",
                    launcher=SimpleLauncher(),
    #                 launcher=MpiExecLauncher(
    #                     bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
    #                 ),  # Ensures 1 manger per node and allows it to divide work to all 64 cores
                    select_options="ngpus=4",
                    nodes_per_block=args.num_nodes,
                    min_blocks=0,
                    max_blocks=1,
                    cpus_per_node=64,
                ),
            ),
        ]
    )
    parsl.load(config)

    # Read in the example data
    data = pd.read_csv('../../data/example_molecules.csv')
    data['atoms'] = data['xyz_neutral'].apply(StringIO).apply(lambda x: read(x, format='xyz'))

    # Run the relaxations for both the neutral and the charged geometry
    settings = args.__dict__.copy()
    futures = []
    for _, row in data.iterrows():  # Loop over each state of each molecule
        for state, charge in zip(['neutral', 'oxidized', 'reduced'], [0, 1, -1]):
            # Check if it has been done
            with connect('data.db') as db:
                if db.count(inchi_key=row['inchi_key'], state=state, **settings) > 0:
                    continue
            
            # Buffer the cell
            atoms: Atoms = row['atoms'].copy()
            buffer_cell(atoms, buffer_size=args.buffer)
            
            # Submit the calculation
            future = run_cp2k(atoms, args.xc, args.basis_set, args.cutoff, charge, args.num_nodes)
            
            # Add some metadata to the future
            future.info = {'inchi_key': row['inchi_key'], 'state': state}
            futures.append(future)
            
    # Save them as they complete
    for future in tqdm(as_completed(futures), total=len(futures)):
        if future.exception() is not None:
            print(future.exception())
        runtime, atoms_msg = future.result()
        out = StringIO(atoms_msg)
        atoms = read(out, format='json')
        with connect('data.db') as db:
            db.write(atoms, runtime=runtime, **settings, **future.info)
