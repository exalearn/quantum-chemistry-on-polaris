# Quantum Chemistry on Polaris

Our workhorse code, NWChem, does not support DFT on GPUs and its accelerator-friendly successor (NWChemEx) is not available broadly yet.
This is a problem because most modern supercomputers (e.g., Polaris) use a GPU on most nodes.
The goal of this repo to test alternative DFT codes that could bridge us until NWChemEx is available.

## Installation

Running the scripts requires ASE and a few other packages, which are listed in the environment file. 
Installing the codes is a little outside of our scope, but I'll add brief notes where applicable.
Unless otherwise listed, I use the Ubuntu package versions.
