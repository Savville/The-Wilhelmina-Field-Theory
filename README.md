# The-Wilhelmina-Field-Theory

Python models and LTspice simulations validating the Wilhelmina Field Theory—a continuous spatial reaction-diffusion framework for phase noise in coupled oscillator networks.

## Overview
This repository contains the computational validation for the Wilhelmina Field Theory. We model phase noise in massive MIMO and phased-array systems as a spatial continuum, replacing discrete Leeson models and systems of ODEs with a Fokker-Planck derived reaction-diffusion Partial Differential Equation (PDE).

## Repository Contents
- `wilhelmina_validation.py`: The core Python script that implements the Crank-Nicolson numerical solver to solve the Wilhelmina $\Psi$ PDE and compares it against analytical predictions.
- `wilhelmina_array.asc`: The LTspice schematic of a 3-node Van der Pol oscillator array coupled via a shared resistive substrate, used to validate the spatial noise gradient.
- **Helper Scripts** (`process_results.py`, `generate_asc.py`, `run_automation.py`): Utilities used to extract data from LTspice `.raw` files and automate circuit generation.
- `LTSPICE_GUIDE.md`: A guide on how to run the LTspice simulations and extract the necessary phase noise data.

## Getting Started

### Prerequisites
You will need Python 3.8+ and standard scientific libraries:
```bash
pip install numpy matplotlib scipy
```

### Running the Python Validation
Simply run the main validation script to generate the regime maps and spatial noise gradients:
```bash
python wilhelmina_validation.py
```

### LTspice Simulation
Open `wilhelmina_array.asc` in LTspice and run the `.tran` command. See `LTSPICE_GUIDE.md` for detailed instructions on extracting the power spectrum and exporting the data back into Python.
