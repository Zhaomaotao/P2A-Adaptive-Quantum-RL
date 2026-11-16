# P2A-Adaptive-Quantum-RL

Python implementation of an adaptive quantum multi-agent reinforcement learning framework for coordinated energy management of power-to-ammonia (P2A)-based industrial multi-energy systems.

## Methods

The repository contains implementations of three methods:

- **MAPPO:** Multi-agent proximal policy optimization.
- **Non-Adaptive:** Quantum multi-agent PPO with fixed quantum actor configurations.
- **Proposed:** World-model-assisted quantum multi-agent reinforcement learning with Q-MAML-inspired parameter initialization and DQAS-inspired architecture selection.

## Repository Contents

- `energy_env.py`: Industrial multi-energy system simulation environment.
- `data_generator.py`: Simulation input data generation.
- `main_train_*.py`: Training implementations.
- `test_*.py`: Policy evaluation.
- `world_model.py`: Learned system dynamics model.
- `quantum_tuning.py` and `dqas_tuning.py`: Quantum policy configuration and adaptation.
- `energy_data.npz`: Input data used in the simulations.
- `results_compare/`: Evaluation results.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data

The repository includes the input arrays used in the numerical simulations. The data generation procedure is provided in `data_generator.py`. Simulation-generated evaluation results are stored separately from the input data.