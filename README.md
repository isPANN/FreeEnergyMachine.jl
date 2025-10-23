# FreeEnergyMachine.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://isPANN.github.io/FreeEnergyMachine.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://isPANN.github.io/FreeEnergyMachine.jl/dev/)
[![Build Status](https://github.com/isPANN/FreeEnergyMachine.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/isPANN/FreeEnergyMachine.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/isPANN/FreeEnergyMachine.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/isPANN/FreeEnergyMachine.jl)

A Julia implementation of the [Free Energy Machine (FEM)](https://github.com/Fanerst/FEM) algorithm for solving combinatorial optimization problems with **GPU acceleration** support via CUDA.jl.

## Features

- **GPU Acceleration**: Full CUDA support for massive parallel optimization
- **Multiple Problems**: MaxCut, QUBO, and extensible to custom problems
- **Flexible Optimization**: Multiple annealing strategies and optimizers (Adam, RMSprop)
- **Efficient**: Manual gradient computation for better GPU performance
- **Easy to Use**: Simple API with automatic device management

## Installation

```julia
using Pkg
Pkg.add("FreeEnergyMachine")
```

## Quick Start

### CPU Example

```julia
using FreeEnergyMachine

# Create a MaxCut problem
coupling = Float64[
    0.0  1.0  1.0
    1.0  0.0  1.0
    1.0  1.0  0.0
]

problem = MaxCut(coupling, device="cpu")

# Configure solver
config = SolverConfig(
    betamin=0.01,
    betamax=0.5,
    annealing=InverseAnnealing(),
    optimizer=AdamOpt(0.1),
    manual_grad=true,
    device="cpu"
)

# Solve with 10 parallel trials and 1000 steps
solver = Solver(problem, 10, 1000; config=config)
p = fem_iterate(solver)

# Get results
E = infer(problem, p)
println("Best energy: ", maximum(E))
```

### GPU Example

```julia
using FreeEnergyMachine
using CUDA

# Check GPU availability
if CUDA.functional()
    # Create problem on GPU
    problem = MaxCut(coupling, device="cuda")
    
    # Configure for GPU
    config = SolverConfig(
        betamin=0.01,
        betamax=0.5,
        manual_grad=true,
        device="cuda"  # Use GPU
    )
    
    # Solve on GPU with 100 parallel trials
    solver = Solver(problem, 100, 1000; config=config)
    p = fem_iterate(solver)
    
    # Get results (automatically on GPU)
    E = infer(problem, p)
    println("Best energy: ", maximum(Array(E)))
end
```

## GPU Support

FreeEnergyMachine.jl fully supports NVIDIA GPUs via CUDA.jl:

- **Automatic device management**: Simply specify `device="cuda"` 
- **Efficient GPU kernels**: Optimized implementations for all operations
- **Large-scale problems**: Solve problems with thousands of variables
- **Massive parallelism**: Run hundreds of trials in parallel

See [docs/GPU_USAGE.md](docs/GPU_USAGE.md) for detailed GPU usage guide.

### Performance

For a MaxCut problem with 100 nodes and 100 parallel trials:
- **CPU**: ~24 seconds
- **GPU**: ~1.5 seconds  
- **Speedup**: ~16x

Performance scales with problem size and number of parallel trials.

## Supported Problems

- **MaxCut**: Maximum cut problem on weighted graphs
- **QUBO**: Quadratic Unconstrained Binary Optimization
- **Custom**: Extensible framework for custom problems

## Documentation

- [GPU Usage Guide](docs/GPU_USAGE.md) - Detailed guide for GPU acceleration
- [Examples](examples/) - Example scripts and notebooks
- See `examples/gpu_example.jl` for comprehensive GPU examples

## Requirements

- Julia ≥ 1.0
- For GPU support:
  - NVIDIA GPU with CUDA support (compute capability ≥ 3.5)
  - CUDA driver (CUDA 11.0+ recommended)
  - CUDA.jl (automatically installed)

## Citation


```bibtex
@software{FreeEnergyMachine.jl,
  author = {Xiwei Pan},
  title = {FreeEnergyMachine.jl},
  year = {2025},
  url = {https://github.com/isPANN/FreeEnergyMachine.jl}
}
```

For original paper, please refer to:
```bibtex
@article{shen2025free,
  title={Free-energy machine for combinatorial optimization},
  author={Shen, Zi-Song and Pan, Feng and Wang, Yao and Men, Yi-Ding and Xu, Wen-Biao and Yung, Man-Hong and Zhang, Pan},
  journal={Nature Computational Science},
  pages={1--11},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```

For official implementation in python, please refer to https://github.com/Fanerst/FEM


## Related Projects

- Python implementation: [FEM]([FEM/](https://github.com/Fanerst/FEM)) 
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) - GPU programming in Julia
