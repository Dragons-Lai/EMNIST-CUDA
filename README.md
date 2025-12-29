# EMNIST in CUDA
## Purpose

This project implements a simple 2-layer MLP (Multi-Layer Perceptron) for **EMNIST Balanced** character classification, progressively optimizing from high-level PyTorch to low-level CUDA implementations. Each version demonstrates different optimization techniques and trade-offs between performance and complexity.

- **Architecture:** 784 → 256 → 47 (input → hidden → output)
- **Dataset:** EMNIST Balanced - 10,000 training samples, 18,800 test samples, 47 classes
- **Batch Size:** 32, **Epochs:** 10
- **Activation:** ReLU, **Loss:** Cross-entropy, **Optimizer:** SGD (lr=0.01)

## Setup

```bash
git clone <repository-url>
cd EMNIST-CUDA
pip install -r requirements.txt
```

### CUDA Setup

For CUDA versions (v4-v5), ensure you have NVIDIA CUDA toolkit installed:

```bash
# Check CUDA installation
nvcc --version

# Check your GPU's compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

## Version Progression

### v1-pytorch.py - PyTorch Baseline
- **Framework:** PyTorch with CUDA tensors
- **Features:** High-level PyTorch operations (Linear, ReLU, CrossEntropyLoss), GPU tensors with automatic memory management
- **Purpose:** Establishes baseline performance and correctness reference

### v2-numpy.py - NumPy Implementation
- **Framework:** Pure NumPy (CPU-only)
- **Features:** Manual forward/backward pass implementation, custom gradient computation and weight updates, He initialization
- **Purpose:** Demonstrates the underlying math without GPU acceleration

### v3-c.c - C/CPU Implementation
- **Framework:** Pure C with timing breakdown
- **Features:** Manual memory management, detailed timing instrumentation per operation
- **Purpose:** Shows CPU performance baseline and prepares for GPU porting

### v4-cuda.cu - Naive CUDA Kernels
- **Framework:** CUDA C with custom kernels
- **Features:** Custom matrix multiplication kernels (naive O(n³) GEMM), element-wise operations (ReLU, bias, softmax) on GPU, manual memory transfers between host and device
- **Purpose:** First GPU implementation with basic CUDA kernels

### v5-tiled-cuda.cu - Tiled CUDA Kernels
- **Framework:** CUDA C with shared memory optimization
- **Features:** Tiled matrix multiplication using shared memory (16x16 tiles), reduced global memory access, improved memory coalescing
- **Purpose:** Demonstrates shared memory optimization for GEMM operations

## Usage

```bash
# Run each version
python3 v1-pytorch.py                                    # PyTorch baseline
python3 v2-numpy.py                                      # NumPy CPU implementation
gcc -O3 -o v3 v3-c.c -lm && ./v3                        # C CPU implementation
nvcc -arch=native -O3 -o v4 v4-cuda.cu && ./v4          # Naive CUDA kernels
nvcc -arch=native -O3 -o v5 v5-tiled-cuda.cu && ./v5    # Tiled CUDA kernels
```

> **Note:** The `-arch=native` flag ensures kernels are compiled for your specific GPU. If not supported, specify your architecture explicitly (e.g., `-arch=sm_86` for Ampere GPUs).

## Performance Comparison

Measured on **RTX 3090** (Ampere architecture, sm_86), CUDA 12.6:

| Version | Implementation | Time | Speedup vs v3 | Final Loss | Test Accuracy |
|---------|---------------|------|---------------|------------|---------------|
| v1.py | PyTorch CUDA | 2.1s | ~9.1x | 0.885 | 67.98% |
| v2.py | NumPy CPU | 11.0s | ~1.7x | 0.889 | 68.10% |
| v3.c | C CPU | 19.2s | 1x (baseline) | 0.875 | 68.05% |
| v4.cu | Naive CUDA | 0.8s | ~24x | 0.882 | 68.19% |
| v5.cu | Tiled CUDA | 0.7s | ~27x | 0.879 | 68.15% |

## Timing Breakdown Analysis

### v4 (Naive CUDA)
```
Total: 0.8s
  Forward Pass:      45.0%
  Backward Pass:     29.0%
  Data Transfer:      9.2%
  Param Update:       5.0%
  Loss Compute:       0.4%
```

### v5 (Tiled CUDA)
```
Total: 0.7s
  Forward Pass:      29.6%
  Backward Pass:     28.7%
  Data Transfer:     21.1%
  Param Update:       6.2%
  Loss Compute:       0.5%
```

## Performance Insights

### Key Observations

1. **CPU vs GPU:** CUDA implementations (v4, v5) achieve **~24-27x speedup** over C implementation
2. **NumPy vs C:** NumPy (v2) is **~1.7x faster** than pure C (v3) due to BLAS-optimized matrix operations
3. **PyTorch Overhead:** PyTorch (v1) is **~3x slower** than custom CUDA kernels due to framework overhead, but provides the best developer experience
4. **Tiled GEMM:** v5 shows **~36% faster forward pass** compared to v4 (29.6% vs 45.0%) due to shared memory optimization

## Key CUDA Concepts Demonstrated

| Concept | v4 (Naive) | v5 (Tiled) |
|---------|-----------|------------|
| Global Memory Access | Direct | Coalesced via tiles |
| Shared Memory | Not used | 16x16 tile caching |
| Thread Synchronization | Minimal | __syncthreads() for tiles |
| Memory Bandwidth | Low efficiency | Improved reuse |

## File Structure

```
EMNIST-CUDA/
├── data/
│   ├── X_train.bin          # Training images (112,800 × 784)
│   ├── y_train.bin          # Training labels
│   ├── X_test.bin           # Test images (18,800 × 784)
│   └── y_test.bin           # Test labels
├── v1-pytorch.py            # PyTorch CUDA implementation
├── v2-numpy.py              # NumPy CPU implementation
├── v3-c.c                   # Pure C CPU implementation
├── v4-cuda.cu               # Naive CUDA kernels
├── v5-tiled-cuda.cu         # Tiled CUDA with shared memory
├── data_downloader.py       # EMNIST dataset downloader
├── requirements.txt         # Python dependencies
└── README.md
```

## References

- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Tiled Matrix Multiplication](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
