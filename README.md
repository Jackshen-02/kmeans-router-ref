# PyTorch K-Means Reference for IVF-Style Sparse Attention Routing

This project is a small forward-only PyTorch reference for K-Means clustering of key vectors with shape `[N, D]` into `C` clusters. It is specifically a correctness reference for IVF-style sparse attention routing, not an optimized production kernel.

The current milestone covers only key clustering in the forward pass. It is meant to be easy to explain, easy to validate against, and stable enough to serve as the baseline for later Triton/CUDA work.

The implementation includes:

- deterministic centroid initialization from keys
- naive and vectorized assignment/update paths
- multi-iteration K-Means
- cluster-based key reordering so same-cluster keys become contiguous
- validation helpers for reordering metadata
- pytest-style correctness tests
- a simple CPU/GPU benchmark script for validation and debugging

## Numerical behavior

Supported input dtypes:

- `torch.float32`
- `torch.bfloat16`

For BF16-friendly correctness, the reference uses FP32 internally for:

- squared Euclidean distance computation
- centroid sum accumulation
- centroid count / mean updates

Updated centroids are returned in FP32. This is intentional: the goal here is numerical clarity and a stable reference path, not maximal throughput.

## Milestone scope

- current milestone: forward-only key clustering reference
- next milestone: larger mock GPU workloads, then Triton kernel work
- current benchmark: useful for sanity checking and regression detection, not final performance reporting

## Files

- `kmeans_reference.py`: reference implementation
- `test_kmeans_reference.py`: pytest-style correctness tests
- `benchmark_kmeans.py`: simple benchmark for CPU or CUDA
- `progress_update.md`: short team-meeting update draft

## Run tests

```bash
python -m pytest test_kmeans_reference.py
```

## Run benchmarks

Small CPU run:

```bash
python benchmark_kmeans.py --num-keys 512 --dim 32 --num-clusters 8 --num-iters 3 --dtype fp32 --device cpu --seed 0
```

Small BF16 CPU run:

```bash
python benchmark_kmeans.py --num-keys 512 --dim 32 --num-clusters 8 --num-iters 3 --dtype bf16 --device cpu --seed 0
```

Medium CUDA run:

```bash
python benchmark_kmeans.py --num-keys 8192 --dim 64 --num-clusters 32 --num-iters 5 --dtype bf16 --device cuda --seed 0
```

## Notes

- Squared Euclidean distance is used throughout.
- The naive path is intentionally slow and explicit so it can serve as a debugging baseline.
- The vectorized path is still written for readability rather than peak speed.
- Empty clusters are handled deterministically by keeping the previous centroid when one is provided.
