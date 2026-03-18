"""Simple benchmark for the PyTorch K-Means reference implementation."""

from __future__ import annotations

import argparse
import time
from typing import Dict

import torch

from kmeans_reference import (
    init_centroids_from_keys,
    run_kmeans_naive,
    run_kmeans_vectorized,
)


def _is_monotonic_nonincreasing(values: list[float]) -> bool:
    return all(curr <= prev for prev, curr in zip(values, values[1:]))


def _format_counts_per_iter(counts_per_iter: object) -> str:
    formatted = []
    for iter_idx, counts in enumerate(counts_per_iter, start=1):
        formatted.append(f"{iter_idx}:{counts.detach().cpu().tolist()}")
    return " ".join(formatted) if formatted else "[]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-keys", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_run(fn, device: torch.device) -> tuple[Dict[str, object], float]:
    _synchronize_if_needed(device)
    start = time.perf_counter()
    result = fn()
    _synchronize_if_needed(device)
    elapsed_s = time.perf_counter() - start
    return result, elapsed_s


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    keys = torch.randn(args.num_keys, args.dim, device=device, dtype=dtype)
    # Keep the initialization fixed across both paths so the comparison only
    # measures implementation differences, not random restarts.
    init_centroids = init_centroids_from_keys(
        keys, num_clusters=args.num_clusters, seed=args.seed
    )

    naive_result, naive_time_s = _time_run(
        lambda: run_kmeans_naive(keys, init_centroids.clone(), args.num_iters), device
    )
    vectorized_result, vectorized_time_s = _time_run(
        lambda: run_kmeans_vectorized(keys, init_centroids.clone(), args.num_iters),
        device,
    )

    centroid_diff = (
        naive_result["centroids"] - vectorized_result["centroids"]
    ).abs()
    assignment_mismatch_count = int(
        (naive_result["assignments"] != vectorized_result["assignments"])
        .sum()
        .item()
    )
    assignment_mismatch_rate = assignment_mismatch_count / args.num_keys
    naive_final_objective = naive_result["distances"].sum().item()
    vectorized_final_objective = vectorized_result["distances"].sum().item()

    print("Configuration")
    print(f"  num_keys={args.num_keys}")
    print(f"  dim={args.dim}")
    print(f"  num_clusters={args.num_clusters}")
    print(f"  num_iters={args.num_iters}")
    print(f"  dtype_arg={args.dtype}")
    print(f"  input_dtype={keys.dtype}")
    print(f"  device={device}")
    print(f"  seed={args.seed}")
    print(
        "  note=BF16 inputs use FP32 internal accumulation in this reference path"
    )
    print()

    print("Naive")
    print(f"  runtime_s={naive_time_s:.6f}")
    print(f"  final_objective={naive_final_objective:.6f}")
    print(f"  output_centroid_dtype={naive_result['centroids'].dtype}")
    print(f"  monotonic_nonincreasing={_is_monotonic_nonincreasing(naive_result['objectives_per_iter'])}")
    print(f"  objectives_per_iter={naive_result['objectives_per_iter']}")
    print(
        f"  cluster_counts_per_iter={_format_counts_per_iter(naive_result['counts_per_iter'])}"
    )
    print()

    print("Vectorized")
    print(f"  runtime_s={vectorized_time_s:.6f}")
    print(f"  final_objective={vectorized_final_objective:.6f}")
    print(f"  output_centroid_dtype={vectorized_result['centroids'].dtype}")
    print(
        f"  monotonic_nonincreasing={_is_monotonic_nonincreasing(vectorized_result['objectives_per_iter'])}"
    )
    print(f"  objectives_per_iter={vectorized_result['objectives_per_iter']}")
    print(
        f"  cluster_counts_per_iter={_format_counts_per_iter(vectorized_result['counts_per_iter'])}"
    )
    print()

    print("Comparison")
    print(f"  centroid_abs_diff_mean={centroid_diff.mean().item():.6e}")
    print(f"  centroid_abs_diff_max={centroid_diff.max().item():.6e}")
    print(f"  centroid_l2_diff={torch.linalg.norm(centroid_diff).item():.6e}")
    print(f"  assignment_mismatch_count={assignment_mismatch_count}")
    print(f"  assignment_mismatch_rate={assignment_mismatch_rate:.6f}")
    print(
        f"  cluster_count_histogram={vectorized_result['counts'].detach().cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
