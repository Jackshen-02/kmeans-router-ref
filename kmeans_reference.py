"""Forward-only PyTorch reference for K-Means clustering of key vectors.

This module is intentionally simple and correctness-oriented. It is meant to
serve as a readable reference for IVF-style sparse attention routing and future
Triton/CUDA kernels rather than a maximally optimized implementation.

Numerical behavior:
- Inputs may be ``torch.float32`` or ``torch.bfloat16``.
- Distance computations and centroid updates use FP32 internally for stability.
- This keeps the reference numerically predictable even when BF16 is used for
  the key storage format.
- Updated centroids are returned in FP32, even when the input keys are BF16.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


Tensor = torch.Tensor


def _check_keys_2d(keys: Tensor) -> None:
    if keys.ndim != 2:
        raise ValueError(f"Expected keys to have shape [N, D], got {tuple(keys.shape)}")


def _check_centroids_2d(centroids: Tensor) -> None:
    if centroids.ndim != 2:
        raise ValueError(
            f"Expected centroids to have shape [C, D], got {tuple(centroids.shape)}"
        )


def _check_matching_feature_dim(keys: Tensor, centroids: Tensor) -> None:
    if keys.shape[1] != centroids.shape[1]:
        raise ValueError(
            "Keys and centroids must have the same feature dimension, "
            f"got {keys.shape[1]} and {centroids.shape[1]}"
        )


def init_centroids_from_keys(keys: Tensor, num_clusters: int, seed: int = 0) -> Tensor:
    """Initialize centroids by selecting keys from the input.

    The selection is deterministic for a fixed ``seed``. The returned centroids
    preserve the input dtype. Later update steps may promote centroids to FP32
    for numerical stability. Using a fixed initialization is also important when
    comparing the naive and vectorized paths: both implementations should start
    from exactly the same centroids so any mismatch is due to the K-Means
    implementation itself rather than different seeds.

    Args:
        keys: Input key matrix with shape ``[N, D]``.
        num_clusters: Number of centroids to sample.
        seed: Seed used for deterministic sampling.

    Returns:
        Tensor with shape ``[C, D]`` where ``C == num_clusters``.
    """

    _check_keys_2d(keys)

    num_keys = keys.shape[0]
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    if num_clusters > num_keys:
        raise ValueError(
            f"Cannot sample {num_clusters} centroids from only {num_keys} keys"
        )

    generator = torch.Generator(device=keys.device)
    generator.manual_seed(seed)
    indices = torch.randperm(num_keys, generator=generator, device=keys.device)[
        :num_clusters
    ]
    return keys.index_select(0, indices).clone()


def assign_clusters_naive(keys: Tensor, centroids: Tensor) -> Tuple[Tensor, Tensor]:
    """Assign each key to its nearest centroid using squared Euclidean distance.

    This is a slow and very explicit baseline intended for debugging and
    correctness checks. Distances are computed in FP32 even when the inputs are
    BF16 so the reference remains stable and easy to compare against lower-level
    kernels later.

    Args:
        keys: Key matrix of shape ``[N, D]``.
        centroids: Centroid matrix of shape ``[C, D]``.

    Returns:
        assignments: ``torch.long`` tensor of shape ``[N]``.
        distances: FP32 tensor of shape ``[N]`` containing the squared distance
            to the assigned centroid.
    """

    _check_keys_2d(keys)
    _check_centroids_2d(centroids)
    _check_matching_feature_dim(keys, centroids)

    num_keys = keys.shape[0]
    num_clusters = centroids.shape[0]
    keys_f = keys.float()
    centroids_f = centroids.float()

    assignments = torch.empty(num_keys, dtype=torch.long, device=keys.device)
    distances = torch.empty(num_keys, dtype=torch.float32, device=keys.device)

    for key_idx in range(num_keys):
        key = keys_f[key_idx]
        best_cluster = 0
        best_distance = torch.tensor(float("inf"), device=keys.device)
        for cluster_idx in range(num_clusters):
            diff = key - centroids_f[cluster_idx]
            distance = torch.sum(diff * diff)
            # Strict '<' keeps the first centroid on ties, matching torch.argmin.
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster_idx

        assignments[key_idx] = best_cluster
        distances[key_idx] = best_distance

    return assignments, distances


def assign_clusters_vectorized(keys: Tensor, centroids: Tensor) -> Tuple[Tensor, Tensor]:
    """Vectorized nearest-centroid assignment using squared Euclidean distance.

    The computation is intentionally direct and readable:
    - expand ``keys`` to ``[N, 1, D]``
    - expand ``centroids`` to ``[1, C, D]``
    - compute all pairwise squared distances in FP32

    Args:
        keys: Key matrix of shape ``[N, D]``.
        centroids: Centroid matrix of shape ``[C, D]``.

    Returns:
        assignments: ``torch.long`` tensor of shape ``[N]``.
        distances: FP32 tensor of shape ``[N]`` containing the squared distance
            to the assigned centroid.
    """

    _check_keys_2d(keys)
    _check_centroids_2d(centroids)
    _check_matching_feature_dim(keys, centroids)

    keys_f = keys.float()
    centroids_f = centroids.float()

    # pairwise_distances[n, c] = ||keys[n] - centroids[c]||_2^2
    pairwise_diffs = keys_f[:, None, :] - centroids_f[None, :, :]
    pairwise_distances = torch.sum(pairwise_diffs * pairwise_diffs, dim=-1)

    assignments = torch.argmin(pairwise_distances, dim=1)
    distances = pairwise_distances.gather(1, assignments[:, None]).squeeze(1)
    return assignments, distances


def update_centroids_naive(
    keys: Tensor,
    assignments: Tensor,
    num_clusters: int,
    prev_centroids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Update centroids from assigned keys using explicit loops.

    Centroid sums, counts, and means are accumulated in FP32 for numerical
    stability. This is deliberate even for BF16 inputs: centroid updates are a
    reduction, so FP32 is a cleaner correctness reference. Empty clusters are
    handled deterministically:
    - if ``prev_centroids`` is provided, keep the previous centroid
    - otherwise raise a ``ValueError``

    Args:
        keys: Key matrix of shape ``[N, D]``.
        assignments: Cluster indices of shape ``[N]``.
        num_clusters: Total number of clusters ``C``.
        prev_centroids: Optional previous centroids with shape ``[C, D]``.

    Returns:
        new_centroids: FP32 tensor of shape ``[C, D]``.
        counts: ``torch.long`` tensor of shape ``[C]``.
    """

    _check_keys_2d(keys)
    if assignments.ndim != 1 or assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "assignments must have shape [N] matching keys.shape[0], "
            f"got {tuple(assignments.shape)} for N={keys.shape[0]}"
        )
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    if prev_centroids is not None:
        _check_centroids_2d(prev_centroids)
        if prev_centroids.shape != (num_clusters, keys.shape[1]):
            raise ValueError(
                "prev_centroids must have shape [C, D], "
                f"got {tuple(prev_centroids.shape)} for C={num_clusters}, D={keys.shape[1]}"
            )

    keys_f = keys.float()
    prev_centroids_f = prev_centroids.float() if prev_centroids is not None else None

    sums = torch.zeros(
        (num_clusters, keys.shape[1]), dtype=torch.float32, device=keys.device
    )
    counts = torch.zeros(num_clusters, dtype=torch.long, device=keys.device)

    for key_idx in range(keys.shape[0]):
        cluster_idx = int(assignments[key_idx].item())
        if cluster_idx < 0 or cluster_idx >= num_clusters:
            raise ValueError(
                f"Found out-of-range cluster assignment {cluster_idx} for C={num_clusters}"
            )
        sums[cluster_idx] += keys_f[key_idx]
        counts[cluster_idx] += 1

    new_centroids = torch.empty_like(sums)
    for cluster_idx in range(num_clusters):
        if counts[cluster_idx] > 0:
            new_centroids[cluster_idx] = sums[cluster_idx] / counts[cluster_idx].float()
        else:
            if prev_centroids_f is None:
                raise ValueError(
                    f"Cluster {cluster_idx} is empty and prev_centroids was not provided"
                )
            new_centroids[cluster_idx] = prev_centroids_f[cluster_idx]

    return new_centroids, counts


def update_centroids_vectorized(
    keys: Tensor,
    assignments: Tensor,
    num_clusters: int,
    prev_centroids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Vectorized centroid update using PyTorch scatter-style primitives.

    This function uses FP32 accumulation via ``index_add_`` and ``bincount``.
    That mirrors the naive reference while avoiding BF16 reduction noise.
    Empty clusters follow the same deterministic behavior as
    ``update_centroids_naive``.

    Args:
        keys: Key matrix of shape ``[N, D]``.
        assignments: Cluster indices of shape ``[N]``.
        num_clusters: Total number of clusters ``C``.
        prev_centroids: Optional previous centroids with shape ``[C, D]``.

    Returns:
        new_centroids: FP32 tensor of shape ``[C, D]``.
        counts: ``torch.long`` tensor of shape ``[C]``.
    """

    _check_keys_2d(keys)
    if assignments.ndim != 1 or assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "assignments must have shape [N] matching keys.shape[0], "
            f"got {tuple(assignments.shape)} for N={keys.shape[0]}"
        )
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    if prev_centroids is not None:
        _check_centroids_2d(prev_centroids)
        if prev_centroids.shape != (num_clusters, keys.shape[1]):
            raise ValueError(
                "prev_centroids must have shape [C, D], "
                f"got {tuple(prev_centroids.shape)} for C={num_clusters}, D={keys.shape[1]}"
            )

    if torch.any(assignments < 0) or torch.any(assignments >= num_clusters):
        raise ValueError(
            f"assignments must be in [0, {num_clusters}), got invalid entries"
        )

    keys_f = keys.float()
    prev_centroids_f = prev_centroids.float() if prev_centroids is not None else None

    sums = torch.zeros(
        (num_clusters, keys.shape[1]), dtype=torch.float32, device=keys.device
    )
    sums.index_add_(0, assignments, keys_f)

    counts = torch.bincount(assignments, minlength=num_clusters)
    new_centroids = torch.empty_like(sums)

    nonempty = counts > 0
    new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None].float()

    empty = ~nonempty
    if torch.any(empty):
        if prev_centroids_f is None:
            empty_indices = torch.nonzero(empty, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Empty clusters {empty_indices} encountered and prev_centroids was not provided"
            )
        new_centroids[empty] = prev_centroids_f[empty]

    return new_centroids, counts


def compute_kmeans_objective(keys: Tensor, centroids: Tensor, assignments: Tensor) -> Tensor:
    """Compute the total within-cluster squared distance.

    Args:
        keys: Key matrix of shape ``[N, D]``.
        centroids: Centroid matrix of shape ``[C, D]``.
        assignments: Cluster indices of shape ``[N]``.

    Returns:
        Scalar FP32 tensor containing the total squared distance.
    """

    _check_keys_2d(keys)
    _check_centroids_2d(centroids)
    _check_matching_feature_dim(keys, centroids)
    if assignments.ndim != 1 or assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "assignments must have shape [N] matching keys.shape[0], "
            f"got {tuple(assignments.shape)} for N={keys.shape[0]}"
        )
    if torch.any(assignments < 0) or torch.any(assignments >= centroids.shape[0]):
        raise ValueError(
            f"assignments must be in [0, {centroids.shape[0]}), got invalid entries"
        )

    keys_f = keys.float()
    centroids_f = centroids.float()
    assigned_centroids = centroids_f.index_select(0, assignments)
    diffs = keys_f - assigned_centroids
    return torch.sum(diffs * diffs)


def run_kmeans_naive(
    keys: Tensor, init_centroids: Tensor, num_iters: int
) -> Dict[str, object]:
    """Run Lloyd-style K-Means using the naive assignment and update steps.

    ``objectives_per_iter`` stores the objective after each centroid update and
    reassignment. This makes the reported sequence easy to compare across
    implementations and should be non-increasing for standard K-Means updates.
    For correctness comparisons, the caller should keep ``init_centroids``
    fixed across implementations.

    Returns:
        Dictionary containing:
        - ``assignments``: final assignments induced by the returned centroids
        - ``centroids``: final centroids in FP32
        - ``counts``: final cluster counts from the returned assignments
        - ``objectives_per_iter``: Python ``float`` values, length ``num_iters``
        - ``counts_per_iter``: cluster counts after each reassignment
        - ``distances``: final squared distances to assigned centroids
    """

    _check_keys_2d(keys)
    _check_centroids_2d(init_centroids)
    _check_matching_feature_dim(keys, init_centroids)
    if num_iters < 0:
        raise ValueError(f"num_iters must be non-negative, got {num_iters}")

    num_clusters = init_centroids.shape[0]
    centroids = init_centroids.clone()
    objectives_per_iter: List[float] = []
    counts_per_iter: List[Tensor] = []

    for _ in range(num_iters):
        assignments, _ = assign_clusters_naive(keys, centroids)
        centroids, _ = update_centroids_naive(
            keys, assignments, num_clusters, prev_centroids=centroids
        )
        assignments, distances = assign_clusters_naive(keys, centroids)
        objectives_per_iter.append(float(distances.sum().item()))
        counts_per_iter.append(torch.bincount(assignments, minlength=num_clusters))

    assignments, distances = assign_clusters_naive(keys, centroids)

    counts = torch.bincount(assignments, minlength=num_clusters)
    return {
        "assignments": assignments,
        "centroids": centroids.float(),
        "counts": counts,
        "objectives_per_iter": objectives_per_iter,
        "counts_per_iter": counts_per_iter,
        "distances": distances,
    }


def run_kmeans_vectorized(
    keys: Tensor, init_centroids: Tensor, num_iters: int
) -> Dict[str, object]:
    """Run Lloyd-style K-Means using vectorized assignment and update steps.

    The API and numerical behavior match ``run_kmeans_naive``. This function is
    still intended as a readable PyTorch reference, not a final optimized
    sparse-routing kernel.
    """

    _check_keys_2d(keys)
    _check_centroids_2d(init_centroids)
    _check_matching_feature_dim(keys, init_centroids)
    if num_iters < 0:
        raise ValueError(f"num_iters must be non-negative, got {num_iters}")

    num_clusters = init_centroids.shape[0]
    centroids = init_centroids.clone()
    objectives_per_iter: List[float] = []
    counts_per_iter: List[Tensor] = []

    for _ in range(num_iters):
        assignments, _ = assign_clusters_vectorized(keys, centroids)
        centroids, _ = update_centroids_vectorized(
            keys, assignments, num_clusters, prev_centroids=centroids
        )
        assignments, distances = assign_clusters_vectorized(keys, centroids)
        objectives_per_iter.append(float(distances.sum().item()))
        counts_per_iter.append(torch.bincount(assignments, minlength=num_clusters))

    assignments, distances = assign_clusters_vectorized(keys, centroids)

    counts = torch.bincount(assignments, minlength=num_clusters)
    return {
        "assignments": assignments,
        "centroids": centroids.float(),
        "counts": counts,
        "objectives_per_iter": objectives_per_iter,
        "counts_per_iter": counts_per_iter,
        "distances": distances,
    }


def reorder_by_cluster(
    keys: Tensor, assignments: Tensor, stable: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reorder keys so that keys from the same cluster are contiguous.

    The returned ``cluster_offsets`` tensor has shape ``[C + 1]`` where
    cluster ``c`` occupies the half-open interval
    ``[cluster_offsets[c], cluster_offsets[c + 1])``. This grouped layout is
    useful for downstream sparse routing because later kernels can process one
    cluster segment at a time, similar to grouped GEMM or bucketed attention
    execution.

    Args:
        keys: Key matrix of shape ``[N, D]``.
        assignments: Cluster indices of shape ``[N]``.
        stable: If ``True``, preserve relative order within each cluster.

    Returns:
        reordered_keys: Keys grouped by cluster, shape ``[N, D]``.
        reordered_indices: Indices into the original ``keys``, shape ``[N]``.
        sorted_assignments: Assignments after reordering, shape ``[N]``.
        cluster_offsets: Offsets with shape ``[C + 1]``.
    """

    _check_keys_2d(keys)
    if assignments.ndim != 1 or assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "assignments must have shape [N] matching keys.shape[0], "
            f"got {tuple(assignments.shape)} for N={keys.shape[0]}"
        )
    if torch.any(assignments < 0):
        raise ValueError("assignments must be non-negative")

    num_keys = keys.shape[0]
    if num_keys == 0:
        reordered_indices = torch.empty(0, dtype=torch.long, device=keys.device)
        cluster_offsets = torch.zeros(1, dtype=torch.long, device=keys.device)
        return keys.clone(), reordered_indices, assignments.clone(), cluster_offsets

    num_clusters = int(assignments.max().item()) + 1

    if stable:
        pieces = []
        offsets = [0]
        for cluster_idx in range(num_clusters):
            cluster_indices = torch.nonzero(
                assignments == cluster_idx, as_tuple=False
            ).flatten()
            pieces.append(cluster_indices)
            offsets.append(offsets[-1] + int(cluster_indices.numel()))

        reordered_indices = (
            torch.cat(pieces, dim=0)
            if pieces
            else torch.empty(0, dtype=torch.long, device=keys.device)
        )
        cluster_offsets = torch.tensor(
            offsets, dtype=torch.long, device=keys.device
        )
    else:
        reordered_indices = torch.argsort(assignments)
        counts = torch.bincount(assignments, minlength=num_clusters)
        cluster_offsets = torch.zeros(
            num_clusters + 1, dtype=torch.long, device=keys.device
        )
        cluster_offsets[1:] = torch.cumsum(counts, dim=0)

    reordered_keys = keys.index_select(0, reordered_indices)
    sorted_assignments = assignments.index_select(0, reordered_indices)
    return reordered_keys, reordered_indices, sorted_assignments, cluster_offsets


def validate_reordering(
    keys: Tensor,
    assignments: Tensor,
    reordered_keys: Tensor,
    reordered_indices: Tensor,
    sorted_assignments: Tensor,
    cluster_offsets: Tensor,
) -> Dict[str, object]:
    """Validate that a cluster-based reordering is internally consistent.

    The function raises ``AssertionError`` on inconsistencies and otherwise
    returns a small dictionary of check results.
    """

    _check_keys_2d(keys)
    _check_keys_2d(reordered_keys)
    if assignments.ndim != 1 or assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "assignments must have shape [N] matching keys.shape[0], "
            f"got {tuple(assignments.shape)} for N={keys.shape[0]}"
        )
    if reordered_indices.ndim != 1 or reordered_indices.shape[0] != keys.shape[0]:
        raise ValueError(
            "reordered_indices must have shape [N], "
            f"got {tuple(reordered_indices.shape)} for N={keys.shape[0]}"
        )
    if sorted_assignments.ndim != 1 or sorted_assignments.shape[0] != keys.shape[0]:
        raise ValueError(
            "sorted_assignments must have shape [N], "
            f"got {tuple(sorted_assignments.shape)} for N={keys.shape[0]}"
        )
    if cluster_offsets.ndim != 1:
        raise ValueError(
            f"cluster_offsets must have shape [C + 1], got {tuple(cluster_offsets.shape)}"
        )

    num_keys = keys.shape[0]
    num_clusters = cluster_offsets.shape[0] - 1
    assert reordered_keys.shape == keys.shape
    assert torch.equal(reordered_keys, keys.index_select(0, reordered_indices))
    assert torch.equal(sorted_assignments, assignments.index_select(0, reordered_indices))

    index_counts = torch.bincount(reordered_indices.cpu(), minlength=num_keys)
    assert int(index_counts.sum().item()) == num_keys
    assert torch.all(index_counts == 1), "Each original key must appear exactly once"

    original_counts = torch.bincount(assignments.cpu(), minlength=num_clusters)
    reordered_counts = torch.bincount(sorted_assignments.cpu(), minlength=num_clusters)
    assert torch.equal(original_counts, reordered_counts)

    for cluster_idx in range(num_clusters):
        start = int(cluster_offsets[cluster_idx].item())
        end = int(cluster_offsets[cluster_idx + 1].item())
        assert 0 <= start <= end <= num_keys
        segment = sorted_assignments[start:end]
        assert torch.all(segment == cluster_idx), (
            f"Cluster segment {cluster_idx} contains incorrect assignments"
        )

    assert int(cluster_offsets[0].item()) == 0
    assert int(cluster_offsets[-1].item()) == num_keys
    assert torch.all(cluster_offsets[1:] >= cluster_offsets[:-1])

    return {
        "num_keys": num_keys,
        "num_clusters": num_clusters,
        "counts_match": True,
        "all_elements_present_once": True,
        "segments_are_pure": True,
    }
