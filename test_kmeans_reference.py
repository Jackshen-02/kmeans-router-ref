import pytest
import torch

from kmeans_reference import (
    assign_clusters_naive,
    assign_clusters_vectorized,
    compute_kmeans_objective,
    init_centroids_from_keys,
    reorder_by_cluster,
    run_kmeans_naive,
    run_kmeans_vectorized,
    update_centroids_naive,
    update_centroids_vectorized,
    validate_reordering,
)


def test_assign_clusters_naive_matches_vectorized() -> None:
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [3.0, 3.0],
            [2.8, 3.2],
            [8.0, 8.0],
        ],
        dtype=torch.float32,
    )
    centroids = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 3.0],
            [9.0, 9.0],
        ],
        dtype=torch.float32,
    )

    naive_assignments, naive_distances = assign_clusters_naive(keys, centroids)
    vec_assignments, vec_distances = assign_clusters_vectorized(keys, centroids)

    expected_assignments = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    assert torch.equal(naive_assignments, expected_assignments)
    assert torch.equal(vec_assignments, expected_assignments)
    assert torch.allclose(naive_distances, vec_distances)


def test_update_centroids_naive_matches_vectorized() -> None:
    keys = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 4.0],
            [6.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    prev_centroids = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)

    naive_centroids, naive_counts = update_centroids_naive(
        keys, assignments, num_clusters=2, prev_centroids=prev_centroids
    )
    vec_centroids, vec_counts = update_centroids_vectorized(
        keys, assignments, num_clusters=2, prev_centroids=prev_centroids
    )

    expected_centroids = torch.tensor([[1.0, 0.0], [5.0, 4.0]], dtype=torch.float32)
    assert torch.equal(naive_counts, torch.tensor([2, 2], dtype=torch.long))
    assert torch.equal(vec_counts, torch.tensor([2, 2], dtype=torch.long))
    assert torch.allclose(naive_centroids, expected_centroids)
    assert torch.allclose(vec_centroids, expected_centroids)
    assert torch.allclose(naive_centroids, vec_centroids)


def test_run_kmeans_naive_matches_vectorized_and_objective_is_monotonic() -> None:
    keys = torch.tensor(
        [
            [0.0, 0.1],
            [0.1, -0.1],
            [4.0, 4.0],
            [4.2, 3.8],
            [8.0, 8.0],
            [7.8, 8.1],
        ],
        dtype=torch.float32,
    )
    init_centroids = torch.tensor(
        [
            [0.0, 0.0],
            [3.5, 3.5],
            [8.5, 8.5],
        ],
        dtype=torch.float32,
    )

    naive = run_kmeans_naive(keys, init_centroids, num_iters=4)
    vec = run_kmeans_vectorized(keys, init_centroids, num_iters=4)

    assert torch.equal(naive["assignments"], vec["assignments"])
    assert torch.allclose(naive["centroids"], vec["centroids"], atol=1e-6, rtol=1e-6)
    assert torch.equal(naive["counts"], vec["counts"])
    assert torch.allclose(naive["distances"], vec["distances"], atol=1e-6, rtol=1e-6)

    naive_objectives = naive["objectives_per_iter"]
    vec_objectives = vec["objectives_per_iter"]
    assert len(naive_objectives) == 4
    assert len(vec_objectives) == 4
    assert naive_objectives == pytest.approx(vec_objectives, rel=1e-6, abs=1e-6)
    assert naive_objectives == sorted(naive_objectives, reverse=True)

    objective = compute_kmeans_objective(
        keys, vec["centroids"], vec["assignments"]
    ).item()
    assert objective == pytest.approx(vec["distances"].sum().item(), rel=1e-6, abs=1e-6)


def test_reorder_by_cluster_is_correct_and_stable() -> None:
    keys = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
            [50.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assignments = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long)

    reordered_keys, reordered_indices, sorted_assignments, cluster_offsets = reorder_by_cluster(
        keys, assignments, stable=True
    )

    assert torch.equal(reordered_indices, torch.tensor([1, 4, 3, 0, 2], dtype=torch.long))
    assert torch.equal(sorted_assignments, torch.tensor([0, 0, 1, 2, 2], dtype=torch.long))
    assert torch.equal(cluster_offsets, torch.tensor([0, 2, 3, 5], dtype=torch.long))
    assert torch.equal(
        reordered_keys[:, 0], torch.tensor([20.0, 50.0, 40.0, 10.0, 30.0])
    )

    checks = validate_reordering(
        keys,
        assignments,
        reordered_keys,
        reordered_indices,
        sorted_assignments,
        cluster_offsets,
    )
    assert checks["all_elements_present_once"]
    assert checks["segments_are_pure"]


def test_vectorized_kmeans_supports_bf16_inputs_with_fp32_internal_accumulation() -> None:
    keys_fp32 = torch.tensor(
        [
            [0.0, 0.2, 0.1],
            [0.1, -0.1, 0.0],
            [5.0, 5.2, 4.8],
            [4.9, 5.1, 5.0],
        ],
        dtype=torch.float32,
    )
    init_centroids_fp32 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.1, 5.1, 5.1],
        ],
        dtype=torch.float32,
    )

    fp32_result = run_kmeans_vectorized(keys_fp32, init_centroids_fp32, num_iters=3)
    bf16_result = run_kmeans_vectorized(
        keys_fp32.to(torch.bfloat16), init_centroids_fp32.to(torch.bfloat16), num_iters=3
    )

    assert bf16_result["centroids"].dtype == torch.float32
    assert torch.allclose(
        bf16_result["centroids"], fp32_result["centroids"], atol=2e-2, rtol=2e-2
    )
    assert torch.equal(bf16_result["assignments"], fp32_result["assignments"])
    assert bf16_result["objectives_per_iter"] == pytest.approx(
        fp32_result["objectives_per_iter"], rel=2e-2, abs=2e-2
    )


def test_empty_cluster_keeps_previous_centroid_when_provided() -> None:
    keys = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]], dtype=torch.float32)
    assignments = torch.tensor([0, 0, 1], dtype=torch.long)
    prev_centroids = torch.tensor(
        [[0.0, 0.0], [10.0, 10.0], [99.0, 99.0]], dtype=torch.float32
    )

    new_centroids, counts = update_centroids_vectorized(
        keys, assignments, num_clusters=3, prev_centroids=prev_centroids
    )

    assert torch.equal(counts, torch.tensor([2, 1, 0], dtype=torch.long))
    assert torch.allclose(new_centroids[0], torch.tensor([0.05, 0.0]))
    assert torch.allclose(new_centroids[1], torch.tensor([10.0, 10.0]))
    assert torch.allclose(new_centroids[2], torch.tensor([99.0, 99.0]))


def test_empty_cluster_raises_without_previous_centroid() -> None:
    keys = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    assignments = torch.tensor([0, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="Empty clusters|Cluster 1 is empty"):
        update_centroids_naive(keys, assignments, num_clusters=2, prev_centroids=None)

    with pytest.raises(ValueError, match="Empty clusters|Cluster 1 is empty"):
        update_centroids_vectorized(
            keys, assignments, num_clusters=2, prev_centroids=None
        )


def test_randomized_small_cases_match_across_multiple_seeds() -> None:
    for seed in range(5):
        torch.manual_seed(seed)
        keys = torch.randn(12, 4, dtype=torch.float32)
        init_centroids = init_centroids_from_keys(keys, num_clusters=3, seed=seed)

        naive = run_kmeans_naive(keys, init_centroids, num_iters=3)
        vec = run_kmeans_vectorized(keys, init_centroids, num_iters=3)

        assert torch.equal(naive["assignments"], vec["assignments"])
        assert torch.allclose(naive["centroids"], vec["centroids"], atol=1e-6, rtol=1e-6)
        assert torch.equal(naive["counts"], vec["counts"])
        assert naive["objectives_per_iter"] == pytest.approx(
            vec["objectives_per_iter"], rel=1e-6, abs=1e-6
        )
        for naive_counts, vec_counts in zip(
            naive["counts_per_iter"], vec["counts_per_iter"]
        ):
            assert torch.equal(naive_counts, vec_counts)


def test_skewed_cluster_sizes_match_between_naive_and_vectorized() -> None:
    keys = torch.tensor(
        [
            [-0.2, 0.0],
            [-0.1, 0.1],
            [0.0, 0.0],
            [0.1, -0.1],
            [0.2, 0.0],
            [0.0, 0.2],
            [0.1, 0.1],
            [5.0, 5.1],
            [5.2, 4.9],
            [10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    init_centroids = torch.tensor(
        [[0.0, 0.0], [5.1, 5.0], [10.0, 10.0]], dtype=torch.float32
    )

    naive = run_kmeans_naive(keys, init_centroids, num_iters=3)
    vec = run_kmeans_vectorized(keys, init_centroids, num_iters=3)

    expected_counts = torch.tensor([7, 2, 1], dtype=torch.long)
    assert torch.equal(naive["counts"], expected_counts)
    assert torch.equal(vec["counts"], expected_counts)
    assert torch.equal(naive["assignments"], vec["assignments"])


def test_tie_assignment_prefers_lowest_cluster_index() -> None:
    keys = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    centroids = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    naive_assignments, naive_distances = assign_clusters_naive(keys, centroids)
    vec_assignments, vec_distances = assign_clusters_vectorized(keys, centroids)

    assert naive_assignments.tolist() == [0, 1]
    assert vec_assignments.tolist() == [0, 1]
    assert torch.allclose(naive_distances, vec_distances)


def test_zero_and_one_iteration_boundary_behavior() -> None:
    keys = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.2, 3.9]], dtype=torch.float32
    )
    init_centroids = torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)

    zero_iter = run_kmeans_vectorized(keys, init_centroids, num_iters=0)
    one_iter = run_kmeans_vectorized(keys, init_centroids, num_iters=1)
    init_assignments, init_distances = assign_clusters_vectorized(keys, init_centroids)

    assert zero_iter["objectives_per_iter"] == []
    assert zero_iter["counts_per_iter"] == []
    assert torch.equal(zero_iter["assignments"], init_assignments)
    assert torch.equal(zero_iter["counts"], torch.tensor([2, 2], dtype=torch.long))
    assert torch.allclose(zero_iter["distances"], init_distances)

    assert len(one_iter["objectives_per_iter"]) == 1
    assert len(one_iter["counts_per_iter"]) == 1
    assert one_iter["objectives_per_iter"][0] <= zero_iter["distances"].sum().item()


def test_reordering_inverse_permutation_recovers_original_order() -> None:
    keys = torch.tensor(
        [[3.0, 0.0], [1.0, 0.0], [4.0, 0.0], [2.0, 0.0]], dtype=torch.float32
    )
    assignments = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    reordered_keys, reordered_indices, _, _ = reorder_by_cluster(
        keys, assignments, stable=True
    )

    inverse = torch.empty_like(reordered_indices)
    inverse[reordered_indices] = torch.arange(keys.shape[0], dtype=torch.long)
    recovered_keys = reordered_keys.index_select(0, inverse)

    assert torch.equal(recovered_keys, keys)
