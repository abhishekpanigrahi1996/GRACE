import argparse
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute GRACE score from stored projected gradients."
    )
    parser.add_argument(
        "--gradients-path",
        type=str,
        required=True,
        help="Path to pickle file containing gradients.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Projected gradient dimension.",
    )
    parser.add_argument(
        "--n-gen-per-prompt",
        type=int,
        default=4,
        help="Number of generations per prompt.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of prompts used for test split.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of random splits.",
    )
    parser.add_argument(
        "--smooth-coeff",
        type=float,
        default=1e-3,
        help="Smoothing coefficient for eigenvalues.",
    )


    return parser.parse_args()


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return x / norms


def _compute_ntk(x: np.ndarray) -> np.ndarray:
    return (x.T @ x) / len(x)


def _smooth_eigenvalues(
    w: np.ndarray,
    smooth_coeff: float = 1e-3,
    smooth_cond: bool = False,
    smooth_max_by_sum: bool = False,
    smooth_erank: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    w = np.clip(w, eps, None)

    mean_eig = np.sum(w) / len(w)
    w_smoothed = (1.0 - smooth_coeff) * w + smooth_coeff * mean_eig

    return np.clip(w_smoothed, eps, None)


def _compute_pinv_from_eigh(
    kernel: np.ndarray,
    smooth_coeff: float = 1e-3,
    smooth_cond: bool = False,
    smooth_max_by_sum: bool = False,
    smooth_erank: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    w, v = np.linalg.eigh(kernel)
    w_smoothed = _smooth_eigenvalues(
        w,
        smooth_coeff=smooth_coeff,
        smooth_cond=smooth_cond,
        smooth_max_by_sum=smooth_max_by_sum,
        smooth_erank=smooth_erank,
        eps=eps,
    )
    return (v * (1.0 / w_smoothed)) @ v.T


def grace(
    gradients: np.ndarray,
    dim: int,
    n_gen_per_prompt: int = 4,
    test_fraction: float = 0.1,
    n_splits: int = 10,
    smooth_coeff: float = 1e-3,
    smooth_cond: bool = False,
    smooth_max_by_sum: bool = False,
    smooth_erank: bool = False,
    normalize_train: bool = True,
    normalize_test: bool = False,
    eps: float = 1e-12,
):
    gradients = np.asarray(gradients, dtype=np.float64)

    if gradients.ndim != 2:
        raise ValueError(f"`gradients` must be 2D, got shape {gradients.shape}")

    if gradients.shape[1] != dim:
        raise ValueError(f"Expected gradients.shape[1] == {dim}, got {gradients.shape[1]}")

    n_total = gradients.shape[0]
    if n_total % n_gen_per_prompt != 0:
        raise ValueError(
            f"Number of rows ({n_total}) must be divisible by n_gen_per_prompt ({n_gen_per_prompt})"
        )

    n_prompts = n_total // n_gen_per_prompt
    n_test_prompts = max(1, int(n_prompts * test_fraction))
    n_train_prompts = n_prompts - n_test_prompts

    if n_train_prompts <= 0:
        raise ValueError(
            f"Not enough prompts for train/test split: "
            f"n_prompts={n_prompts}, test_fraction={test_fraction}"
        )

    grouped = gradients.reshape(n_prompts, n_gen_per_prompt, dim)
    scores = []

    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_prompts)
        shuffled = grouped[perm]

        train_grads = shuffled[:n_train_prompts].reshape(-1, dim)
        test_grads = shuffled[n_train_prompts:].reshape(-1, dim)

        if normalize_train:
            train_grads = _normalize_rows(train_grads, eps=eps)
        if normalize_test:
            test_grads = _normalize_rows(test_grads, eps=eps)

        kernel_train = _compute_ntk(train_grads)
        kernel_test = _compute_ntk(test_grads)

        kernel_train_pinv = _compute_pinv_from_eigh(
            kernel_train,
            smooth_coeff=smooth_coeff,
            smooth_cond=smooth_cond,
            smooth_max_by_sum=smooth_max_by_sum,
            smooth_erank=smooth_erank,
            eps=eps,
        )

        score = np.trace(kernel_train_pinv @ kernel_test)
        scores.append(score)

    scores = np.asarray(scores, dtype=np.float64)
    mean_score = float(scores.mean())

    return mean_score


def load_gradients(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        gradients = pickle.load(f)

    gradients = np.asarray(gradients)

    if gradients.ndim != 2:
        raise ValueError(
            f"Loaded gradients must be 2D after conversion to np.array, got shape {gradients.shape}"
        )

    return gradients


def main():
    args = parse_args()


    print(f"Loading gradients from: {args.gradients_path}")
    gradients = load_gradients(args.gradients_path)
    print(f"Loaded gradients with shape: {gradients.shape}")

    grace_score = grace(
        gradients=gradients,
        dim=args.dim,
        n_gen_per_prompt=args.n_gen_per_prompt,
        test_fraction=args.test_fraction,
        n_splits=args.n_splits,
        smooth_coeff=args.smooth_coeff,
        normalize_train=True,
        normalize_test=False,
    )

    print(f"GRACE score: {grace_score:.10f}")



if __name__ == "__main__":
    main()