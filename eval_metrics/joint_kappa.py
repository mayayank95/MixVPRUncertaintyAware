"""Joint vMF concentration utilities."""
import numpy as np


def compute_joint_kappa(
    predictions: np.ndarray,
    q_var: np.ndarray,
    db_var: np.ndarray,
    q_desc: np.ndarray,
    db_desc: np.ndarray,
) -> np.ndarray:
    """Return top-k joint kappa scores for retrieved query/reference pairs."""
    kappa_q = np.mean(q_var, axis=1)
    mean_db_var = np.mean(db_var, axis=1)
    total_queries, k = predictions.shape

    q_norms = np.linalg.norm(q_desc, axis=1, keepdims=True)
    db_norms = np.linalg.norm(db_desc, axis=1, keepdims=True)
    q_normed = q_desc / np.maximum(q_norms, 1e-8)
    db_normed = db_desc / np.maximum(db_norms, 1e-8)

    joint_top_k = np.zeros((total_queries, k), dtype=np.float64)
    for rank in range(k):
        ref_idx = predictions[:, rank]
        kappa_ref = mean_db_var[ref_idx]
        mu_dot = np.sum(q_normed * db_normed[ref_idx], axis=1)
        joint_top_k[:, rank] = np.sqrt(
            np.maximum(
                kappa_q**2 + kappa_ref**2 + 2 * kappa_q * kappa_ref * mu_dot,
                0.0,
            )
        )

    return joint_top_k
