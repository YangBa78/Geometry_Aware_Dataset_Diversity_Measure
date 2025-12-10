import numpy as np
import gudhi as gd
from persim.landscapes import PersLandscapeApprox

def compute_PLDiv(
    data,
    distance_matrix=False,
    hom_deg=0,
    pooled_mode="resample",
    sparse=0.3,
    closed_form=True,
    num_steps=500,
    plot=False,
    max_edge_length=2, 
    max_dimension=1
):
    """
    Compute persistence-based diversity score : PLDiv
    using GUDHI with Vietoris–Rips complex.

    Parameters
    ----------
    data : array-like
        Input point cloud (n_points x dim) or distance matrix.
    distance_matrix : bool, optional
        If True, interpret `data` as a distance matrix.
    hom_deg : {0, 1, "both"}, optional
        Homology degree(s) to compute:
            0 → H0 only,
            1 → H1 only,
            "both" → compute H0 and H1 and pool them.
    pooled_mode : {"resample", "add"}, optional
        Pooling method when hom_deg="both":
            - "resample": resample H0 and H1 landscapes on common grid
            - "add": sum scalar integrals
    sparse : float, optional
        Fraction of edges to retain (0 < sparse <= 1).
    closed_form : bool, optional
        If True, compute PLDiv directly (fast closed-form formula).
        If False, use persistence landscapes and integrate.
    num_steps : int, optional
        Number of discretization steps for landscapes (ignored if closed_form=True).
    plot : bool, optional
        If True, plot persistence landscapes (only if closed_form=False).

    Returns
    -------
    np.ndarray
        1D NumPy vector [value].
    """

    # Build sparse Rips complex 
    if distance_matrix:
        rips_complex = gd.RipsComplex(distance_matrix=data, max_edge_length=max_edge_length, sparse=sparse)
    else:
        rips_complex = gd.RipsComplex(points=data, sparse=sparse)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    simplex_tree.compute_persistence()

    # Helper for each homology degree
    def _compute_for_degree(deg):
        dgm = np.array(simplex_tree.persistence_intervals_in_dimension(deg))
        if dgm.size == 0:
            return None, None, None
        dgm = dgm[np.isfinite(dgm[:, 1])]
        if dgm.shape[0] == 0:
            return None, None, None

        if closed_form:
            # Fast closed-form PLDiv computation
            val = 0.25 * np.sum((dgm[:, 1] - dgm[:, 0]) ** 2)
            return {"PLDiv": val}, None, None

        # Persistence landscapes computation
        pla = PersLandscapeApprox(dgms=[dgm], hom_deg=deg, num_steps=num_steps)
        if plot:
            from persim import plot_landscape_simple
            plot_landscape_simple(pla)
        landscape_values = pla.values
        x_grid = np.linspace(0, dgm[:, 1].max(), landscape_values.shape[1])
        step = x_grid[1] - x_grid[0]
        stats = {"PLDiv": np.sum(landscape_values) * step}
        return stats, landscape_values, step

    # Compute values based on hom_deg 
    results = {}
    H0_vals = H1_vals = step0 = step1 = None

    if hom_deg in [0, "both"]:
        results["H0"], H0_vals, step0 = _compute_for_degree(0)
    if hom_deg in [1, "both"]:
        results["H1"], H1_vals, step1 = _compute_for_degree(1)

    if hom_deg == "both":
        if closed_form:
            h0 = results["H0"]["PLDiv"] if results.get("H0") else 0.0
            h1 = results["H1"]["PLDiv"] if results.get("H1") else 0.0
            val = h0 + h1
        else:
            if pooled_mode == "resample" and H0_vals is not None and H1_vals is not None:
                pooled_vals = np.vstack([H0_vals, H1_vals])
                step = min(step0, step1)
                val = np.sum(pooled_vals) * step
            elif pooled_mode == "add":
                h0 = results["H0"]["PLDiv"] if results.get("H0") else 0.0
                h1 = results["H1"]["PLDiv"] if results.get("H1") else 0.0
                val = h0 + h1
            else:
                val = 0.0
    else:
        val = (
            results["H0"]["PLDiv"] if hom_deg == 0 and results.get("H0")
            else results["H1"]["PLDiv"] if hom_deg == 1 and results.get("H1")
            else 0.0
        )

    return val