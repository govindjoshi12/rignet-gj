import torch

def epanechnikov_kernel(r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Compute the Epanechnikov kernel for pairwise differences.

    Args:
        r (torch.Tensor): Tensor of shape [..., D], where the last dimension
                          represents D-dimensional difference vectors
                          (e.g., shape [N, N, D] for pairwise differences).
        h (torch.Tensor): Scalar tensor, the bandwidth (kernel radius).

    Returns:
        torch.Tensor: Tensor of shape matching r without the last dimension
                      (e.g., [N, N]), containing kernel weights
                      K = max(0, 1 - ||r||^2 / h^2).
    """
    dist2 = (r ** 2).sum(dim=r.dim() - 1)  # squared L2 norm over last axis
    return torch.clamp(1.0 - (dist2 / (h ** 2)), min=0.0)  # kernel values

def mean_shift_update_step(x: torch.Tensor, 
                           a: torch.Tensor,
                           h: torch.Tensor):

    """
    Perform one mean-shift update step on a set of points.

    Args:
        x (torch.Tensor): Tensor of shape [N, D], the current point positions.
        a (torch.Tensor): Tensor of shape [N], attention weights for each point.
        h (torch.Tensor): Scalar tensor, the bandwidth (kernel radius).

    Returns:
        torch.Tensor: Tensor of shape [N, D], the updated point positions.
    """

    N, _ = x.shape

    # Vectorized pairwise differences: [N, 1, D] - [1, N, D] = [N, N, D]
    diff = x.unsqueeze(dim=1) - x.unsqueeze(dim=0)

    # Each the i-th row in K represents the attention-scaled kernel values
    # for each point x_j with a fixed x_i
    K = epanechnikov_kernel(diff, h)

    # broadcast a[j] for each row
    w = K * a.view(1, N)

    # Dot product of i-th row with i-th vertex for each row
    numerator = w @ x
    denominator = w.sum(dim=1, keepdim=True)

    eps = 1e-8 
    x_next = numerator / (denominator + eps)

    return x_next

def mean_shift_clustering(vertices: torch.Tensor, 
                          attention: torch.Tensor, 
                          h: float = 1.0,
                          tol: float = 1e-3, 
                          max_iters: int = 50) -> torch.Tensor:
    """
    Perform mean-shift clustering by iteratively updating point positions.

    Args:
        vertices (torch.Tensor): Tensor of shape [N, D], the initial point positions.
        attention (torch.Tensor): Tensor of shape [N], attention weights per point.
        h (float, optional): Bandwidth (kernel radius) for the Epanechnikov kernel. Defaults to 1.0.
        tol (float, optional): Convergence tolerance. If the maximum point shift magnitude
                               falls below this value, iteration stops early. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of mean-shift iterations. Defaults to 50.

    Returns:
        torch.Tensor: Tensor of shape [N, D], the final (converged) point positions.
    """
    x = vertices.clone()
    N, _ = x.shape

    for _ in range(max_iters):
        x_next = mean_shift_update_step(x, attention, h)
        # Norm over D dims gives each point’s shift magnitude; take the max over N points
        if torch.max((x_next - x).norm(dim=1)) < tol:
            break
        x = x_next

    return x

def mode_extraction(vertices: torch.Tensor, 
                    attention: torch.Tensor,
                    h: torch.Tensor,
                    densities: torch.Tensor = None) -> torch.Tensor:
    """
    Extract cluster modes via blurring-sharpening after mean-shift convergence.

    Args:
        vertices (torch.Tensor): Tensor of shape [N, D], converged point positions.
        attention (torch.Tensor): Tensor of shape [N], attention weights per point.
        h (torch.Tensor): Scalar tensor, the bandwidth (suppression radius).
        densities (torch.Tensor, optional): Precomputed densities of shape [N].
            If provided, skips density computation. Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape [M, D], the extracted mode (cluster center) coordinates.
                       Returns an empty tensor of shape [0, D] if no modes found.
    """
    x = vertices
    a = attention
    N, D = x.shape

    # Compute densities if not provided (attention‑weighted kernel sums)
    if densities is None:
        diff = x.unsqueeze(1) - x.unsqueeze(0)     # [N, N, D]
        K = epanechnikov_kernel(diff, h)           # [N, N]
        w = a.view(1, N) * K                       # [N, N]
        densities = w.sum(dim=1)                   # [N]

    # Precompute pairwise Euclidean distances for suppression
    diff = x.unsqueeze(1) - x.unsqueeze(0)         # [N, N, D]
    dist = torch.sqrt((diff * diff).sum(dim=2))    # [N, N]

    # Mode selection with suppression
    unused_mask = torch.ones(N, dtype=torch.bool)  # True = point is still eligible
    modes = []

    while unused_mask.any():
        # Mask out used points by setting their density to -inf
        masked_density = densities.masked_fill(~unused_mask, float('-inf'))
        # Select the index of the highest-density unused point
        i_star = torch.argmax(masked_density).item()
        modes.append(x[i_star])

        # Suppress all points within radius h of the chosen mode
        suppress = dist[i_star] <= h
        unused_mask = unused_mask & (~suppress)

    return torch.stack(modes, dim=0) if modes else torch.empty((0, D))

