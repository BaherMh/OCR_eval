
import json

import paddle
import paddle.nn.functional as F
from merge_utils import build_and_load, load_config, process_config, save_merged_model

from ppocr.modeling.architectures import apply_to_static, build_model


def slerp(t, w1, w2, eps=1e-8, key=None):
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same shape")

    v0 = w1.flatten()
    v1 = w2.flatten()

    v0_norm = F.normalize(v0, axis=0)
    v1_norm = F.normalize(v1, axis=0)

    dot = paddle.sum(v0_norm * v1_norm)
    dot = paddle.clip(dot, -1.0 + eps, 1.0 - eps)

    theta = paddle.acos(dot)
    sin_theta = paddle.sin(theta)

    if sin_theta < eps:
        # fallback to LERP
        return (1 - t) * w1 + t * w2

    s0 = paddle.sin((1 - t) * theta) / sin_theta
    s1 = paddle.sin(t * theta) / sin_theta

    out = s0 * v0 + s1 * v1
    return out.reshape(w1.shape)


import paddle
import paddle.nn.functional as F


def cosine_midpoint(w1, w2, eps=1e-8):
    """
    Compute a matrix W that minimizes d_cos(W, W1) + d_cos(W, W2),
    where d_cos(A, B) = 1 - cos_sim(A, B).
    
    The solution is: W ∝ normalize(W1) + normalize(W2)
    (with normalization in Frobenius norm, i.e., L2 over all entries).
    
    Returns W with the same shape as w1/w2.
    """
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same shape")

    # Flatten to treat as vectors
    v1 = w1.flatten()
    v2 = w2.flatten()
    print("Using cosine!!")
    # Compute Frobenius norms (L2 norm of flattened vector)
    norm1 = paddle.norm(v1, p=2)
    norm2 = paddle.norm(v2, p=2)

    # Avoid division by zero
    v1_norm = v1 / paddle.clip(norm1, min=eps)
    v2_norm = v2 / paddle.clip(norm2, min=eps)

    # Sum of normalized vectors
    w_flat = v1_norm + v2_norm

    # Optional: normalize the result (not required for minimizing cosine distance,
    # but often useful for stability or consistency)
    # w_flat = w_flat / paddle.clip(paddle.norm(w_flat, p=2), min=eps)

    return w_flat.reshape(w1.shape)

def weight_average(
    w_base,
    w_ft,
    alpha=0.5,
):
    """
    Weight averaging between base and fine-tuned model.

    Args:
        w_base: base model weights (Tensor)
        w_ft: fine-tuned model weights (Tensor)
        alpha: interpolation factor (0 → base, 1 → fine-tuned)

    Returns:
        merged weights
    """

    if w_base.shape != w_ft.shape:
        raise ValueError("w_base and w_ft must have the same shape")

    return (1.0 - alpha) * w_base + alpha * w_ft



def dare_merge(
    w_base,
    w_ft,
    p=0.9,
    alpha=1.0,
):
    """
    DARE (Drop And REscale) merging technique using PaddlePaddle.

    Args:
        w_base: base model weights (paddle.Tensor)
        w_ft: fine-tuned model weights (paddle.Tensor)
        p: drop rate (probability of setting a delta to zero, e.g., 0.9)
        alpha: scaling factor for the final merged delta

    Returns:
        merged weights (paddle.Tensor)
    """

    if w_base.shape != w_ft.shape:
        raise ValueError("w_base and w_ft must have the same shape")

    # 1. Calculate the delta
    delta = w_ft - w_base

    # 2. Drop: Create a mask using Bernoulli distribution
    # In Paddle, we provide a tensor of probabilities to paddle.bernoulli
    probs = paddle.full_like(delta, 1.0 - p)
    mask = paddle.bernoulli(probs)

    # 3. Rescale: Adjust the delta to maintain expected value
    # We divide by (1 - p) to compensate for the dropped parameters
    rescaled_delta = (delta * mask) / (1.0 - p)

    # 4. Final Merge
    w_merged = w_base + alpha * rescaled_delta

    return w_merged


import paddle


def svd_based_merge(
    w1,
    w2,
    alpha=0.5,
    use_so=False
):
    """
    Merge two weight tensors using SVD-based interpolation for 2D weights,
    and simple averaging otherwise.

    For 2D tensors (e.g., Linear layers):
        W1 = U1 @ Sigma1 @ V1.T
        W2 = U2 @ Sigma2 @ V2.T
        U* = orthogonal_mean(U1, U2)
        V* = orthogonal_mean(V1, V2)
        Sigma* = (1-alpha)*Sigma1 + alpha*Sigma2
        W_merged = U* @ diag(Sigma*) @ V*.T

    For non-2D tensors (bias, LN, embeddings, etc.):
        W_merged = (1-alpha)*w1 + alpha*w2

    Args:
        w1 (Tensor): First weight tensor.
        w2 (Tensor): Second weight tensor.
        alpha (float): Interpolation factor (0 → w1, 1 → w2).
        use_so (bool): If True, enforce det=+1 for U*, V* (rotation group SO(n)).

    Returns:
        Tensor: Merged weight tensor of same shape as w1/w2.
    """
    if w1.shape != w2.shape:
        raise ValueError(f"Weight shapes must match: {w1.shape} vs {w2.shape}")

    # Non-2D: fall back to standard interpolation
    if w1.ndim != 2:
        return slerp(0.5, w1, w2)

    print("USING SVD!")
    # --- 2D case: SVD-based structured merge ---
    U1, S1, V1t = paddle.linalg.svd(w1)
    U2, S2, V2t = paddle.linalg.svd(w2)
    V1 = V1t.t()
    V2 = V2t.t()

    # Average singular values
    S_star = (1.0 - alpha) * S1 + alpha * S2

    def orthogonal_mean(Q1, Q2, use_so=False):
        M = 0.5 * (Q1 + Q2)
        U, _, Vt = paddle.linalg.svd(M)
        Q_star = U @ Vt

        if use_so:
            det = paddle.linalg.det(Q_star)
            if det < 0:
                U = U.clone()
                U[:, -1] *= -1
                Q_star = U @ Vt
        return Q_star

    U_star = orthogonal_mean(U1, U2, use_so=use_so)
    V_star = orthogonal_mean(V1, V2, use_so=use_so)

    # Reconstruct merged weight
    W_merged = U_star @ paddle.diag(S_star) @ V_star.t()
    return W_merged


def procrustes_weight_merge(
    w_base,
    w_ft,
    alpha=0.5,
):
    """
    Merge two weight tensors using Procrustes alignment for 2D weights (e.g., Linear layers),
    otherwise fall back to standard interpolation.

    This aligns w_ft to w_base via an orthogonal transformation Q that minimizes
        ||w_ft @ Q - w_base||_F,
    then interpolates: (1 - alpha) * w_base + alpha * (w_ft @ Q)

    Args:
        w_base (Tensor): Base model weight tensor.
        w_ft (Tensor): Fine-tuned model weight tensor.
        alpha (float): Interpolation factor (0 → base, 1 → aligned fine-tuned).

    Returns:
        Tensor: Merged weight tensor.
    """
    if w_base.shape != w_ft.shape:
        raise ValueError(f"w_base and w_ft must have the same shape, got {w_base.shape} vs {w_ft.shape}")

    # Only apply Procrustes to 2D weight matrices (e.g., Linear layers: [out_features, in_features])
    if w_base.ndim == 2:
        print("Using new strategy!")
        # Compute M = w_ft^T @ w_base
        M = paddle.matmul(w_ft.T, w_base)  # [in_features, in_features]

        # SVD: M = U @ Sigma @ Vt
        U, _, Vt = paddle.linalg.svd(M)

        # Optimal orthogonal matrix Q = U @ Vt
        Q = paddle.matmul(U, Vt)

        # Align w_ft: w_ft_aligned = w_ft @ Q
        w_ft_aligned = paddle.matmul(w_ft, Q)

        # Interpolate
        merged = (1.0 - alpha) * w_base + alpha * w_ft_aligned
    else:
        # For biases, BN params, embeddings, etc.: no rotation freedom → standard avg
        merged = (1.0 - alpha) * w_base + alpha * w_ft

    return merged

def slerp_lerp(t, w1, w2, eps=1e-8, key=None):
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same shape")
    print("IN SLERP LERP")
    v0 = w1.flatten()
    v1 = w2.flatten()
    r0 = paddle.norm(v0, p=2, keepdim=True)
    r1 = paddle.norm(v1, p=2, keepdim=True)
    v0_norm = F.normalize(v0, axis=0)
    v1_norm = F.normalize(v1, axis=0)
    return w1+2*(w2-w1)
    # if "_mean" in key or "_variance" in key or "bn" in key or "lab" in key or 'identity' in key or 'norm' in key:
    if "conv" in key:
        print("averaging")
        return (r1*w1 + r0*w2)/(r0+r1)
        return (1 - t) * w1 + t * w2
    dot = paddle.sum(v0_norm * v1_norm)
    dot = paddle.clip(dot, -1.0 + eps, 1.0 - eps)

    theta = paddle.acos(dot)
    sin_theta = paddle.sin(theta)

    if sin_theta < eps:
        # fallback to LERP
        return (1 - t) * w1 + t * w2

    s0 = paddle.sin((1 - t) * theta) / sin_theta
    s1 = paddle.sin(t * theta) / sin_theta

    out = s0 * v0 + s1 * v1
    return out.reshape(w1.shape)





def min_jerk(t, w1, w2, eps=1e-8, key=None):
    """
    Smooth interpolation between w1 and w2 using:
      - Minimum-jerk easing on radius (magnitude)
      - Spherical linear interpolation (slerp) on direction
    Works for tensors of any shape (interpreted as vectors in R^d).

    Args:
        t (float): Interpolation parameter in [0, 1]
        w1, w2 (paddle.Tensor): Tensors of same shape
        eps (float): Numerical stability threshold

    Returns:
        paddle.Tensor: Interpolated tensor of same shape as w1/w2
    """
    print("MERGING USING MIN-JERK")
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same shape")

    # Flatten to 1D vectors
    v0 = w1.flatten()  # (D,)
    v1 = w2.flatten()  # (D,)

    # if 'conv' in key:
    #     print("Averaging conv")
    #     return (1-t)*w1 + t*w2
    # Compute magnitudes (radii)
    r0 = paddle.norm(v0, p=2, keepdim=True)  # (1,)
    r1 = paddle.norm(v1, p=2, keepdim=True)  # (1,)
    
    r_ratio = r0.item()/r1.item()
    print("key is: ", key)
    print("R ratio is: ", r0.item()/r1.item())
    # if "_mean" in key or "_variance" in key:
    #     print("Getting Stats LERP and prefer base")
    #     return (0.7) * w1 + 0.3 * w2
    print(r_ratio)
    # if r_ratio >= 1.1:
    #     return w1
    # if r_ratio <= 0.9:
    #     return w2
    # Handle zero vectors: fall back to linear interpolation
    # if (r0.item() < eps) or (r1.item() < eps):
    #     return (1 - t) * w1 + t * w2

    # Normalize to get directions
    u0 = v0 / r0  # (D,)
    u1 = v1 / r1  # (D,)

    # Compute angle between directions
    dot = paddle.sum(u0 * u1)  # scalar
    print("COSIne SIm", dot.item())
    print('*'*30)
    dot = paddle.clip(dot, -1.0 + eps, 1.0 - eps)
    theta = paddle.acos(dot)   # scalar
    sin_theta = paddle.sin(theta)

    # Minimum-jerk easing: s(t) = 10t^3 - 15t^4 + 6t^5
    s = 10*t**3 - 15*t**4 + 6*t**5

    # Interpolate radius
    r_interp = (1 - s) * r0 + s * r1  # (1,)

    # Slerp for direction
    if sin_theta < eps:
        # Directions are parallel or antipodal → use linear interp
        u_interp = (1 - s) * u0 + s * u1
        u_interp = F.normalize(u_interp, p=2, axis=0)
    else:
        s0 = paddle.sin((1 - s) * theta) / sin_theta
        s1 = paddle.sin(s * theta) / sin_theta
        u_interp = s0 * u0 + s1 * u1

    # Combine radius and direction
    out = r_interp * u_interp  # (D,)

    return out.reshape(w1.shape)


def investigate(model1, model2, config_path=None):
    if isinstance(model1, str):
        model1 = build_and_load(model1, config_path)
    if isinstance(model2, str):
        model2 = build_and_load(model2, config_path)

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    stats = {}
    for key in state_dict1:
        if key not in state_dict2:
            print(f"[Warning] Key '{key}' not found in model2. Keeping model1's value.")
            continue

        w1 = state_dict1[key]
        w2 = state_dict2[key]

        v1 = w1.flatten()  # (D,)
        v2 = w2.flatten()  # (D,)

        # Compute magnitudes (radii)
        r1 = paddle.norm(v1, p=2, keepdim=True)  # (1,)
        r2 = paddle.norm(v2, p=2, keepdim=True)  # (1,)
        
        r_ratio = r1.item()/r2.item()
        u1 = v1 / r1  # (D,)
        u2 = v2 / r2  # (D,)

        # Compute angle between directions
        dot = paddle.sum(u1 * u2)  # scalar
        cos_sim = dot.item()

        mse = paddle.mean((v1 - v2) ** 2)

        stats[key] = {"r_ratio": r_ratio, 'cos_sim': cos_sim, 'mse': mse.item()}
    
    return stats


        

def merge_models(model1, model2, config_path=None, t=0.5, method='slerp'):
    """
    Merge two PaddlePaddle models using SLERP for floating-point parameters.
    
    Args:
        model1 (str or paddle.nn.Layer): Path to model1 (.pdparams) or model instance.
        model2 (str or paddle.nn.Layer): Path to model2 (.pdparams) or model instance.
        config_path (str, optional): Path to config file used by `build_and_load`.
        t (float): Interpolation coefficient in [0, 1]. t=0 → model1, t=1 → model2.
    
    Returns:
        paddle.nn.Layer: A new model instance with merged weights.
    """
    # Load models if paths are provided
    if isinstance(model1, str):
        model1 = build_and_load(model1, config_path)
    if isinstance(model2, str):
        model2 = build_and_load(model2, config_path)

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    merged_state = {}

    for key in state_dict1:
        if key not in state_dict2:
            print(f"[Warning] Key '{key}' not found in model2. Keeping model1's value.")
            merged_state[key] = state_dict1[key]
            continue

        w1 = state_dict1[key]
        w2 = state_dict2[key]

        if w1.shape != w2.shape:
            raise ValueError(f"Shape mismatch for parameter '{key}': {w1.shape} vs {w2.shape}")

        # Use Paddle's utility to check floating-point tensors
        if paddle.is_floating_point(w1):
            print("processing: ", key)
            if method=='slerp':
                merged_state[key] = slerp(t, w1, w2)
            elif method=='slerp_lerp':
                merged_state[key] = slerp_lerp(t, w1, w2, key=str(key))
            elif method=='avg':
                merged_state[key] = weight_average(w1, w2, t)
            elif method=='frank':
                merged_state[key] = frank(t, w1, w2, key=key)
            elif method=='pro':
                merged_state[key] = procrustes_weight_merge(w1, w2, t)
            elif method=='jerk':
                merged_state[key] = min_jerk(t, w1, w2, key=str(key))
            elif method=='svd':
                merged_state[key] = svd_based_merge(w1, w2, t)
            elif method=='cos':
                merged_state[key] = cosine_midpoint(w1, w2, t)
            elif method=='dare':
                merged_state[key] = dare_merge(w1, w2, t)
            else:
                print("Method is not supported!")
        else:
            # Non-floating tensors (e.g., int buffers like steps, version, etc.)
            # Do not interpolate—just carry over from model1
            merged_state[key] = w1

    # Reconstruct model instance
    # ⚠️ Assumes `type(model1)` can be instantiated without arguments.
    # If your model requires config, you must adjust this accordingly.
    config = load_config(config_path)
    config = process_config(config)
    merged_model = build_model(config["Architecture"])

    try:
        merged_model.set_state_dict(merged_state)
    except Exception as e:
        print(f"[Error] Failed to load merged state dict: {e}")
        print("Make sure the model class can be re-instantiated without arguments.")
        raise

    return merged_model

def merge_and_save(model1, model2, config_path=None, save_path=None, t=0.5, method='slerp'):
    model_merged = merge_models(model1, model2, config_path, t, method)
    save_merged_model(model_merged, save_path)




















def frank(t, w1, w2, eps=1e-8, key=None):
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same shape")

    if w1.numel() == 0:
        return w1
    with open("conv_imp_01minus.json", "r") as f:
        filter_change_indicators = json.load(f)
    # Extract layer key (e.g., 'backbone.conv1.weight' -> 'backbone.conv1')
    layer_key = None
    if key is not None:
        if key.endswith('.weight') or key.endswith('.bias'):
            layer_key = key.rsplit('.', 1)[0]
        else:
            layer_key = key

    # Check if we have ternary indicators for this layer
    use_frank_merge = False
    indicator = None
    if layer_key in filter_change_indicators:
        indicator = filter_change_indicators[layer_key]
        if len(w1.shape) in [1, 4] and w1.shape[0] == len(indicator):
            use_frank_merge = True

    if use_frank_merge:
        print("USing Frank!")
        # Convert to tensor
        indicator_tensor = paddle.to_tensor(indicator, dtype='int32')  # [-1, 0, 1]
        C = w1.shape[0]

        # Prepare output tensor
        merged = paddle.zeros_like(w1)

        # Identify which filters to take from where
        take_from_w1 = (indicator_tensor == -1)  # [C]
        take_from_w2 = (indicator_tensor == 1)   # [C]
        do_slerp      = (indicator_tensor == 0)  # [C]

        # Helper: reshape masks for broadcasting
        if len(w1.shape) == 4:  # weight: [C, C_in, H, W]
            view_shape = [-1, 1, 1, 1]
        elif len(w1.shape) == 1:  # bias: [C]
            view_shape = [-1]
        else:
            # Fallback to full SLERP if shape unexpected
            use_frank_merge = False

        if use_frank_merge:
            mask_w1 = take_from_w1.astype(w1.dtype).reshape(view_shape)
            mask_w2 = take_from_w2.astype(w1.dtype).reshape(view_shape)
            mask_slerp = do_slerp.astype(w1.dtype).reshape(view_shape)

            # Direct assignment for -1 and 1
            merged += mask_w1 * w1
            merged += mask_w2 * w2

            # For filters where indicator == 0: apply SLERP per-filter
            if paddle.sum(mask_slerp) > 0:
                # We need to SLERP only the "0" filters
                # Extract those filters
                indices_slerp = paddle.nonzero(do_slerp).squeeze(1)  # [K]
                if indices_slerp.numel() > 0:
                    for idx in indices_slerp:
                        idx = idx.item()
                        if len(w1.shape) == 4:
                            w1_filt = w1[idx:idx+1]   # [1, C_in, H, W]
                            w2_filt = w2[idx:idx+1]
                            slerp_filt = slerp_single_filter(t, w1_filt, w2_filt, eps)
                            merged[idx] = slerp_filt[0]
                        else:  # bias
                            w1_f = w1[idx:idx+1]  # [1]
                            w2_f = w2[idx:idx+1]
                            slerp_f = slerp_single_filter(t, w1_f, w2_f, eps)
                            merged[idx] = slerp_f[0]
            return merged

    # Fallback: full tensor SLERP
    return slerp_single_filter(t, w1, w2, eps)


def slerp_single_filter(t, w1, w2, eps=1e-8):
    """Standard SLERP for a single tensor (or batch of 1 filter)."""
    v0 = w1.flatten()
    v1 = w2.flatten()

    norm_v0 = v0.norm()
    norm_v1 = v1.norm()

    if norm_v0 < eps or norm_v1 < eps:
        return (1 - t) * w1 + t * w2

    v0_norm = v0 / norm_v0
    v1_norm = v1 / norm_v1

    dot = paddle.sum(v0_norm * v1_norm)
    dot = paddle.clip(dot, -1.0 + eps, 1.0 - eps)

    theta = paddle.acos(dot)
    sin_theta = paddle.sin(theta)

    if sin_theta < eps:
        return (1 - t) * w1 + t * w2

    s0 = paddle.sin((1 - t) * theta) / sin_theta
    s1 = paddle.sin(t * theta) / sin_theta

    out = s0 * v0 + s1 * v1
    return out.reshape(w1.shape)