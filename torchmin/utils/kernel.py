def tune_bounds(_bounds:tuple[tuple[float, float]], target_indices:tuple[int], dim:int)->tuple[tuple[float, float], ...]:
    bounds = []
    for idx, i in enumerate(range(dim)):
        if i in target_indices:
            bounds.append(_bounds[idx])
        else:
            bounds.append(None)
    return tuple(bounds)