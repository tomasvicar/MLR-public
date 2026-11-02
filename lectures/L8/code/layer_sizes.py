"""Output sizes and parameter counts, layer by layer.

Code example for the eighth lecture (Convolutional networks 1). The formulas from the
slides:

    convolution W_(l+1) = (W_l - F_l + 2 P_l) / S_l + 1,   D_(l+1) = K_l
                parameters  F_l * F_l * D_l * K_l  weights  +  K_l  biases
    pooling     W_(l+1) = (W_l - F_l) / S_l + 1,           D_(l+1) = D_l
                parameters  0
    FC          input (vectorized) W_l * H_l * D_l  ->  K_l
                parameters  W_l * H_l * D_l * K_l  +  K_l

The script computes them for the whole architecture from the lecture and
prints the table shown on the slide "Sizes and parameters, layer by layer".

    uv run python lectures/L8/code/layer_sizes.py
"""

# The architecture from the last block of the lecture: input 32x32x3, twice a
# pair of convolutions with ReLU closed by max pooling, two FC layers at the end.
ARCHITECTURE = [
    ("conv", {"F": 3, "S": 1, "P": 1, "K": 16}),
    ("relu", {}),
    ("conv", {"F": 3, "S": 1, "P": 1, "K": 16}),
    ("relu", {}),
    ("pool", {"F": 2, "S": 2}),
    ("conv", {"F": 3, "S": 1, "P": 1, "K": 32}),
    ("relu", {}),
    ("conv", {"F": 3, "S": 1, "P": 1, "K": 32}),
    ("relu", {}),
    ("pool", {"F": 2, "S": 2}),
    ("fc", {"K": 64}),
    ("relu", {}),
    ("fc", {"K": 10}),
    ("softmax", {}),
]


def exact_div(numerator, denominator, label):
    """The output size has to come out as an integer — otherwise the architecture is wrong."""
    if numerator % denominator:
        raise ValueError(f"{label}: {numerator} / {denominator} is not an integer")
    return numerator // denominator


def walk(input_shape, layers=ARCHITECTURE):
    """Return a list of (layer name, output shape, number of parameters)."""
    w, h, d = input_shape
    rows = [("input", (w, h, d), 0)]
    for kind, p in layers:
        if kind == "conv":
            w = exact_div(w - p["F"] + 2 * p["P"], p["S"], "conv W") + 1
            h = exact_div(h - p["F"] + 2 * p["P"], p["S"], "conv H") + 1
            params = p["F"] * p["F"] * d * p["K"] + p["K"]
            d = p["K"]
            name = f"conv {p['F']}x{p['F']}, K={p['K']}, S={p['S']}, P={p['P']}"
        elif kind == "pool":
            w = exact_div(w - p["F"], p["S"], "pool W") + 1
            h = exact_div(h - p["F"], p["S"], "pool H") + 1
            params = 0
            name = f"max pool {p['F']}x{p['F']}, S={p['S']}"
        elif kind == "fc":
            params = w * h * d * p["K"] + p["K"]
            w, h, d = 1, 1, p["K"]
            name = f"FC, K={p['K']}"
        else:                                   # ReLU, softmax — no change
            params = 0
            name = kind
        rows.append((name, (w, h, d), params))
    return rows


def table(rows):
    width = max(len(r[0]) for r in rows)
    lines = [f"{'layer':<{width}}  {'output':>14}  {'parameters':>12}"]
    for name, (w, h, d), params in rows:
        shape = f"{w}x{h}x{d}" if w > 1 else f"{d}"
        lines.append(f"{name:<{width}}  {shape:>14}  {params:>12,}")
    total = sum(r[2] for r in rows)
    lines.append(f"{'total':<{width}}  {'':>14}  {total:>12,}")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = walk((32, 32, 3))
    print(table(rows))

    total = sum(r[2] for r in rows)
    fc = max(r[2] for r in rows)
    print(f"\nthe most expensive layer has {fc:,} parameters, "
          f"i.e. {100 * fc / total:.0f} % of the whole network")

    # The output size has to come out as an integer:
    try:
        walk((64, 64, 3), [("conv", {"F": 5, "S": 2, "P": 2, "K": 8})])
    except ValueError as err:
        print(f"\ninvalid combination of hyperparameters: {err}")
