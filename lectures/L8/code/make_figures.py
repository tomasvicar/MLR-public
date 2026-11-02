"""Create all the illustrations for the eighth lecture (Convolutional networks 1 — basics).

The original lecture exists only as a PowerPoint PDF
`moodle_2025/extracted/prednasky/2025/07_MLR_deep_learning_1_basics.pdf`
(author Jiří Chmelík), so there is no Colab notebook to draw from. All the
figures are redrawn from scratch following the rendered pages of the PDF: they
carry the same message, the same labels and the same colour convention (input
image orange, filter blue, activation maps coloured, arrows orange), only in
the typography of lectures L1–L4 and L6–L7.

Not a single raster is taken out of the PDF — not even a photograph, because
there is not a single necessary one in the whole PDF.

What is really computed instead of drawn by eye:

* the decision regions of networks with no hidden layer / with one / with two
  hidden layers are really trained classifiers (`MLPClassifier`),
* the nonlinear separating surface in 3D is the 0.5 level set of a really
  trained network, found by bisection,
* the numerical convolution example, the max/average pooling example from the
  slide and all the output sizes and parameter counts are computed and guarded
  by an `assert` (the formulas and the architecture are in `layer_sizes.py`),
* the sizes and depths of the boxes in the architecture diagram are taken
  from `layer_sizes.walk()` (front face proportional to W x H, depth to the
  number of maps),
* translation equivariance (a shifted input gives the shifted map) and the
  comparison of max pooling with a strided convolution are computed on small
  integer maps and checked exactly,
* the parameter counts of the PyTorch slide are checked against `torch.nn`
  (when torch is installed),
* the learned filters of the first layer and their activation maps are really
  trained by the small convolutional network from `cnn_filters.py`.

The isometric diagrams of a convolutional layer (pages 9–13, 15 and 22 of the
PDF) are drawn by a single parametric box generator `box()` / `stack()`; the
five stages of the gradually built figure are five configurations of the same
function.

The `figsize` of each figure is chosen according to how large the figure will
be on the slide: for the SVGs the `width=` attribute in the slides is close to
the natural width of the SVG in pixels, so 1 pt of type ≈ 1.33 px on the slide
and the smallest labels (9 pt) come out at ~12 px. The GIFs are rendered at
110 dpi (1 pt ≈ 1.53 px) and shown slightly downscaled.
When a figure is enlarged or shrunk on the slide, it is `figsize` that gets
recomputed, not `fontsize`.

    uv run python lectures/L8/code/make_figures.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from PIL import Image
from sklearn.datasets import load_sample_image
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cnn_filters                                            # noqa: E402
import layer_sizes                                      # noqa: E402

# A 3-D scatter with set_rasterized(True) warns that the rasterization "will be
# ignored", yet in the SVG it is applied (the 3-D figure is 233 kB with it,
# 287 kB without); the warning is repeated for every GIF frame, so silence it.
warnings.filterwarnings("ignore", message="Rasterization of .*Path3DCollection")

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams["svg.hashsalt"] = "mpa-mlr-l8"
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "figure.max_open_warning": 0,      # the GIF frames are rendered together
    # figure annotations in navy, as in the seventh lecture (author, 26 October
    # 2026); tick labels and axis labels stay black
    "text.color": "#12355b",
})
SVG = {"Date": None}

# Palette shared with lectures L1–L4 and L6–L7.
BLUE = "#12355b"
TURQUOISE = "#007f86"
RED = "#c73e1d"
ORANGE = "#e9a23b"
GREEN = "#3a8a52"
PURPLE = "#7b5ea7"
GREY = "#8895a0"
GOLD = "#f2c14e"
LIGHT_BLUE = "#d9eef7"
LIGHT_ORANGE = "#fbe6c4"
LIGHT_GREY = "#eef2f4"
LIGHT_GREEN = "#d8ecdd"

# The colour roles of the original slides (input orange, filter blue, maps
# coloured), mapped to the course palette (author, 26 October 2026). The 2025
# values were orange #e0952f, red #e02b1d, green #1f9c3f, yellow #f5e017; blue
# and turquoise had no counterpart in the palette and are kept.
PDF_ORANGE = ORANGE
PDF_BLUE = "#2a72b5"
PDF_TURQUOISE = "#5ec8d8"
PDF_RED = RED
PDF_GREEN = GREEN
PDF_YELLOW = GOLD

ISO = (-0.50, 0.40)      # depth direction of the boxes: back to the left and up


def thousands(n) -> str:
    """Thousands separator as in the slides — a thin space, not a comma."""
    return f"{n:,}".replace(",", "\u2009")


def save(fig, name: str) -> None:
    # dpi only applies to artists marked set_rasterized(True) — the point
    # clouds, the filled decision regions and the 3D surface, which would make
    # hundreds of kilobytes of vector paths (same as in the L7 generator)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata=SVG,
                dpi=200)
    plt.close(fig)


def render_frame(fig, dpi=100):
    """Render the figure into a raster for one frame of the GIF."""
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return img


def render_frames(figures, dpi=110, pad=0.06):
    """Render the frames of a GIF with one shared tight bounding box.

    The union of the tight boxes of all the frames is used — the analogue of
    `bbox_inches="tight"` of the SVG — so that labels outside the axes are not
    cut off and the content does not jump between frames. dpi 110 makes the
    GIF a little wider than its `width=` on the slide, so it is shown
    downscaled, never blurred by upscaling.
    """
    import io
    from matplotlib.transforms import Bbox
    boxes = [f.get_tightbbox(f.canvas.get_renderer()) for f in figures]
    union = Bbox.union(boxes).padded(pad)
    images = []
    for f in figures:
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=dpi, bbox_inches=union,
                  facecolor="white")
        plt.close(f)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    assert len({im.size for im in images}) == 1
    return images


def save_gif(frames, name: str, first_duration=1500, duration=800,
             last_duration=3500) -> None:
    """Looped GIF: the first frame is the final state, the last one is held long.

    The frames are quantized with a shared palette so that the colours do not
    flicker between frames.
    """
    # MAXCOVERAGE keeps the background pure white; both MEDIANCUT and
    # FASTOCTREE would average white with the antialiased shades into
    # (252–254, …), so the GIF would look like a grey rectangle on a white
    # slide. Pixels are assigned to the palette exactly, in numpy —
    # `Image.quantize(palette=…)` uses an approximate lookup table and turns
    # white into (250, 250, 250).
    palette = frames[0].quantize(colors=255, method=Image.Quantize.MAXCOVERAGE,
                                dither=Image.Dither.NONE)
    colours = np.array(palette.getpalette()[:3 * 255], np.int32).reshape(-1, 3)

    def quantize(img):
        px = np.asarray(img).reshape(-1, 3).astype(np.int32)
        idx = np.empty(len(px), np.uint8)
        for start in range(0, len(px), 16384):
            block = px[start:start + 16384]
            d = ((block[:, None, :] - colours[None, :, :]) ** 2).sum(axis=-1)
            idx[start:start + 16384] = d.argmin(axis=1)
        out = Image.fromarray(idx.reshape(img.size[1], img.size[0]), "P")
        out.putpalette(palette.getpalette())
        return out

    quantized = [quantize(s) for s in frames]
    assert quantized[0].convert("RGB").getpixel((2, 2)) == (255, 255, 255)
    frame_durations = ([first_duration] + [duration] * (len(quantized) - 2)
                     + [last_duration])
    quantized[0].save(OUT / f"{name}.gif", save_all=True, append_images=quantized[1:],
                  duration=frame_durations, loop=0, optimize=False)


def _mix(colour, s, ratio):
    a = np.array(matplotlib.colors.to_rgb(colour))
    b = np.array(matplotlib.colors.to_rgb(s))
    return tuple((1 - ratio) * a + ratio * b)


def arrow(ax, start, stop, colour=BLUE, lw=1.2, style="-", size=11, zorder=3):
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>",
                                 mutation_scale=size, color=colour,
                                 linewidth=lw, linestyle=style, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def block_arrow(ax, start, stop, colour=PDF_ORANGE, size=90, zorder=3):
    """A wide "PowerPoint" arrow with a flat tail and a wide head.

    `arrow()` draws a line with a tip; once the line is made thick enough to
    hold a white label, the tip is barely wider than the shaft and the arrow
    stops looking like an arrow. This shape (`arrowstyle="simple"`) has a head
    whose width is independent of the shaft, just like the arrows in the
    original slides.
    """
    ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="simple",
                                 mutation_scale=size, facecolor=colour,
                                 edgecolor="none", linewidth=0, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def line(ax, start, stop, colour=ORANGE, lw=1.0, style="-", zorder=2, alpha=1.0):
    ax.plot([start[0], stop[0]], [start[1], stop[1]], color=colour, linewidth=lw,
            linestyle=style, zorder=zorder, alpha=alpha,
            solid_capstyle="round")


# =========================================================================
# Parametric generator of isometric boxes (pages 9–13, 15, 22 of the PDF)
# =========================================================================

def _face(ax, points, colour, zorder, lw=1.0):
    ax.add_patch(Polygon(points, closed=True, facecolor=colour,
                         edgecolor="black", linewidth=lw, zorder=zorder,
                         joinstyle="round"))


def box(ax, x, y, w, h, d, colour, zorder=1, lw=1.0, labels=None,
          label_size=11):
    """A box in oblique projection: front face w × h at (x, y), depth d to the back.

    The front, top and left faces are visible — just as in the original slides.
    `labels` is a dictionary {"H": text, "W": text, "D": text}.
    """
    ox, oy = d * ISO[0], d * ISO[1]
    top_face = [(x, y + h), (x + w, y + h), (x + w + ox, y + h + oy),
                (x + ox, y + h + oy)]
    left_face = [(x, y), (x, y + h), (x + ox, y + h + oy), (x + ox, y + oy)]
    front_face = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    _face(ax, top_face, _mix(colour, "white", 0.35), zorder, lw)
    _face(ax, left_face, _mix(colour, "black", 0.12), zorder, lw)
    _face(ax, front_face, colour, zorder + 0.01, lw)
    if labels:
        if "H" in labels:
            ax.text(x + w + 0.6, y + h / 2, labels["H"], ha="left",
                    va="center", fontsize=label_size, zorder=zorder + 1)
        if "W" in labels:
            ax.text(x + w / 2, y - 1.4, labels["W"], ha="center", va="top",
                    fontsize=label_size, zorder=zorder + 1)
        if "D" in labels:
            ax.text(x + ox, y + oy - 1.2, labels["D"], ha="right",
                    va="top", fontsize=label_size, zorder=zorder + 1)


def stack(ax, x, y, w, h, thickness, colours, zorder=1, lw=1.0):
    """A box glued from slabs of the given colours — one slab per activation map."""
    for i, colour in reversed(list(enumerate(colours))):
        ox, oy = i * thickness * ISO[0], i * thickness * ISO[1]
        box(ax, x + ox, y + oy, w, h, thickness, colour,
              zorder=zorder + 0.02 * (len(colours) - i), lw=lw)


MAP_COLOURS = (PDF_BLUE, PDF_RED, PDF_GREEN, PDF_YELLOW, PDF_TURQUOISE)


def _conv_axes(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _input_box(ax, x=24, y=10, w=20, h=46, d=8, labels=True):
    box(ax, x, y, w, h, d, PDF_ORANGE,
          labels={"H": "$H_x$", "W": "$W_x$", "D": "$D_x$"} if labels else None)


def _filter_in_image(ax, x, y, w=7, h=8, d=8, colour=PDF_BLUE):
    box(ax, x, y, w, h, d, colour, zorder=2)


def _rays(ax, x, y, w, h, target, colour="black"):
    for corner in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        line(ax, corner, target, colour=colour, lw=0.8, zorder=3)
    ax.add_patch(Rectangle((target[0] - 1.1, target[1] - 1.1), 2.2, 2.2,
                           facecolor=BLUE, edgecolor="black", lw=0.8,
                           zorder=4))


def conv_stages():
    """The five stages of the convolutional layer diagram (pages 9–13 of the PDF)."""
    # Stages 1–3 share a narrower canvas so that the input box stays in place.
    for stage in (1, 2, 3):
        fig, ax = _conv_axes((7.2, 4.4), (0, 118), (0, 72))
        if stage < 3:
            _input_box(ax)
        else:
            _input_box(ax)
            _filter_in_image(ax, 28, 40)
            _rays(ax, 28, 40, 7, 8, (84, 44))
            ax.text(84, 48.5, r"$z_k(i,j)$", ha="center", va="bottom",
                    fontsize=12)
        if stage == 2:
            # The depth of the filter equals the depth of the input
            # (D_w = D_x), so it is drawn with the same d = 8 as the input
            # slab; the original draws the filter thinner, so the figure
            # contradicted its own text.
            box(ax, 72, 31, 9, 10, 8, PDF_BLUE,
                  labels={"H": "$H_w$", "W": "$W_w$", "D": "$D_w$"})
        save(fig, f"konvoluce-{stage}")

    # Stages 4 and 5 need a wider canvas.
    for stage in (4, 5):
        fig, ax = _conv_axes((11.0, 3.7), (0, 232), (0, 78))
        _input_box(ax, x=20, y=8, w=17, h=40, d=7, labels=False)
        filter_colour = PDF_BLUE if stage == 4 else PDF_TURQUOISE
        _filter_in_image(ax, 23, 32, w=6, h=7, d=7, colour=filter_colour)
        label = r"$z_1(i,j)$" if stage == 4 else r"$z_K(i,j)$"
        _rays(ax, 23, 32, 6, 7, (68, 34))
        ax.text(68, 38, label, ha="center", va="bottom", fontsize=12)

        arrow(ax, (82, 30), (104, 30), colour=PDF_ORANGE, lw=8, size=26)
        ax.text(93, 34, "convolution", ha="center", va="bottom", fontsize=11)

        if stage == 4:
            box(ax, 124, 8, 16, 38, 2, PDF_BLUE,
                  labels={"H": r"$H_Z=H_x-H_w+1$",
                           "W": r"$W_Z=W_x-W_w+1$"})
            ax.text(124, 54, r"$Z_1$ — 2D activation map",
                    ha="left", va="bottom", fontsize=11)
            ax.text(122, 49, r"$D_Z=1$", ha="left", va="bottom",
                    fontsize=11)
        else:
            for i, colour in enumerate(MAP_COLOURS):
                box(ax, 118 + 7 * i, 8, 16, 38, 2, colour,
                      zorder=1 + 0.05 * i)
            ax.text(140, 54,
                    r"individual maps $Z_1 \dots Z_K$",
                    ha="center", va="bottom", fontsize=11)
            arrow(ax, (162, 30), (180, 30), colour=PDF_ORANGE, lw=8,
                  size=26)
            ax.text(171, 34, "stack", ha="center", va="bottom", fontsize=11)
            stack(ax, 200, 8, 16, 38, 2, MAP_COLOURS[::-1])
            ax.text(216.6, 27, r"$H_A$", ha="left", va="center",
                    fontsize=11)
            ax.text(208, 6.6, r"$W_A$", ha="center", va="top",
                    fontsize=11)
            ax.text(196, 54, "new $D_A=K$", ha="center",
                    va="bottom", fontsize=11)
        save(fig, f"konvoluce-{stage}")


def activation_layer():
    """Page 15 of the PDF: two identical stacks with the function f(.) between them."""
    fig, ax = _conv_axes((7.6, 3.0), (0, 152), (0, 60))
    stack(ax, 14, 6, 16, 40, 2, MAP_COLOURS[::-1])
    stack(ax, 118, 6, 16, 40, 2, MAP_COLOURS[::-1])
    centre = (76, 27)
    for y in (8, 44):
        line(ax, (30, y), (centre[0] - 8, centre[1]), colour="black", lw=0.9)
        line(ax, (118, y), (centre[0] + 8, centre[1]), colour="black", lw=0.9)
    ax.add_patch(Circle(centre, 9, facecolor=PDF_ORANGE, edgecolor="black",
                        lw=1.0, zorder=5))
    ax.text(centre[0], centre[1], r"$f(\cdot)$", ha="center", va="center",
            color="white", fontsize=13, zorder=6)
    ax.text(22, 50, r"$A_l$", ha="center", va="bottom", fontsize=12)
    ax.text(126, 50, r"$f(A_l)$", ha="center", va="bottom", fontsize=12)
    save(fig, "aktivacni-vrstva")


# =========================================================================
# Page 4 of the PDF — decision regions of shallow networks
# =========================================================================

def _network_sketch(ax, x0, y0, layers, width=13.0, height=11.0):
    """A small diagram of a fully connected network for a table cell."""
    positions = []
    for i, n in enumerate(layers):
        xx = x0 + (width * i / max(len(layers) - 1, 1))
        positions.append([(xx, y0 + height * (j + 0.5) / n) for j in range(n)])
    for a, b in zip(positions, positions[1:]):
        for p in a:
            for q in b:
                line(ax, p, q, colour=GREY, lw=0.5, zorder=1)
    for i, layer in enumerate(positions):
        colour = "white" if i < len(positions) - 1 else LIGHT_BLUE
        for p in layer:
            ax.add_patch(Circle(p, 1.1, facecolor=colour, edgecolor="black",
                                lw=0.7, zorder=2))


def _spirals(rng, n=300):
    """Two interleaved spirals — "classes with meshed regions"."""
    t = np.sqrt(rng.uniform(0.05, 1.0, n)) * 2.6 * np.pi
    r = t / (2.6 * np.pi)
    points = []
    for shift in (0.0, np.pi):
        x = r * np.cos(t + shift) + rng.normal(0, 0.03, n)
        y = r * np.sin(t + shift) + rng.normal(0, 0.03, n)
        points.append(np.column_stack([x, y]))
    x = np.vstack(points)
    y = np.repeat([0, 1], n)
    return x, y


def decision_regions():
    """Decision regions of networks with 0, 1 and 2 hidden layers (after Lippmann 1987)."""
    rng = np.random.default_rng(3)
    centres = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], float)
    xor_x = np.repeat(centres, 120, axis=0) + rng.normal(0, 0.32, (480, 2))
    xor_y = np.repeat([0, 0, 1, 1], 120)
    mesh_x, mesh_y = _spirals(rng)

    models = [
        ("no hidden layer", "half-plane bounded\nby a hyperplane",
         (2, 1), lambda: LogisticRegression()),
        # Two hidden neurons as in Lippmann: the output is then an
        # intersection of two half-planes, that is a convex strip — exactly
        # what the caption of the row claims. With a larger hidden layer the
        # region comes out non-convex and the figure would contradict the
        # text next to it.
        ("one hidden layer", "convex open\nor closed regions",
         (2, 2, 1), lambda: MLPClassifier((2,), max_iter=6000, random_state=3,
                                          alpha=1e-3)),
        ("two hidden layers", "arbitrary regions\n(limited by the\nnumber of nodes)",
         (2, 8, 8, 1), lambda: MLPClassifier((8, 8), max_iter=12000,
                                             random_state=0, alpha=1e-4)),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(8.0, 4.4),
                            gridspec_kw={"width_ratios": [1.15, 1.25, 1, 1],
                                         "wspace": 0.12, "hspace": 0.12})
    titles = ["structure", "type of decision regions",
               "XOR problem", "interleaved classes"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=9.5, pad=8)

    for i, (name, label, layers, factory) in enumerate(models):
        ax = axes[i, 0]
        ax.set(xlim=(0, 20), ylim=(0, 16))
        ax.set_aspect("equal")
        ax.axis("off")
        _network_sketch(ax, 3, 1.4, layers)
        ax.text(10, 15.8, name, ha="center", va="top", fontsize=9)

        ax = axes[i, 1]
        ax.axis("off")
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=10,
                transform=ax.transAxes)

        for j, (x, y, extent) in enumerate([(xor_x, xor_y, 2.1),
                                            (mesh_x, mesh_y, 1.25)]):
            model = factory().fit(x, y)
            grid = np.linspace(-extent, extent, 220)
            xx, yy = np.meshgrid(grid, grid)
            z = model.predict(np.column_stack([xx.ravel(), yy.ravel()]))
            ax = axes[i, 2 + j]
            ax.contourf(xx, yy, z.reshape(xx.shape), levels=[-0.5, 0.5, 1.5],
                        colors=["white", LIGHT_GREEN]).set_rasterized(True)
            ax.contour(xx, yy, z.reshape(xx.shape), levels=[0.5],
                       colors=[GREEN], linewidths=1.0).set_rasterized(True)
            ax.scatter(*x[y == 0].T, s=2.5, color=RED,
                       zorder=3).set_rasterized(True)
            ax.scatter(*x[y == 1].T, s=2.5, color=BLUE,
                       zorder=3).set_rasterized(True)
            ax.set(xticks=[], yticks=[], xlim=(-extent, extent),
                   ylim=(-extent, extent))
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_color(GREY)

            # The figure has to agree with the caption of the row next to it:
            # with no hidden layer and with one hidden layer the green region
            # has to be convex (with two hidden layers it need not be).
            if i < 2:
                inside = np.column_stack([xx.ravel(), yy.ravel()])[z == 1]
                sample = np.random.default_rng(0).integers(
                    0, len(inside), (3000, 2))
                segment_midpoints = (inside[sample[:, 0]]
                                 + inside[sample[:, 1]]) / 2
                assert model.predict(segment_midpoints).all(), (
                    f"the decision region in row {i} is not convex")
    save(fig, "rozhodovaci-oblasti")


def nonlinear_surface():
    """The 0.5 level set of a really trained network between two 3D point clouds."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (2400, 3))
    boundary = 0.45 * np.sin(2.4 * x[:, 0]) + 0.35 * np.cos(2.2 * x[:, 1])
    y = (x[:, 2] > boundary).astype(int)
    # One hidden layer, because the slide is "What one hidden layer can and
    # cannot do" (until October 2026 this was a 24-24 network with two).
    model = MLPClassifier((64,), max_iter=4000, random_state=0).fit(x, y)
    assert model.score(x, y) > 0.98, model.score(x, y)

    # The level set is found by bisection along z.
    grid = np.linspace(-1, 1, 44)
    xx, yy = np.meshgrid(grid, grid)
    lo = np.full(xx.size, -1.0)
    hi = np.full(xx.size, 1.0)
    for _ in range(24):
        centre = (lo + hi) / 2
        p = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel(),
                                                 centre]))[:, 1]
        lo = np.where(p < 0.5, centre, lo)
        hi = np.where(p < 0.5, hi, centre)
    zz = ((lo + hi) / 2).reshape(xx.shape)

    sample = rng.choice(len(x), 700, replace=False)
    # the bisection has converged and the surface really is the 0.5 level
    p_surface = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel(),
                                                     zz.ravel()]))[:, 1]
    inside = np.abs(zz.ravel()) < 0.999        # the level set lies inside the cube
    assert np.abs(p_surface[inside] - 0.5).max() < 0.01
    # ... and it follows the true boundary the data were generated from
    true = 0.45 * np.sin(2.4 * xx) + 0.35 * np.cos(2.2 * yy)
    assert np.abs(zz - true).max() < 0.2, np.abs(zz - true).max()

    save(_surface_figure(x, y, xx, yy, zz, sample, azim=-58),
         "nelinearni-plocha-3d")

    # GIF: one full turn around the vertical axis, so that the bent shape of
    # the surface is seen in depth; it starts and ends at the view of the SVG.
    azims = -58 + np.arange(0, 361, 15)
    frames = render_frames([_surface_figure(x, y, xx, yy, zz, sample, -58)]
                           + [_surface_figure(x, y, xx, yy, zz, sample, a)
                              for a in azims], dpi=100)
    save_gif(frames, "nelinearni-plocha-3d", duration=110)


def _surface_figure(x, y, xx, yy, zz, sample, azim):
    """The level set and the two point clouds seen from the azimuth `azim`."""
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="coolwarm", alpha=0.85, linewidth=0.2,
                    edgecolor=_mix(BLUE, "white", 0.55), rstride=2,
                    cstride=2).set_rasterized(True)
    for cls, colour in ((0, BLUE), (1, RED)):
        m = sample[y[sample] == cls]
        ax.scatter(x[m, 0], x[m, 1], x[m, 2], s=2.5, color=colour, alpha=0.6,
                   depthshade=False).set_rasterized(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color("black")
    ax.grid(False)
    ax.set(xticks=[], yticks=[], zticks=[], xlim=(-1, 1), ylim=(-1, 1),
           zlim=(-1, 1))
    ax.set_xlabel("$x_1$", labelpad=-12)
    ax.set_ylabel("$x_2$", labelpad=-12)
    ax.set_zlabel("$x_3$", labelpad=-12)
    ax.view_init(elev=18, azim=azim)
    ax.set_box_aspect((1, 1, 0.9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


# =========================================================================
# Page 5 of the PDF — shallow versus deep network
# =========================================================================

def _network(ax, x0, width, layers, colours, radius=1.35, height=34, y0=6):
    positions = []
    for i, n in enumerate(layers):
        xx = x0 + width * i / (len(layers) - 1)
        spacing = height / max(layers)
        centre = y0 + height / 2
        positions.append([(xx, centre + spacing * (j - (n - 1) / 2))
                       for j in range(n)])
    for a, b in zip(positions, positions[1:]):
        for p in a:
            for q in b:
                line(ax, p, q, colour=GREY, lw=0.25, zorder=1, alpha=0.8)
    for layer, colour in zip(positions, colours):
        for p in layer:
            ax.add_patch(Circle(p, radius, facecolor=colour,
                                edgecolor="black", lw=0.5, zorder=3))
    for p in positions[0]:
        arrow(ax, (p[0] - 5, p[1]), (p[0] - radius - 0.4, p[1]),
              colour="black", lw=0.7, size=7)
    for p in positions[-1]:
        arrow(ax, (p[0] + radius + 0.4, p[1]), (p[0] + 5, p[1]),
              colour="black", lw=0.7, size=7)


def shallow_vs_deep():
    # ≈ 860 px wide on the slide since the visual pass (704 px before)
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.set(xlim=(0, 200), ylim=(0, 52))
    ax.set_aspect("equal")
    ax.axis("off")
    red, orange, blue = RED, ORANGE, PDF_BLUE
    _network(ax, 16, 42, (5, 7, 4), [red, orange, blue])
    _network(ax, 104, 78, (5, 8, 8, 8, 8, 4),
         [red, orange, orange, orange, orange, blue])
    ax.text(37, 48, "shallow neural network", ha="center", fontsize=12,
            fontweight="bold")
    ax.text(143, 48, "deep neural network", ha="center", fontsize=12,
            fontweight="bold")
    for i, (colour, label) in enumerate([(red, "input layer"),
                                        (orange, "hidden layer"),
                                        (blue, "output layer")]):
        x = 52 + 34 * i
        ax.add_patch(Circle((x, 1.5), 1.35, facecolor=colour,
                            edgecolor="black", lw=0.5))
        ax.text(x + 3, 1.5, label, ha="left", va="center", fontsize=10)
    save(fig, "melka-vs-hluboka")


# =========================================================================
# Pages 7 and 18 of the PDF — ANN versus CNN connections and the FC layer
# =========================================================================

def _pixel_image():
    """The small pixel image that serves as the input in the original slides."""
    img = np.zeros((13, 11), int)
    img[1, 2:9] = 1
    img[2, 1], img[2, 9] = 1, 1
    img[3:5, 0], img[3:5, 10] = 1, 1
    img[4, 2:4] = 1
    img[4, 7:9] = 1
    img[5, 0], img[5, 10] = 1, 1
    img[6, 5] = 1
    img[7, 0], img[7, 10] = 1, 1
    img[8, 2] = 1
    img[8, 8] = 1
    img[9, 3:8] = 1
    img[10, 1], img[10, 9] = 1, 1
    img[11, 2:9] = 1
    return img


def _plane(ax, x0, y0, width, height, slant=0.30, grid=None):
    """Draw the pixel image as an oblique plane; return the node position function."""
    img = _pixel_image()
    n_rows, n_cols = img.shape
    dx, dy = width / n_cols, height / n_rows

    def point(col, row):
        return (x0 + col * dx + (n_rows - row) * slant * dx,
                y0 + (n_rows - row) * dy)

    for r in range(n_rows):
        for s in range(n_cols):
            points = [point(s, r + 1), point(s + 1, r + 1), point(s + 1, r), point(s, r)]
            ax.add_patch(Polygon(points, closed=True,
                                 facecolor="black" if img[r, s] else GOLD,
                                 edgecolor=GREY, linewidth=0.25, zorder=2))
    if grid is not None:
        for r, s in grid:
            points = [point(s, r + 1), point(s + 1, r + 1), point(s + 1, r), point(s, r)]
            ax.add_patch(Polygon(points, closed=True, facecolor="none",
                                 edgecolor=PDF_BLUE, linewidth=1.4, zorder=4))
    return point, n_rows, n_cols


def _neurons(ax, x, ys, colour, radius=1.5, arrows=False):
    points = []
    for y in ys:
        ax.add_patch(Circle((x, y), radius, facecolor=colour,
                            edgecolor="black", lw=0.6, zorder=5))
        points.append((x, y))
        if arrows:
            arrow(ax, (x + radius + 0.3, y), (x + radius + 6, y),
                  colour=ORANGE, lw=0.9, size=8)
    return points


def ann_connections():
    """Page 7, left: every neuron is connected to the whole input."""
    fig, ax = plt.subplots(figsize=(6.5, 4.33))
    ax.set(xlim=(0, 108), ylim=(0, 72))
    ax.set_aspect("equal")
    ax.axis("off")
    point, n_rows, n_cols = _plane(ax, 6, 6, 30, 56)
    neurons = _neurons(ax, 72, np.linspace(10, 66, 8), ORANGE)
    for r in range(0, n_rows, 2):
        for s in range(0, n_cols, 2):
            for q in neurons:
                line(ax, point(s + 0.5, r + 0.5), q, colour=ORANGE, lw=0.25,
                     alpha=0.5, zorder=3)
    ax.text(72, 70, "every neuron sees\nthe whole image", ha="center",
            va="bottom", fontsize=10)
    save(fig, "ann-propojeni")


def cnn_connections():
    """Page 7, right: a neuron is connected only to a local window."""
    fig, ax = plt.subplots(figsize=(6.5, 4.33))
    ax.set(xlim=(0, 108), ylim=(0, 72))
    ax.set_aspect("equal")
    ax.axis("off")
    window = [(r, s) for r in range(5, 8) for s in range(4, 7)]
    point, n_rows, n_cols = _plane(ax, 6, 6, 30, 56, grid=window)
    blue_neurons = _neurons(ax, 72, [44, 30], PDF_BLUE)
    for r, s in window:
        for q in blue_neurons:
            line(ax, point(s + 0.5, r + 0.5), q, colour=PDF_BLUE, lw=0.7,
                 zorder=3)
    for y in (60, 16):
        ax.add_patch(Circle((72, y), 1.5, facecolor=ORANGE,
                            edgecolor="black", lw=0.6, zorder=5))
    for y in (52, 38, 24):
        ax.add_patch(Circle((72, y), 0.5, facecolor="black", lw=0, zorder=5))
    ax.text(72, 70, "one neuron sees one window,\nall of them share the weights",
            ha="center", va="bottom", fontsize=10)
    save(fig, "cnn-propojeni")


def fc_layer():
    """Page 18: the vectorized input of a fully connected layer."""
    fig, ax = plt.subplots(figsize=(6.0, 4.25))
    ax.set(xlim=(0, 96), ylim=(0, 68))
    ax.set_aspect("equal")
    ax.axis("off")
    point, n_rows, n_cols = _plane(ax, 4, 6, 26, 50)
    ys = [60, 52, 44, 22, 14]
    neurons = _neurons(ax, 64, ys, ORANGE, arrows=True)
    for r in range(0, n_rows, 2):
        for s in range(0, n_cols, 2):
            for q in neurons:
                line(ax, point(s + 0.5, r + 0.5), q, colour=ORANGE, lw=0.3,
                     alpha=0.55, zorder=3)
    for y in (38, 33, 28):
        ax.add_patch(Circle((64, y), 0.6, facecolor="black", lw=0, zorder=6))
    ax.text(64, 66, "$K$ neurons", ha="center", va="bottom", fontsize=10)
    save(fig, "fc-vrstva")


def weights_ann_cnn():
    """Page 7: how many weights a layer has when fully connected and when convolutional."""
    input_shape = (200, 200, 3)
    n_inputs = input_shape[0] * input_shape[1] * input_shape[2]
    ann_neuron = n_inputs
    ann_layer = n_inputs * 50
    cnn_neuron = 3 * 3 * 3
    cnn_two_maps = cnn_neuron * 2
    cnn_layer = cnn_neuron * 50
    assert (n_inputs, ann_layer, cnn_neuron, cnn_two_maps) == (
        120_000, 6_000_000, 27, 54)

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    labels = ["FC neuron", "FC layer\n(50 neurons)", "conv filter\n$3\\times3\\times3$",
               "conv layer\n(50 filters)"]
    values = [ann_neuron, ann_layer, cnn_neuron, cnn_layer]
    colours = [ORANGE, ORANGE, PDF_BLUE, PDF_BLUE]
    bars = ax.bar(labels, values, color=colours, edgecolor="black",
                   linewidth=0.6, width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(10, 3e7)
    ax.set_ylabel("weights (log scale)")
    for bar, h in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.35, thousands(h),
                ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=10)
    fig.tight_layout()
    save(fig, "vahy-ann-cnn")


# =========================================================================
# Convolution, stride, padding, pooling — numerical examples
# =========================================================================

def _number_grid(ax, x0, y0, values, step, colours=None, fontsize=11,
                  number_format="{:.0f}", lw=0.8, alpha=1.0):
    n_rows, n_cols = values.shape
    for r in range(n_rows):
        for s in range(n_cols):
            x, y = x0 + s * step, y0 - (r + 1) * step
            colour = "white" if colours is None else colours[r][s]
            ax.add_patch(Rectangle((x, y), step, step, facecolor=colour,
                                   edgecolor="black", linewidth=lw,
                                   alpha=alpha, zorder=1))
            if np.isnan(values[r, s]):           # a cell not computed yet
                continue
            ax.text(x + step / 2, y + step / 2,       # with a true minus sign
                    number_format.format(values[r, s]).replace("-", "−"),
                    ha="center",
                    va="center", fontsize=fontsize, zorder=2,
                    # navy digits are too faint on the course green
                    color="white" if colour == GREEN else None)


CONV_X = np.array([[3, 0, 1, 2, 7],
                   [1, 5, 8, 9, 3],
                   [2, 7, 2, 5, 1],
                   [0, 1, 3, 1, 7],
                   [4, 2, 1, 6, 2]], float)
CONV_W = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], float)


def _conv_example_frame(output, step, figsize=(9.0, 3.3), full=False):
    """One state of the numerical example: the window at position `step`, the
    outputs 0 … step filled in (all of them with `full`) and the dot product
    of this very position written out."""
    x, w = CONV_X, CONV_W
    i, j = divmod(step, 3)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlim=(0, 180), ylim=(-6, 60))
    ax.set_aspect("equal")
    ax.axis("off")

    colours = [["white"] * 5 for _ in range(5)]
    for r in range(3):
        for s in range(3):
            colours[i + r][j + s] = LIGHT_ORANGE
    _number_grid(ax, 4, 54, x, 9, colours)
    ax.add_patch(Rectangle((4 + 9 * j, 54 - 9 * (i + 3)), 27, 27,
                           facecolor="none", edgecolor=RED, linewidth=1.8,
                           zorder=3))
    ax.text(26.5, 56, "input $\\mathbf{x}$  ($5\\times5$)", ha="center",
            va="bottom", fontsize=11)

    _number_grid(ax, 60, 45, w, 9, [[LIGHT_BLUE] * 3 for _ in range(3)])
    ax.text(73.5, 47, "filter $\\mathbf{w}$  ($3\\times3$)", ha="center",
            va="bottom", fontsize=11)

    # Outputs that have not been computed yet stay empty.
    shown = output if full else np.where(np.arange(9).reshape(3, 3) <= step,
                                         output, np.nan)
    out_colours = [["white"] * 3 for _ in range(3)]
    out_colours[i][j] = LIGHT_ORANGE
    _number_grid(ax, 148, 45, shown, 9, out_colours)
    ax.add_patch(Rectangle((148 + 9 * j, 45 - 9 * (i + 1)), 9, 9,
                           facecolor="none", edgecolor=RED, linewidth=1.8,
                           zorder=3))
    ax.text(161.5, 47, r"output $Z$  ($3\times3$)", ha="center",
            va="bottom", fontsize=11)

    arrow(ax, (100, 26), (142, 26), colour=PDF_ORANGE, lw=6, size=20)
    ax.text(121, 29.5, "cross-correlation\n(libraries call it convolution)",
            ha="center", va="bottom", fontsize=10)

    # The dot product written out: nine terms, row by row of the window.
    window = x[i:i + 3, j:j + 3]
    terms = []
    for a, b in zip(window.ravel(), w.ravel()):
        terms.append(f"{a:.0f}\\cdot{'(-1)' if b < 0 else f'{b:.0f}'}")
    ax.text(90, -1.5,
            r"$z(" + f"{i},{j}" + r")=" + "+".join(terms)
            + f"={output[i, j]:.0f}$",
            ha="center", va="center", fontsize=10.5)
    return fig


def conv_example():
    """Numerical convolution example: input 5x5, filter 3x3, one element written out.

    The static SVG and the first frame of the GIF show the whole output map
    with the first window written out — the element the slide computes in
    words; the GIF then walks through all nine window positions.
    """
    x, w = CONV_X, CONV_W
    output = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            output[i, j] = np.sum(x[i:i + 3, j:j + 3] * w)
    assert output.shape == (5 - 3 + 1, 5 - 3 + 1)
    assert output[0, 0] == 3 + 1 + 2 - (1 + 8 + 2)          # -5
    # The filter is "left column minus right column" — checked for all elements.
    assert np.array_equal(output, x[:3, :3] + x[1:4, :3] + x[2:, :3]
                          - (x[:3, 2:] + x[1:4, 2:] + x[2:, 2:]))

    assert np.array_equal(output, [[-5, -4, 0], [-10, -2, 2], [0, -2, -4]])
    fig = _conv_example_frame(output, 0, full=True)
    save(fig, "konvoluce-priklad")

    frames = [render_frame(_conv_example_frame(output, 0, full=True))]
    for step in range(9):
        frames.append(render_frame(_conv_example_frame(output, step)))
    save_gif(frames, "konvoluce-priklad", duration=1000)
    return output


STRIDE_SETTINGS = [(5, 3, 1, 0), (5, 3, 2, 0), (5, 3, 1, 1)]   # (W, F, S, P)


def _stride_positions(w_in, f, s, p):
    """Top-left corners (row, column) of all window positions, row by row."""
    n = w_in + 2 * p
    w_out = (w_in - f + 2 * p) // s + 1
    assert (w_in - f + 2 * p) % s == 0
    corners = [(r * s, c * s) for r in range(w_out) for c in range(w_out)]
    # the last window must end exactly at the (padded) border
    assert corners[-1][0] + f == n and corners[-1][1] + f == n
    return n, w_out, corners


def _stride_padding_axes(ax, n, p):
    for r in range(n):
        for c in range(n):
            is_pad = p and (r < p or c < p or r >= n - p or c >= n - p)
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1,
                                   facecolor=LIGHT_GREY if is_pad else "white",
                                   edgecolor=GREY, linewidth=0.7))
            if is_pad:
                ax.text(c + 0.5, n - 0.5 - r, "0", ha="center",
                        va="center", fontsize=9, color=GREY)


def stride_padding():
    """Effect of stride and zero padding on the output size — window positions computed."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    window_colours = [LIGHT_ORANGE, LIGHT_GREEN, LIGHT_BLUE]
    for ax, (w_in, f, s, p) in zip(axes, STRIDE_SETTINGS):
        n, w_out, _ = _stride_positions(w_in, f, s, p)
        ax.set(xlim=(-0.6, n + 0.6), ylim=(-1.6, n + 0.6))
        ax.set_aspect("equal")
        ax.axis("off")
        _stride_padding_axes(ax, n, p)
        for k in range(min(w_out, 3)):
            ax.add_patch(Rectangle((k * s, n - f - k * s), f, f,
                                   facecolor=window_colours[k], alpha=0.55,
                                   edgecolor=RED, linewidth=1.4,
                                   zorder=3))
        ax.set_title(f"$F={f}$, $S={s}$, $P={p}$", fontsize=11, pad=6)
        ax.text(n / 2, -1.0, f"output ${w_out}\\times{w_out}$", ha="center",
                va="center", fontsize=11)
    fig.tight_layout()
    save(fig, "stride-padding")
    stride_padding_animation()


def _stride_padding_frame(step):
    """All three settings at once: the window at its `step`-th position (a
    setting with fewer positions stays at its last one) and a dot in the centre
    of every position visited so far — the dots are the output grid."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, (w_in, f, s, p) in zip(axes, STRIDE_SETTINGS):
        n, w_out, corners = _stride_positions(w_in, f, s, p)
        ax.set(xlim=(-0.6, n + 0.6), ylim=(-1.6, n + 0.6))
        ax.set_aspect("equal")
        ax.axis("off")
        _stride_padding_axes(ax, n, p)
        k = min(step, len(corners) - 1)
        r, c = corners[k]
        ax.add_patch(Rectangle((c, n - f - r), f, f, facecolor=LIGHT_ORANGE,
                               alpha=0.55, edgecolor=RED, linewidth=1.6,
                               zorder=3))
        for rr, cc in corners[:k + 1]:
            ax.plot(cc + f / 2, n - rr - f / 2, "o", color=RED, ms=5,
                    zorder=4)
        ax.set_title(f"$F={f}$, $S={s}$, $P={p}$", fontsize=11, pad=6)
        ax.text(n / 2, -1.0,
                f"position {k + 1} of {len(corners)}:  "
                f"output ${w_out}\\times{w_out}$",
                ha="center", va="center", fontsize=10.5)
    # a fixed layout, so that the grids do not jitter between frames
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90,
                        wspace=0.12)
    return fig


def stride_padding_animation():
    """GIF: the three windows slide at the same time; S = 2 is done after four
    steps, P = 1 needs all 25 — the number of dots is the size of the output."""
    counts = [len(_stride_positions(*setting)[2]) for setting in STRIDE_SETTINGS]
    assert counts == [9, 4, 25], counts
    last = max(counts) - 1
    frames = render_frames([_stride_padding_frame(last)]
                           + [_stride_padding_frame(k) for k in range(last + 1)])
    save_gif(frames, "stride-padding", duration=450)


POOL_X = np.array([[1, 0, 2, 2],
                   [6, 4, 3, 4],
                   [1, 9, 4, 0],
                   [7, 8, 0, 1]], float)
POOL_QUADRANTS = np.array([[ORANGE, GREEN], [PDF_TURQUOISE, GOLD]])


def _pooling_frame(maximum, mean_rounded, step=None):
    """The pooling example; with `step` = 0 … 3 the window sits on that quadrant
    and only the outputs computed so far are filled in (None = all of them)."""
    x = POOL_X
    input_colours = [[POOL_QUADRANTS[r // 2, s // 2] for s in range(4)]
                    for r in range(4)]
    output_colours = [[POOL_QUADRANTS[r, s] for s in range(2)] for r in range(2)]
    if step is not None:
        done = np.arange(4).reshape(2, 2) <= step
        maximum = np.where(done, maximum, np.nan)
        mean_rounded = np.where(done, mean_rounded, np.nan)
        output_colours = [[output_colours[r][s] if done[r, s] else "white"
                           for s in range(2)] for r in range(2)]

    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.set(xlim=(0, 172), ylim=(0, 60))
    ax.set_aspect("equal")
    ax.axis("off")
    _number_grid(ax, 6, 52, x, 11, input_colours, fontsize=13)
    ax.text(28, 54, "input $4\\times4$", ha="center", va="bottom", fontsize=11)

    # The arrows have to be thick enough to hold a white label.
    block_arrow(ax, (58, 27), (108, 43))
    ax.text(83, 35, "max pool", ha="center", va="center", fontsize=10,
            rotation=np.degrees(np.arctan2(16, 50)), color="white",
            zorder=5)
    block_arrow(ax, (58, 19), (108, 7))
    ax.text(83, 13, "average pool", ha="center", va="center", fontsize=10,
            rotation=np.degrees(np.arctan2(-12, 50)), color="white",
            zorder=5)

    _number_grid(ax, 120, 56, maximum, 11, output_colours, fontsize=13)
    _number_grid(ax, 120, 26, mean_rounded, 11, output_colours,
                  fontsize=11, number_format="{:.1f}")
    ax.text(153, 45, "$2\\times2$", ha="left", va="center", fontsize=10)
    ax.text(153, 15, "$2\\times2$", ha="left", va="center", fontsize=10)
    if step is not None:
        r, s = divmod(step, 2)
        ax.add_patch(Rectangle((6 + 22 * s, 52 - 22 * (r + 1)), 22, 22,
                               facecolor="none", edgecolor="black", linewidth=3.2,
                               zorder=3))
        for y0 in (56, 26):
            ax.add_patch(Rectangle((120 + 11 * s, y0 - 11 * (r + 1)), 11, 11,
                                   facecolor="none", edgecolor="black",
                                   linewidth=3.2, zorder=3))
    return fig


def pooling_example():
    """Page 16: max and average pooling 2x2 with stride 2, the numbers computed."""
    x = POOL_X
    blocks = x.reshape(2, 2, 2, 2)
    maximum = blocks.max(axis=(1, 3))
    mean = blocks.mean(axis=(1, 3))
    assert np.array_equal(maximum, [[6, 4], [9, 4]])
    assert np.allclose(mean, [[2.75, 2.75], [6.25, 1.25]])
    # The original slide rounds halves up (2.75 -> 2.8; 6.25 -> 6.3).
    mean_rounded = np.floor(mean * 10 + 0.5) / 10
    assert np.array_equal(mean_rounded, [[2.8, 2.8], [6.3, 1.3]])
    # the top left quadrant, worked out in the notes: 1, 0, 6, 4
    assert sorted(x[:2, :2].ravel()) == [0, 1, 4, 6]

    save(_pooling_frame(maximum, mean_rounded), "pooling-priklad")

    # GIF: the 2x2 window jumps by S = 2 over the four quadrants; the first
    # and the last frame are the finished outputs.
    frames = render_frames(
        [_pooling_frame(maximum, mean_rounded)]
        + [_pooling_frame(maximum, mean_rounded, k) for k in range(4)]
        + [_pooling_frame(maximum, mean_rounded)])
    save_gif(frames, "pooling-priklad", duration=1400)


def pooling_windows():
    """Effect of pooling F and S on the output size — windows computed from the formula."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, (f, s) in zip(axes, [(2, 2), (3, 3)]):
        n = 6
        w_out = (n - f) // s + 1
        assert (n - f) % s == 0
        ax.set(xlim=(-0.5, n + 0.5), ylim=(-1.6, n + 0.5))
        ax.set_aspect("equal")
        ax.axis("off")
        for r in range(n):
            for c in range(n):
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor="white",
                                       edgecolor=GREY, linewidth=0.7))
        colours = [ORANGE, GREEN, PDF_BLUE, GOLD, RED, TURQUOISE,
                 PURPLE, GREY, BLUE]
        for i in range(w_out):
            for j in range(w_out):
                ax.add_patch(Rectangle((j * s, n - f - i * s), f, f,
                                       facecolor=colours[(i * w_out + j) % len(colours)],
                                       alpha=0.35, edgecolor="black",
                                       linewidth=1.2, zorder=3))
        ax.set_title(f"$F={f}$, $S={s}$", fontsize=11, pad=6)
        ax.text(n / 2, -1.0, f"$6\\times6 \\rightarrow {w_out}\\times{w_out}$",
                ha="center", va="center", fontsize=11)
    fig.tight_layout()
    save(fig, "pooling-okna")


def relu_softmax():
    """Plots of the two activation functions the slide names but does not draw."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    a = np.linspace(-3, 3, 400)
    ax1.plot(a, np.maximum(a, 0), color=BLUE, lw=2.0)
    ax1.axhline(0, color="black", lw=0.9)
    ax1.axvline(0, color="black", lw=0.9)
    # no titles: the formulas stand on the slide right above the plots
    ax1.set(xlim=(-3, 3), ylim=(-0.6, 3), xlabel=r"$z$",
            ylabel=r"$f(z)$", yticks=[0, 1, 2, 3])
    ax1.spines[["top", "right"]].set_visible(False)

    logits = np.array([2.0, 1.0, 0.1])
    p = np.exp(logits) / np.exp(logits).sum()
    assert np.allclose(p.sum(), 1.0)
    assert np.allclose(np.round(p, 2), [0.66, 0.24, 0.10])
    x = np.arange(3)
    ax2.bar(x - 0.19, logits, width=0.36, color=GREY, edgecolor="black",
            linewidth=0.6, label=r"logit $z_c$")
    ax2.bar(x + 0.19, p, width=0.36, color=ORANGE, edgecolor="black",
            linewidth=0.6, label=r"softmax $f(z_c)$")
    for i in range(3):
        ax2.text(x[i] + 0.19, p[i] + 0.05, f"{p[i]:.2f}", ha="center",
                 fontsize=10)
        ax2.text(x[i] - 0.19, logits[i] + 0.05, f"{logits[i]:.1f}",
                 ha="center", fontsize=10)
    ax2.set(xticks=x, xticklabels=["$c=1$", "$c=2$", "$c=3$"], ylim=(0, 2.5))
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "relu-softmax")


# =========================================================================
# Page 22 of the PDF — CNN architecture
# =========================================================================

def cnn_pipeline():
    """Diagram of a typical architecture; the box sizes come from a real computation."""
    rows = layer_sizes.walk((32, 32, 3))
    # (label, spatial size, depth, colour of the front face), built from the
    # computed rows up to the first FC layer, so that the boxes follow the
    # table of the next slide
    front = iter([PDF_TURQUOISE, PDF_GREEN, PDF_RED, PDF_BLUE])
    blocks, colour = [], PDF_ORANGE
    for name, (w_, h_, d_), _ in rows:
        if name.startswith("FC"):
            break
        if name == "relu":
            blocks.append(("relu", None, None, None))
            continue
        if name.startswith("conv"):
            colour = next(front)
        label = ("Input" if name == "input" else
                 "Conv" if name.startswith("conv") else "Pool")
        blocks.append((label, w_, d_, PDF_ORANGE if label == "Input" else colour))
    assert [(b[0], b[1], b[2]) for b in blocks if b[0] != "relu"] == [
        ("Input", 32, 3), ("Conv", 32, 16), ("Conv", 32, 16), ("Pool", 16, 16),
        ("Conv", 16, 32), ("Conv", 16, 32), ("Pool", 8, 32)]
    assert sum(b[0] == "relu" for b in blocks) == 4

    fig, ax = plt.subplots(figsize=(11.6, 2.5))
    ax.set(xlim=(0, 372), ylim=(5, 85))
    ax.set_aspect("equal")
    ax.axis("off")

    centre_y = 42
    x = 8
    for label, size, depth, colour in blocks:
        if label == "relu":
            ax.add_patch(Circle((x + 6, centre_y), 5.4, facecolor=PDF_ORANGE,
                                edgecolor="black", lw=0.8, zorder=6))
            ax.text(x + 6, centre_y - 6.6, "ReLU", ha="center", va="top",
                    fontsize=10)
            x += 26
            continue
        h = size * 1.35                       # front face proportional to W x H
        d = depth * 0.95                        # depth proportional to the number of maps
        w = 5.0
        y = centre_y - h / 2
        n_slabs = 1 if depth <= 3 else 7
        colours = ([MAP_COLOURS[i % len(MAP_COLOURS)] for i in range(n_slabs - 1)]
                 + [colour])
        stack(ax, x, y, w, h, d / n_slabs, colours[::-1], lw=0.7)
        ox = abs(d * ISO[0])
        ax.text(x + w / 2 + ox / 2, y + h + d * ISO[1] + 2, label,
                ha="center", va="bottom", fontsize=10)
        if label in ("Input", "Pool"):     # otherwise the labels would run together
            ax.text(x + w / 2 + ox / 2, y - 2.5,
                    f"{size}$\\times${size}$\\times${depth}",
                    ha="center", va="top", fontsize=9, color=GREY)
        x += w + ox + 18

    # The fully connected part
    class_y = [centre_y + 9, centre_y, centre_y - 9]
    for y in class_y:
        ax.add_patch(Circle((x + 22, y), 3.4, facecolor=PDF_ORANGE,
                            edgecolor="black", lw=0.8, zorder=6))
        arrow(ax, (x + 25.8, y), (x + 34, y), colour=PDF_ORANGE, lw=1.0,
              size=9)
        for source in np.linspace(centre_y - 12, centre_y + 12, 7):
            line(ax, (x - 14, source), (x + 18.6, y), colour=PDF_ORANGE,
                 lw=0.5, alpha=0.8, zorder=4)
    ax.text(x + 22, centre_y + 14, "FC layers\n+ softmax", ha="center",
            va="bottom", fontsize=10)
    save(fig, "cnn-pipeline")


def layer_parameters():
    """Where in the architecture the parameters sit — computed from `layer_sizes`."""
    rows = [r for r in layer_sizes.walk((32, 32, 3)) if r[2] > 0]
    total = sum(r[2] for r in rows)
    assert total == 148_442
    names = ["conv 1\n16", "conv 2\n16", "conv 3\n32", "conv 4\n32",
             "FC\n64", "FC\n10"]
    values = [r[2] for r in rows]
    colours = [PDF_BLUE] * 4 + [ORANGE, ORANGE]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.bar(names, values, color=colours, edgecolor="black",
                   linewidth=0.6, width=0.62)
    for bar, h in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3000, thousands(h),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("parameters")
    ax.set_ylim(0, 155_000)
    # the same thin-space grouping as the bar labels and the table
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: thousands(int(v))))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=9)
    # no title: the sentence under the chart on the slide says it
    assert round(100 * max(values) / total) == 88
    fig.tight_layout()
    save(fig, "parametry-vrstev")


# =========================================================================
# Pages 23 and 24 of the PDF — feature hierarchy, learned filters and maps
# =========================================================================

def _receptive_field(layers):
    """Size of the receptive field given the layers [(F, S), ...]."""
    rf, jump = 1, 1
    sizes = []
    for f, s in layers:
        rf += (f - 1) * jump
        jump *= s
        sizes.append(rf)
    return sizes


def feature_hierarchy():
    """Page 23: why deeper features are more abstract — the receptive field grows."""
    layers = [(5, 1), (2, 2), (3, 1), (2, 2), (3, 1)]
    rf = _receptive_field(layers)
    assert [rf[0], rf[2], rf[4]] == [5, 10, 20]
    levels = [("low-level feature", rf[0]), ("mid-level feature", rf[2]),
              ("high-level feature", rf[4])]

    image = load_sample_image("china.jpg")[70:390, 130:450] / 255.0
    image = image.reshape(32, 10, 32, 10, 3).mean(axis=(1, 3))

    fig = plt.figure(figsize=(8.7, 3.63))
    top_ax = fig.add_axes((0.0, 0.60, 1.0, 0.40))
    top_ax.set(xlim=(0, 100), ylim=(0, 20))
    top_ax.axis("off")
    names = ["low-level\nfeature", "mid-level\nfeature", "high-level\nfeature",
             "trainable\nclassifier"]
    for i, name in enumerate(names):
        x = 24 + 19 * i
        top_ax.add_patch(Rectangle((x, 4), 14, 11, facecolor="white",
                                   edgecolor=RED, linewidth=1.6))
        top_ax.text(x + 7, 9.5, name, ha="center", va="center", fontsize=10.5)
        arrow(top_ax, (x - 5, 9.5), (x - 0.6, 9.5), colour=PDF_BLUE, lw=1.4,
              size=11)
    arrow(top_ax, (x + 14.6, 9.5), (x + 20, 9.5), colour=PDF_BLUE, lw=1.4,
          size=11)
    top_ax.text(11, 9.5, "input\nimage", ha="center", va="center", fontsize=10.5)
    for i in range(3):
        top_ax.plot([31 + 19 * i, 12.5 + 32 * i], [3.6, -2.2], color=GREEN,
                    lw=1.2, zorder=2, clip_on=False)

    for i, (name, size) in enumerate(levels):
        left = 0.06 + 0.32 * i
        ax = fig.add_axes((left, 0.10, 0.13, 0.42))
        ax.imshow(image, interpolation="nearest")
        r0 = (32 - size) // 2
        ax.add_patch(Rectangle((r0 - 0.5, r0 - 0.5), size, size,
                               facecolor="none", edgecolor=RED,
                               linewidth=1.6))
        ax.set_axis_off()
        ax2 = fig.add_axes((left + 0.155, 0.10, 0.13, 0.42))
        ax2.imshow(image[r0:r0 + size, r0:r0 + size],
                   interpolation="nearest")
        ax2.set_axis_off()
        for spine in ("top", "bottom", "left", "right"):
            ax2.spines[spine].set_visible(False)
        ax2.set_title(f"{size}$\\times${size} px", fontsize=10, pad=3)
        ax.set_title(f"receptive field\nof a {name.split()[0]} unit",
                     fontsize=10, pad=3)
    save(fig, "hierarchie-priznaku")


def learned_filters():
    """Page 24: really learned 5x5 filters and their activation maps."""
    weight_frames = []
    model, (x_test, _), history = cnn_filters.train(frames=weight_frames)
    order = cnn_filters.filter_order(model, x_test[:200])
    filters = cnn_filters.filters(model)[order]
    input_image = x_test[3]
    maps = cnn_filters.forward(model, input_image[None])["a"][0][..., order]
    print(f"  trained test accuracy: {history[-1]:.3f}")
    assert history[-1] > 0.85, history[-1]
    assert len(weight_frames) == len(history) + 1
    learned_filters_training(weight_frames, history, order)

    k = cnn_filters.K
    ratio = 8.9 / 4.0
    fig = plt.figure(figsize=(8.9, 4.0))

    ax = fig.add_axes((0.025, 0.30, 0.15, 0.15 * ratio))
    ax.imshow(input_image, interpolation="nearest")
    ax.set(xticks=[], yticks=[])
    ax.set_title("input image $32\\times32\\times3$", fontsize=10, pad=4)

    highlights = {0: RED, 5: PDF_BLUE}
    width, spacing = 0.052, 0.060
    filter_centre = {}
    for i in range(k):
        left = 0.235 + i * spacing
        af = fig.add_axes((left, 0.74, width, width * ratio))
        af.imshow(filters[i], interpolation="nearest")
        af.set(xticks=[], yticks=[])
        filter_centre[i] = left + width / 2
        for spine in af.spines.values():
            spine.set_color(highlights.get(i, "black"))
            spine.set_linewidth(2.4 if i in highlights else 0.5)
    fig.text(0.235 + k * spacing / 2, 0.955,
             f"$K={k}$ learned filters $5\\times5\\times3$", ha="center",
             fontsize=11)

    n_cols, map_width, map_spacing = 6, 0.098, 0.112
    map_centre = {}
    for i in range(k):
        r, col = divmod(i, n_cols)
        left = 0.29 + col * map_spacing
        bottom = 0.36 - r * 0.30
        am = fig.add_axes((left, bottom, map_width, map_width * ratio))
        am.imshow(maps[..., i], cmap="gray", interpolation="nearest")
        am.set(xticks=[], yticks=[])
        map_centre[i] = (left + map_width / 2, bottom + map_width * ratio)
        for spine in am.spines.values():
            spine.set_color(highlights.get(i, "black"))
            spine.set_linewidth(2.4 if i in highlights else 0.5)

    for i, colour in highlights.items():
        fig.add_artist(FancyArrowPatch(
            (filter_centre[i], 0.735), map_centre[i], transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=13, color=colour, lw=1.3,
            shrinkA=2, shrinkB=2))
    fig.text(0.10, 0.20, "activation maps\n$28\\times28$ each", ha="center",
             va="top", fontsize=11)
    fig.text(0.10, 0.10, "one filter\n$\\Rightarrow$ one map", ha="center",
             va="top", fontsize=11, color=RED)
    save(fig, "naucene-filtry")


def _filters_training_frame(weights, history, order, epoch):
    """One frame of the animation: the filters after `epoch` epochs and the accuracy curve."""
    k = cnn_filters.K
    images = cnn_filters.filters_from_weights(weights[epoch])[order]
    fig = plt.figure(figsize=(8.0, 2.9))
    width, spacing = 0.066, 0.076
    for i in range(k):
        af = fig.add_axes((0.045 + i * spacing, 0.60, width, width * 8.0 / 2.9))
        af.imshow(images[i], interpolation="nearest")
        af.set(xticks=[], yticks=[])
        for spine in af.spines.values():
            spine.set_linewidth(0.5)
    if epoch == 0:
        status = "epoch 0 — random initialization"
    else:
        status = (f"epoch {epoch} of {len(history)} — "
                f"test accuracy {history[epoch - 1]:.2f}")
    fig.text(0.5, 0.95, status, ha="center", va="top", fontsize=11)

    ax = fig.add_axes((0.10, 0.17, 0.85, 0.34))
    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, history, color=GREY, lw=1.0, alpha=0.5)
    if epoch > 0:
        ax.plot(epochs[:epoch], history[:epoch], color=BLUE, lw=1.8)
        ax.plot(epoch, history[epoch - 1], "o", color=RED, ms=5)
    ax.axhline(1 / cnn_filters.N_CLASSES, color=GREY, lw=0.8, ls=":")
    ax.text(len(history), 1 / cnn_filters.N_CLASSES + 0.03, "chance",
            ha="right", va="bottom", fontsize=8, color=GREY)
    ax.set(xlim=(0, len(history)), ylim=(0, 1.0), yticks=[0, 0.5, 1],
           xlabel="epoch", ylabel="test accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    return fig


def learned_filters_training(weights, history, order):
    """GIF: how oriented colour edges emerge out of random weights."""
    last = len(history)
    epochs = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60,
              80, 100, last]
    frames = [render_frame(_filters_training_frame(weights, history, order, last))]
    for e in epochs:
        frames.append(render_frame(_filters_training_frame(weights, history, order, e)))
    save_gif(frames, "naucene-filtry-trenink", duration=700)
    # Static state for PDF/print: the same first frame as an SVG.
    save(_filters_training_frame(weights, history, order, last),
         "naucene-filtry-trenink")


# =========================================================================
# Page 5 of the PDF — vanishing signal in a deep network (computed, not drawn)
# =========================================================================

def _deep_network(rng, n_layers, width, x, y, activation, weight_std):
    """A random `n_layers` × `width` network; returns the std of the output of
    every layer in the forward pass and the gradient norm of the weights of
    every layer in the backward pass."""
    weights = [rng.normal(0, weight_std, (width, width)) for _ in range(n_layers)]
    w_out = rng.normal(0, np.sqrt(1 / width), (width, 1))
    outputs, pre, h = [x], [], x
    for w in weights:
        z = h @ w
        h = 1 / (1 + np.exp(-z)) if activation == "sigmoid" else np.maximum(z, 0)
        pre.append(z)
        outputs.append(h)
    dh = ((h @ w_out) - y) / len(y) @ w_out.T        # MSE, back through the output
    gradients = []
    for l in reversed(range(n_layers)):
        if activation == "sigmoid":
            s = outputs[l + 1]
            dz = dh * s * (1 - s)                     # sigma' = s (1 - s) <= 1/4
        else:
            dz = dh * (pre[l] > 0)
        gradients.append(np.linalg.norm(outputs[l].T @ dz))
        dh = dz @ weights[l].T
    return [v.std() for v in outputs[1:]], gradients[::-1]


# weight std for d = 64 inputs per neuron; "std", not sigma_w, because the
# same slide uses sigma for the sigmoid
FORWARD_CURVES = [(1.0, r"std $= 1/\sqrt{d}$", RED, "--"),
                  (np.sqrt(2), r"std $= \sqrt{2/d}$  (He)", BLUE, "-"),
                  (2.0, r"std $= 2/\sqrt{d}$", ORANGE, "-.")]


def _vanishing_figure(forward_std, g_sig, g_relu, n_layers, shown=None,
                      ylims=None):
    """The forward pass (left) and the backward pass (right).

    `shown` = number of layers the signal has passed so far: the forward pass
    is drawn for layers 1 … shown, the backward pass for layers
    L − shown + 1 … L (it starts at the output); None = everything."""
    layers = np.arange(1, n_layers + 1)
    k = n_layers if shown is None else shown
    fwd = slice(0, k)
    bwd = slice(n_layers - k, n_layers)
    # full slide width since the October 2026 review; taller and rendered at
    # dpi 125 since the visual pass (≈ 1170 × 390 px, 9 pt ≈ 15.6 px)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.1))
    for c, label, colour, style in FORWARD_CURVES:
        ax1.plot(layers[fwd], forward_std[c][fwd], color=colour, lw=1.8,
                 ls=style, marker="o", ms=3, label=label)
    ax1.set_yscale("log")
    ax1.set(xlabel="layer $l$", ylabel="std of the layer output",
            xticks=[1, 4, 8, 12], ylim=(2e-3, 80))
    ax1.set_title("forward pass — ReLU, random weights", fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax2.plot(layers[bwd], (g_sig / g_sig[-1])[bwd], color=RED, lw=1.8,
             marker="o", ms=3, label="sigmoid, Glorot init")
    ax2.plot(layers[bwd], (g_relu / g_relu[-1])[bwd], color=BLUE, lw=1.8,
             marker="o", ms=3, label="ReLU, He init")
    ax2.plot(layers, 4.0 ** (layers - n_layers), color=GREY, lw=1.2, ls=":",
             label=r"$(1/4)^{L-l}$ ($\sigma'$ factors only)")
    ax2.set_yscale("log")
    ax2.set(xlabel="layer $l$", ylabel="gradient norm / last layer",
            xticks=[1, 4, 8, 12])
    ax2.set_title("backward pass — 12 layers, 64 neurons", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    if shown is not None:
        # the layer the signal has just reached, in both directions
        ax1.axvline(k, color=GREY, lw=0.8, ls="--", zorder=0)
        ax2.axvline(n_layers - k + 1, color=GREY, lw=0.8, ls="--", zorder=0)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.45, n_layers + 0.55)
    if ylims is not None:
        ax2.set_ylim(ylims)
    fig.tight_layout()
    return fig, ax2


def vanishing_gradient():
    """Forward and backward pass through a random network of 12 layers of 64 neurons."""
    rng = np.random.default_rng(0)
    n_layers, width = 12, 64
    x = rng.normal(size=(256, width))
    y = rng.normal(size=(256, 1))

    # Forward pass with ReLU: the variance of the signal depends only on the
    # variance of the weights.
    forward_std = {}
    for c, *_ in FORWARD_CURVES:
        std, _ = _deep_network(np.random.default_rng(1), n_layers, width, x, y,
                              "relu", c / np.sqrt(width))
        forward_std[c] = np.array(std)
    # The figure has to show what the slide claims: the signal dies out,
    # holds, or explodes.
    assert forward_std[1.0][-1] / forward_std[1.0][0] < 0.05
    assert forward_std[2.0][-1] / forward_std[2.0][0] > 20
    assert 0.5 < forward_std[np.sqrt(2)][-1] / forward_std[np.sqrt(2)][0] < 2
    # every point lies inside the fixed y range of the left plot
    assert all(2e-3 < v.min() and v.max() < 80 for v in forward_std.values())

    # Backward pass: sigmoid versus ReLU.
    _, g_sig = _deep_network(np.random.default_rng(2), n_layers, width, x, y,
                            "sigmoid", np.sqrt(1 / width))
    _, g_relu = _deep_network(np.random.default_rng(2), n_layers, width, x, y,
                             "relu", np.sqrt(2 / width))
    g_sig, g_relu = np.array(g_sig), np.array(g_relu)
    assert g_sig[0] / g_sig[-1] < 1e-5, g_sig[0] / g_sig[-1]
    assert g_relu[0] / g_relu[-1] > 0.1, g_relu[0] / g_relu[-1]
    # "seven orders of magnitude" in the notes
    assert 1e-8 < g_sig[0] / g_sig[-1] < 1e-7, g_sig[0] / g_sig[-1]

    fig, ax2 = _vanishing_figure(forward_std, g_sig, g_relu, n_layers)
    ylims = ax2.get_ylim()
    save(fig, "mizeni-gradientu")
    print(f"  vanishing gradient: sigmoid {g_sig[0] / g_sig[-1]:.1e}, "
          f"ReLU {g_relu[0] / g_relu[-1]:.2f}")

    # GIF: the signal goes forward from the first layer, the gradient goes
    # backward from the last one — one layer per frame. The y range of the
    # right plot is fixed to that of the full figure.
    figures = [_vanishing_figure(forward_std, g_sig, g_relu, n_layers,
                                 ylims=ylims)[0]]
    for k in range(1, n_layers + 1):
        figures.append(_vanishing_figure(forward_std, g_sig, g_relu, n_layers,
                                         shown=k, ylims=ylims)[0])
    figures.append(_vanishing_figure(forward_std, g_sig, g_relu, n_layers,
                                     ylims=ylims)[0])
    save_gif(render_frames(figures, dpi=125), "mizeni-gradientu", duration=650)


# =========================================================================
# Pages 19 and 20 of the PDF — dropout and batch normalization on a map, by hand
# =========================================================================

def _dropout_frame(x, mask, p, mask_title, footer):
    y = x * mask / p                                    # inverted dropout
    # three panels 30 units apart, so that their titles do not run together
    fig, ax = plt.subplots(figsize=(9.8, 3.3))
    ax.set(xlim=(0, 196), ylim=(-5, 61))
    ax.set_aspect("equal")
    ax.axis("off")
    step = 7
    grey_cells = [[LIGHT_GREY if mask[r, s] == 0 else "white" for s in range(6)]
            for r in range(6)]
    _number_grid(ax, 4, 48, x, step, fontsize=10)
    ax.text(25, 50, "input $6\\times6$\n(one channel)", ha="center",
            va="bottom", fontsize=10.5)
    _number_grid(ax, 76, 48, mask, step, grey_cells, fontsize=10)
    ax.text(97, 50, mask_title, ha="center", va="bottom", fontsize=10.5)
    _number_grid(ax, 148, 48, y, step, grey_cells, fontsize=10)
    ax.text(169, 50, "output\n= input $\\cdot$ mask $/\\,p$", ha="center",
            va="bottom", fontsize=10.5)
    for x0 in (51, 123):
        arrow(ax, (x0, 27), (x0 + 20, 27), colour=PDF_ORANGE, lw=5,
              size=18)
    ax.text(98, 0.5, footer, ha="center", va="top", fontsize=10)
    return fig


DROPOUT_BATCHES = 16


def dropout_map():
    """Dropout on one 6x6 map with p = 0.5: the mask, the number kept, the 1/p scaling."""
    rng = np.random.default_rng(10)
    x = rng.integers(1, 10, (6, 6)).astype(float)
    p = 0.5
    mask = (rng.uniform(size=x.shape) < p).astype(float)
    y = x * mask / p                                    # inverted dropout
    kept = int(mask.sum())
    assert kept == 20, kept                    # the number is on the slide
    assert np.allclose(y[mask == 1], 2 * x[mask == 1])
    # On this one mask the mean of the map is NOT preserved (the notes quote
    # 5.6 -> 6.5); it is preserved only in expectation over the masks, which
    # the GIF assert below checks.
    assert (round(x.mean(), 2), round(y.mean(), 2)) == (5.61, 6.5), (
        x.mean(), y.mean())
    # the two cells the notes let the students check: 7 -> 14, a disabled 9 -> 0
    assert any(x[m] == 7 and y[m] == 14 for m in np.ndindex(6, 6))
    assert any(x[m] == 9 and y[m] == 0 for m in np.ndindex(6, 6))

    save(_dropout_frame(
        x, mask, p, f"mask, $p={p}$\nkept {kept} of 36",
        f"training: a new mask for every sample and mini-batch, expected "
        f"{36 * p:.0f} kept;   prediction: mask $= 1$ everywhere, no scaling"),
        "dropout-mapa")

    # GIF: one new mask per mini-batch. The last mini-batch is the mask of the
    # static figure, so that the first frame (= the final state) is the same
    # picture the notes talk about.
    other = np.random.default_rng(13)
    masks = [(other.uniform(size=x.shape) < p).astype(float)
             for _ in range(DROPOUT_BATCHES - 1)] + [mask]
    kept_counts = np.array([int(m.sum()) for m in masks])
    running = np.cumsum(kept_counts) / np.arange(1, DROPOUT_BATCHES + 1)
    # the output mean matches the input mean only on average over the masks
    output_means = np.array([(x * m / p).mean() for m in masks])
    assert abs(output_means.mean() - x.mean()) < 0.6, (output_means.mean(),
                                                      x.mean())
    assert abs(running[-1] - 36 * p) < 1.5, running[-1]
    # the numbers in the notes and in the fig-alt of the slide
    assert (kept_counts.min(), kept_counts.max()) == (14, 23)
    assert np.isclose(running[-1], 18.5)
    print(f"  dropout: kept {kept_counts.tolist()}, mean {running[-1]:.2f}; "
          f"input mean {x.mean():.2f}, output mean over the masks "
          f"{output_means.mean():.2f}")

    def frame(t):
        return _dropout_frame(
            x, masks[t], p,
            f"mini-batch {t + 1}, $p={p}$\nkept {kept_counts[t]} of 36",
            f"the map of one sample, a new mask in every mini-batch — kept on "
            f"average over {t + 1} mini-batch{'es' if t else ''}: "
            f"{running[t]:.1f}  (expected {36 * p:.0f})")

    frames = render_frames([frame(DROPOUT_BATCHES - 1)]
                           + [frame(t) for t in range(DROPOUT_BATCHES)])
    save_gif(frames, "dropout-mapa", duration=900)


def batchnorm_channel():
    """Batch normalization of one channel j over a batch of b = 2 images 2x2.

    The notation is that of the previous lecture (channel j = its feature j,
    batch size b, mu_{B_j}, sigma^2_{B_j}, gamma_j, beta_j); the position in
    the map is deliberately not indexed.
    """
    channel = np.array([[[2, 4], [6, 8]],
                      [[2, 6], [6, 6]]], float)          # b x H x W
    mu = channel.mean()
    var = channel.var()
    assert (mu, var) == (5.0, 4.0)
    norm = (channel - mu) / np.sqrt(var)                   # epsilon omitted
    gamma, beta = 2.0, 1.0
    output = gamma * norm + beta
    assert np.array_equal(norm[0], [[-1.5, -0.5], [0.5, 1.5]])
    assert np.array_equal(output[1], [[-2, 2], [2, 2]])
    assert (norm.min(), norm.max(), output.min(), output.max()) == (-1.5, 1.5,
                                                                    -2, 4)
    assert output[0, 1, 1] == 4                 # the 8 of the notes: 1.5 -> 4

    fig, ax = plt.subplots(figsize=(9.8, 3.2))
    ax.set(xlim=(0, 196), ylim=(0, 64))
    ax.set_aspect("equal")
    ax.axis("off")
    step = 10                                   # wide enough for "−1.5"

    def channel_stack(x0, y0, values, highlight, fmt="{:.0f}", fontsize=10.5):
        # three channels as offset slabs, the highlighted one in front with numbers
        for k, shift in enumerate((2, 1)):
            ax.add_patch(Rectangle((x0 + 3 * shift, y0 - 2 * step + 3 * shift),
                                   2 * step, 2 * step, facecolor=LIGHT_GREY,
                                   edgecolor=GREY, lw=0.7, zorder=1 - 0.1 * k))
        colour = [[LIGHT_GREEN if highlight else "white"] * 2 for _ in range(2)]
        _number_grid(ax, x0, y0, values, step, colour, fontsize=fontsize,
                      number_format=fmt)

    for i in range(2):
        y0 = 50 - 28 * i
        channel_stack(6, y0, channel[i], True)
        ax.text(2, y0 - step, f"image {i + 1}", ha="right", va="center",
                fontsize=10, rotation=90)
        # 9.5 pt: at 10.5 pt "−1.5" and "−0.5" filled their cells edge to edge
        channel_stack(98, y0, norm[i], False, "{:.1f}", fontsize=9.5)
        channel_stack(156, y0, output[i], False)
    ax.text(19, 58, "channel $j$ of a batch, $b=2$", ha="center", va="bottom",
            fontsize=10.5)
    ax.text(111, 58, r"$\tilde x = (x-\mu_{B_j})/\sigma_{B_j}$", ha="center",
            va="bottom", fontsize=10.5)
    ax.text(169, 58, r"$\tilde h = \gamma_j \tilde x + \beta_j$", ha="center",
            va="bottom", fontsize=10.5)

    arrow(ax, (36, 29), (92, 29), colour=PDF_ORANGE, lw=5, size=18)
    ax.text(64, 33, f"8 values:  $\\mu_{{B_j}} = {mu:.0f}$,  "
            f"$\\sigma^2_{{B_j}} = {var:.0f}$", ha="center", va="bottom",
            fontsize=10.5)
    ax.text(64, 25, "over the batch and\nall positions", ha="center",
            va="top", fontsize=9.5, color=GREY)
    arrow(ax, (127, 29), (151, 29), colour=PDF_ORANGE, lw=5, size=18)
    ax.text(139, 33, f"$\\gamma_j = {gamma:.0f}$\n$\\beta_j = {beta:.0f}$",
            ha="center", va="bottom", fontsize=10.5)
    ax.text(189, 29, "learned", ha="center", va="center", fontsize=9.5,
            color=GREY, rotation=90)
    save(fig, "batchnorm-kanal")


# =========================================================================

# =========================================================================
# Interactive playgrounds in index.qmd — the same numbers computed in Python
# =========================================================================
#
# The three playgrounds are vanilla JavaScript inside index.qmd. Their data
# (the input matrix, the filter presets, the architecture) is copied here and
# every number the speaker notes quote is recomputed and asserted, so that the
# notes, the static slides and the JavaScript cannot drift apart. The
# rendered playgrounds were additionally driven with Playwright and their
# displayed numbers compared with these functions (see sources.md).

# "Playground: slide the filter yourself" — must be the input of the worked example.
CV_INPUT = [[3, 0, 1, 2, 7], [1, 5, 8, 9, 3], [2, 7, 2, 5, 1],
            [0, 1, 3, 1, 7], [4, 2, 1, 6, 2]]
CV_PRESETS = {
    "vertical edge": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    "horizontal edge": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    "sum": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
}


def playground_conv(x, w, s, p):
    """Cross-correlation with stride and zero padding, as the playground computes it.

    Returns (output, valid): when (W - F + 2P) / S is not an integer the layer
    is invalid and, like PyTorch, the last positions that do not fit are dropped.
    """
    x = np.pad(np.asarray(x, float), p)
    w = np.asarray(w, float)
    f = w.shape[0]
    valid = (x.shape[0] - f) % s == 0
    n = (x.shape[0] - f) // s + 1
    out = np.array([[np.sum(x[i * s:i * s + f, j * s:j * s + f] * w)
                     for j in range(n)] for i in range(n)])
    return out, valid


# "Playground: the receptive field" — the layers are (F, S); a pool is (2, 2).
RF_PRESETS = {
    "the stack from the slide": [(5, 1), (2, 2), (3, 1), (2, 2), (3, 1)],
    "the network from the table": [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1),
                                   (2, 2)],
    "two 3x3": [(3, 1), (3, 1)],
    "one 5x5": [(5, 1)],
}


# "Playground: size a network yourself" — the architecture of the table.
def playground_network(w=32, k1=16, k2=32, hidden=64, pool1=True, pool2=True,
                       gap=False):
    """Rows (name, shape, parameters) of the playground network; None if invalid."""
    conv = lambda k: ("conv", {"F": 3, "S": 1, "P": 1, "K": k})   # noqa: E731
    pool = ("pool", {"F": 2, "S": 2})
    layers = [conv(k1), conv(k1)] + [pool] * pool1 + [conv(k2), conv(k2)] \
        + [pool] * pool2
    try:
        rows = layer_sizes.walk((w, w, 3), layers)
    except ValueError:
        return None
    d = rows[-1][1][2]
    if gap:                                  # global average pooling: W x H x D -> D
        rows.append(("global average pool", (1, 1, d), 0))
    rows += layer_sizes.walk(rows[-1][1], [("fc", {"K": hidden}),
                                           ("fc", {"K": 10})])[1:]
    return rows


def check_playgrounds():
    """Every number quoted in the notes of the three playgrounds."""
    # --- convolution: the default view is the worked example -----------------
    assert np.array_equal(np.array(CV_INPUT, float), CONV_X)
    assert np.array_equal(np.array(CV_PRESETS["vertical edge"], float), CONV_W)
    out, ok = playground_conv(CV_INPUT, CV_PRESETS["vertical edge"], 1, 0)
    assert ok and out.shape == (3, 3) and out[0, 0] == -5 and out[2, 2] == -4
    # output side (5 - 3 + 2P) / S + 1 for every setting of the sliders
    sides = {(s, p): playground_conv(CV_INPUT, CONV_W, s, p)
             for s in (1, 2, 3) for p in (0, 1, 2)}
    shown = {k: (v[0].shape[0], v[1]) for k, v in sides.items()}
    assert shown == {(1, 0): (3, True), (2, 0): (2, True), (3, 0): (1, False),
                     (1, 1): (5, True), (2, 1): (3, True), (3, 1): (2, False),
                     (1, 2): (7, True), (2, 2): (4, True), (3, 2): (3, True)}
    # P = 1, S = 1 ("same"): the corner sees only zeros on its left
    same = sides[(1, 1)][0]
    assert same[0, 0] == -(0 + 5) and same[0, 4] == 2 + 9
    assert same[:, 4].tolist() == [11, 16, 15, 12, 7]       # border column, notes
    # P = 1, S = 2: 3 x 3 again, but from every other position of the 5 x 5
    assert np.array_equal(sides[(2, 1)][0], same[::2, ::2])
    hor, _ = playground_conv(CV_INPUT, CV_PRESETS["horizontal edge"], 1, 0)
    assert hor[0, 0] == (3 + 0 + 1) - (2 + 7 + 2)                  # -7
    box_, _ = playground_conv(CV_INPUT, CV_PRESETS["sum"], 1, 0)
    assert box_[0, 0] == 29 and box_.max() == 41
    print(f"  conv playground: same padding {same[0, 0]:.0f} … "
          f"{same[0, 4]:.0f}; horizontal {hor[0, 0]:.0f}; sum max {box_.max():.0f}")

    # --- receptive field ----------------------------------------------------
    stack = RF_PRESETS["the stack from the slide"]
    assert _receptive_field(stack) == [5, 6, 10, 12, 20]
    assert _receptive_field(RF_PRESETS["two 3x3"])[-1] == 5
    assert _receptive_field(RF_PRESETS["one 5x5"])[-1] == 5
    assert _receptive_field(stack[:3])[-1] == 10                   # quiz III
    # the network of the table (padding does not change the field): the last
    # convolution sees 14 x 14 pixels, after the last pooling 16 x 16
    table = RF_PRESETS["the network from the table"]
    assert _receptive_field(table) == [3, 5, 6, 10, 14, 16]
    assert [(p["F"], p["S"]) for kind, p in layer_sizes.ARCHITECTURE
            if kind in ("conv", "pool")] == table
    # weights per (input channel, output channel) pair: sum of F^2 over convs
    assert sum(f * f for f, s in RF_PRESETS["two 3x3"]) == 18
    assert sum(f * f for f, s in RF_PRESETS["one 5x5"]) == 25

    # --- network sizes ------------------------------------------------------
    def total(**kw):
        rows = playground_network(**kw)
        return None if rows is None else sum(r[2] for r in rows)

    rows = playground_network()
    assert [r[2] for r in rows] == [0, 448, 2320, 0, 4640, 9248, 0,
                                    131_136, 650]
    assert total() == 148_442
    assert playground_network(pool2=False)[-2][2] == 524_352     # FC slide, row 3
    assert total(pool2=False) == 541_658
    assert total(w=64) == 148_442 - 131_136 + 524_352 == 541_658
    assert total(gap=True) == 16_656 + 32 * 64 + 64 + 650 == 19_418
    assert total(w=34) is None and total(w=30) is None           # 17, 15: not even
    assert total(w=34, pool2=False) == 16_656 + 17 * 17 * 32 * 64 + 64 + 650
    print(f"  network playground: default {total():,}, without 2nd pool "
          f"{total(pool2=False):,}, 64 px {total(w=64):,}, GAP {total(gap=True):,}")


# =========================================================================
# Added in the review of October 2026 (ideas chosen by the author)
# =========================================================================

EQUI_SHIFT = (2, 3)                  # rows down, columns right


def translation_equivariance():
    """Shifted input -> shifted activation map, computed and checked exactly.

    The filter is the vertical-edge filter of the worked example."""
    x = np.zeros((12, 12))
    x[2:7, 3] = 9                    # an "L": a vertical stroke ...
    x[6, 3:7] = 9                    # ... and a horizontal one
    dr, dc = EQUI_SHIFT
    shifted = np.roll(x, (dr, dc), axis=(0, 1))
    # the shape stays inside the image, so the roll really is a shift
    assert shifted.sum() == x.sum()
    assert not x[-dr:, :].any() and not x[:, -dc:].any()
    z = playground_conv(x, CONV_W, 1, 0)[0]
    z_shifted = playground_conv(shifted, CONV_W, 1, 0)[0]
    assert z.shape == (10, 10) and np.abs(z).sum() > 0
    # equivariance: the map of the shifted input is the shifted map, exactly
    assert not z[-dr:, :].any() and not z[:, -dc:].any()
    assert np.array_equal(z_shifted, np.roll(z, (dr, dc), axis=(0, 1)))
    peak = np.unravel_index(np.argmax(z), z.shape)
    peak_s = np.unravel_index(np.argmax(z_shifted), z.shape)
    assert (peak_s[0] - peak[0], peak_s[1] - peak[1]) == (dr, dc)
    assert z.max() == 27 and z.min() == -27   # 3 * 9 on the two sides of the stroke

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set(xlim=(-6.5, 39), ylim=(-0.6, 29.5))
    ax.set_aspect("equal")
    ax.axis("off")
    lim = np.abs(z).max()

    def panel(img, x0, y0, cmap, vmin, vmax, peak_at=None):
        n = img.shape[0]
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                  extent=(x0, x0 + n, y0, y0 + n), zorder=1)
        for k in range(n + 1):
            ax.plot([x0, x0 + n], [y0 + k, y0 + k], color=GREY, lw=0.3, zorder=2)
            ax.plot([x0 + k, x0 + k], [y0, y0 + n], color=GREY, lw=0.3, zorder=2)
        ax.add_patch(Rectangle((x0, y0), n, n, facecolor="none",
                               edgecolor="black", lw=0.9, zorder=3))
        if peak_at is not None:
            r, c = peak_at
            ax.add_patch(Rectangle((x0 + c, y0 + n - 1 - r), 1, 1,
                                   facecolor="none", edgecolor=RED, lw=1.8,
                                   zorder=4))

    rows = [(16, x, z, peak, "input $\\mathbf{x}$", "activation map"),
            (0, shifted, z_shifted, peak_s,
             "shifted input", "map of the shifted input")]
    for y0, img, zz, pk, t_in, t_out in rows:
        panel(img, 0, y0, "Greys", 0, 12)
        panel(zz, 22, y0 + 1, "RdBu_r", -lim, lim, pk)
        ax.text(6, y0 + 12.4, t_in, ha="center", va="bottom", fontsize=10.5)
        ax.text(27, y0 + 11.4, t_out, ha="center", va="bottom", fontsize=10.5)
        arrow(ax, (12.8, y0 + 6), (21.2, y0 + 6), colour=PDF_ORANGE, lw=4,
              size=16)
        ax.text(17, y0 + 6.8, "same\n$3\\times3$\nweights", ha="center",
                va="bottom", fontsize=9.5)
    for xa, label in ((-0.8, f"shift by\n({dr}, {dc})"),
                      (33.0, "shifted\nthe same\nway")):
        arrow(ax, (xa, 16.4), (xa, 13.4), colour=GREY, lw=1.6, size=12)
        ax.text(xa - 0.5 if xa < 0 else xa + 0.5, 14.9, label,
                ha="right" if xa < 0 else "left", va="center", fontsize=9.5,
                color=GREY)
    save(fig, "posun-ekvivariance")
    print(f"  equivariance: shift {EQUI_SHIFT}, map peak "
          f"{tuple(map(int, peak))} -> {tuple(map(int, peak_s))}")


POOL_VS_CONV_W = np.ones((3, 3))      # a "learned" filter, all ones here so
                                      # that the outputs can be checked by hand


def pooling_vs_strided_conv():
    """The same 8x8 map: max pooling 2x2 (S=2) against a conv 3x3 (S=2, P=1)."""
    rng = np.random.default_rng(5)
    x = rng.integers(0, 10, (8, 8)).astype(float)
    x[rng.uniform(size=x.shape) < 0.35] = 0          # a ReLU-like map
    pooled = x.reshape(4, 2, 4, 2).max(axis=(1, 3))
    conv, valid = playground_conv(x, POOL_VS_CONV_W, 2, 1)
    # sizes: (8 - 2)/2 + 1 = 4 and (8 - 3 + 2)/2 + 1 = 4.5 -> floor 4
    assert pooled.shape == (4, 4) and conv.shape == (4, 4)
    assert not valid and (8 - 3 + 2 * 1) % 2 == 1
    # the half step that is dropped falls on the zero padding only: the last
    # window covers padded rows 6..8, row 9 is the bottom padding ring
    cover = np.zeros((10, 10), int)
    for r0 in range(0, 8, 2):
        for c0 in range(0, 8, 2):
            cover[r0:r0 + 3, c0:c0 + 3] += 1
    assert (cover[1:9, 1:9] >= 1).all()               # every pixel contributes
    assert cover[1:9, 1:9].max() == 4                 # ... some to four windows
    assert not cover[9, :].any() and not cover[:, 9].any()
    # the highlighted output (1, 1), by hand
    xp = np.pad(x, 1)
    assert pooled[1, 1] == x[2:4, 2:4].max()
    assert conv[1, 1] == xp[2:5, 2:5].sum()
    # every pooled value is one of the four values of its own block
    assert all(pooled[r, c] in x[2 * r:2 * r + 2, 2 * c:2 * c + 2]
               for r in range(4) for c in range(4))

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.set(xlim=(-0.3, 33), ylim=(-0.6, 11.4))
    ax.set_aspect("equal")
    ax.axis("off")
    for r in range(10):
        for c in range(10):
            pad = r in (0, 9) or c in (0, 9)
            ax.add_patch(Rectangle((c, 9 - r), 1, 1,
                                   facecolor=LIGHT_GREY if pad else "white",
                                   edgecolor=GREY, lw=0.5, zorder=1))
            v = 0 if pad else x[r - 1, c - 1]
            ax.text(c + 0.5, 9.5 - r, f"{v:.0f}", ha="center", va="center",
                    fontsize=8.5 if pad else 10,
                    color=GREY if pad or v == 0 else "black", zorder=2)
    ax.text(5, 10.2, "the same $8\\times8$ map (zero padding in grey)",
            ha="center", va="bottom", fontsize=10.5)
    # the windows of the output cell (1, 1)
    ax.add_patch(Rectangle((3, 5), 2, 2, facecolor="none", edgecolor=RED,
                           lw=2.4, zorder=4))
    ax.add_patch(Rectangle((2, 5), 3, 3, facecolor="none", edgecolor=PDF_BLUE,
                           lw=2.0, ls="--", zorder=4))

    def out_grid(x0, y0, values, colour, title, lines):
        for r in range(4):
            for c in range(4):
                ax.add_patch(Rectangle((x0 + c, y0 + 3 - r), 1, 1,
                                       facecolor="white", edgecolor="black",
                                       lw=0.7, zorder=1))
                ax.text(x0 + c + 0.5, y0 + 3.5 - r, f"{values[r, c]:.0f}",
                        ha="center", va="center", fontsize=10, zorder=2)
        ax.add_patch(Rectangle((x0 + 1, y0 + 2), 1, 1, facecolor="none",
                               edgecolor=colour, lw=2.4, zorder=3))
        ax.text(x0 + 2, y0 + 4.15, title, ha="center", va="bottom",
                fontsize=10.5)
        ax.text(x0 + 4.6, y0 + 2, lines, ha="left", va="center", fontsize=10)

    out_grid(14, 5.6, pooled, RED, "max pool $2\\times2$, $S=2$",
             "$4\\times4$, no parameters\nkeeps the largest of 4 values,\n"
             "3 of every 4 are dropped")
    out_grid(14, -0.4, conv, PDF_BLUE, "conv $3\\times3$, $S=2$, $P=1$",
             "$\\lfloor (8-3+2)/2 \\rfloor + 1 = 4$\n9 weights + 1 bias, learned\n"
             "(all ones here); windows overlap,\nevery pixel contributes")
    arrow(ax, (10.4, 6.6), (13.4, 7.6), colour=RED, lw=1.6, size=13)
    arrow(ax, (10.4, 3.4), (13.4, 1.6), colour=PDF_BLUE, lw=1.6, size=13)
    save(fig, "pooling-vs-konvoluce")
    print(f"  pooling vs strided conv: pooled[1,1] = {pooled[1, 1]:.0f}, "
          f"conv[1,1] = {conv[1, 1]:.0f}")


def check_pytorch():
    """The parameter counts quoted on the slide "The same layers in PyTorch"."""
    try:
        import torch
        from torch import nn
    except ImportError:                    # the generator also runs without torch
        print("  torch not installed: the PyTorch slide was not checked")
        return
    count = lambda m: sum(p.numel() for p in m.parameters())   # noqa: E731
    assert count(nn.Conv2d(3, 16, kernel_size=3, padding=1)) == 3 * 3 * 3 * 16 + 16 == 448
    assert count(nn.MaxPool2d(2)) == 0
    bn = nn.BatchNorm2d(16)
    assert count(bn) == 32                                     # gamma_j, beta_j
    assert [n for n, _ in bn.named_buffers()] == ["running_mean", "running_var",
                                                  "num_batches_tracked"]
    assert abs(bn.momentum - 0.1) < 1e-12                      # = 1 - rho
    assert count(nn.Dropout(0.5)) == 0 and count(nn.ReLU()) == 0
    assert count(nn.Linear(8 * 8 * 32, 64)) == 131_136
    # nn.Dropout(p): p is the probability of DROPPING (the slides: of keeping)
    torch.manual_seed(0)
    d = nn.Dropout(0.25).train()
    y = d(torch.ones(100_000))
    assert abs((y == 0).float().mean().item() - 0.25) < 0.01
    assert torch.allclose(y[y > 0], torch.tensor(1 / 0.75))
    assert torch.equal(d.eval()(torch.ones(10)), torch.ones(10))   # prediction
    # the whole network of the table
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(8 * 8 * 32, 64), nn.ReLU(), nn.Linear(64, 10))
    assert count(net) == 148_442
    assert tuple(net[:10](torch.zeros(1, 3, 32, 32)).shape) == (1, 32, 8, 8)
    # the strided convolution of the pooling figure: 8 -> floor(4.5) + ... = 4
    assert tuple(nn.Conv2d(1, 1, 3, stride=2, padding=1)(
        torch.zeros(1, 1, 8, 8)).shape) == (1, 1, 4, 4)
    print(f"  PyTorch (torch {torch.__version__}): 448, 0, 32, 131 136, "
          f"the whole network {count(net):,}")


def check_numbers():
    """The numbers printed on the slides have to agree with the computation."""
    rows = layer_sizes.walk((32, 32, 3))
    params = [count for _, _, count in rows if count]
    assert params == [448, 2320, 4640, 9248, 131_136, 650]
    assert sum(r[2] for r in rows) == 148_442
    # Convolution 32x32x3, F=3, S=1, P=1, K=16 -> 32x32x16, 448 parameters.
    assert (32 - 3 + 2 * 1) // 1 + 1 == 32
    assert 3 * 3 * 3 * 16 + 16 == 448
    # Pooling 32x32x16, F=2, S=2 -> 16x16x16, no parameters.
    assert (32 - 2) // 2 + 1 == 16
    # Page 7 of the PDF.
    assert 200 * 200 * 3 == 120_000 and 120_000 * 50 == 6_000_000
    assert 3 * 3 * 3 == 27 and 27 * 2 == 54
    print("check of the numbers on the slides: OK")


def main() -> None:
    check_numbers()
    check_playgrounds()
    check_pytorch()
    translation_equivariance()
    pooling_vs_strided_conv()
    decision_regions()
    nonlinear_surface()
    shallow_vs_deep()
    vanishing_gradient()
    ann_connections()
    cnn_connections()
    weights_ann_cnn()
    conv_stages()
    conv_example()
    stride_padding()
    activation_layer()
    relu_softmax()
    pooling_example()
    pooling_windows()
    fc_layer()
    dropout_map()
    batchnorm_channel()
    cnn_pipeline()
    layer_parameters()
    feature_hierarchy()
    learned_filters()
    print(f"done: {len(list(OUT.glob('*.svg')))} SVG and "
          f"{len(list(OUT.glob('*.gif')))} GIF in {OUT}")


if __name__ == "__main__":
    main()
