"""A calculator of shapes, parameter counts and operations (MAC) for convolutional architectures.

The tenth lecture gives concrete numbers for several architectures — the map
sizes in VGG-16 and in AlexNet, the chain 572 -> ... -> 388 in U-Net, the
parameter counts and the operations in the "Top-1 accuracy vs. Operations" chart.
There is no `torch` in the project, so all these numbers are computed
**analytically** from a description of the layers: the network is written down
as a sequence of layers and the shape of every map, the number of weights and
the number of operations follow from it by plain arithmetic.

The conventions (the same ones used by benchmark papers and by libraries such
as `thop`):

* **params** = weights + biases + batch normalization parameters,
* **operations** are counted as **multiply–accumulate (MAC)**, that is
  `k*k*Cin/g*Cout*Hout*Wout` for a convolution and `Cin*Cout` for a fully
  connected layer; activations, pooling and batch norm are neglected. Whoever
  counts the multiplications and the additions separately gets twice as many.

Verification: the computed parameter counts are compared with the published
values by an `assert` (AlexNet 61.1 M, VGG-16 138.4 M, ResNet-18 11.7 M,
ResNet-50 25.6 M, MobileNet-v2 3.5 M). If they match exactly to the unit,
the description of the layers is right. The U-Net paper gives no parameter
count; its 31 030 658 and the 60 965 224 of the 2012 AlexNet are our own
computation, asserted exactly so that a change is noticed.

To run:

    uv run python lectures/L10/code/parameter_counts.py
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    shape: tuple[int, ...]  # shape of the output, (C, H, W) or (N,)
    params: int
    macs: int


@dataclass
class Net:
    """A network built up step by step: holds the current tensor shape and the layers."""

    c: int
    h: int
    w: int
    name: str = "net"
    layers: list[Layer] = field(default_factory=list)

    # ---- building blocks ------------------------------------------------
    def conv(self, name, k, cout, stride=1, pad=0, groups=1, bias=True):
        hout = (self.h + 2 * pad - k) // stride + 1
        wout = (self.w + 2 * pad - k) // stride + 1
        params = k * k * (self.c // groups) * cout + (cout if bias else 0)
        macs = k * k * (self.c // groups) * cout * hout * wout
        self.c, self.h, self.w = cout, hout, wout
        self._add(name, params, macs)
        return self

    def upconv(self, name, k, cout, stride=2, bias=True):
        """Transposed convolution: the size grows stride times."""
        hout = (self.h - 1) * stride + k
        wout = (self.w - 1) * stride + k
        params = k * k * self.c * cout + (cout if bias else 0)
        macs = k * k * self.c * cout * self.h * self.w
        self.c, self.h, self.w = cout, hout, wout
        self._add(name, params, macs)
        return self

    def pool(self, name, k=2, stride=2, pad=0):
        self.h = (self.h + 2 * pad - k) // stride + 1
        self.w = (self.w + 2 * pad - k) // stride + 1
        self._add(name, 0, 0)
        return self

    def adaptive_pool(self, name, size=1):
        self.h = self.w = size
        self._add(name, 0, 0)
        return self

    def bn(self, name):
        self._add(name, 2 * self.c, 0)
        return self

    def concat(self, name, channels):
        """Concatenation of a skip connection — adds channels, costs nothing."""
        self.c += channels
        self._add(name, 0, 0)
        return self

    def crop(self, name, size):
        self.h = self.w = size
        self._add(name, 0, 0)
        return self

    def flatten(self, name="flatten"):
        self.c, self.h, self.w = self.c * self.h * self.w, 1, 1
        self._add(name, 0, 0)
        return self

    def fc(self, name, cout, bias=True):
        params = self.c * cout + (cout if bias else 0)
        macs = self.c * cout
        self.c = cout
        self._add(name, params, macs)
        return self

    # ---- helpers --------------------------------------------------------
    def _add(self, name, params, macs):
        shape = (self.c,) if self.h == 1 and self.w == 1 else (self.c, self.h, self.w)
        self.layers.append(Layer(name, shape, params, macs))

    @property
    def params(self) -> int:
        return sum(v.params for v in self.layers)

    @property
    def macs(self) -> int:
        return sum(v.macs for v in self.layers)

    def shape(self) -> tuple[int, int, int]:
        return (self.c, self.h, self.w)

    def summary(self) -> str:
        lines = [f"{self.name}: input -> {len(self.layers)} layers",
                 f"{'layer':<18}{'output shape':>20}{'params':>14}{'MACs':>16}"]
        for v in self.layers:
            shape = "x".join(str(i) for i in v.shape)
            lines.append(f"{v.name:<18}{shape:>20}{v.params:>14,}{v.macs:>16,}")
        lines.append(f"{'TOTAL':<18}{'':>20}{self.params:>14,}{self.macs:>16,}")
        lines.append(f"= {self.params / 1e6:.2f} M parameters, "
                     f"{self.macs / 1e9:.2f} GMAC")
        return "\n".join(lines)


# =========================================================================
# Architectures
# =========================================================================

def vgg16(classes: int = 1000) -> Net:
    """VGG-16 — Simonyan & Zisserman, ICLR 2015 (configuration D)."""
    s = Net(3, 224, 224, "VGG-16")
    for repeats, channels in [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]:
        for i in range(repeats):
            s.conv(f"conv{channels}_{i + 1}", 3, channels, pad=1)
        s.pool(f"maxpool {channels}")
    s.flatten()
    s.fc("FC1 (fc6)", 4096).fc("FC2 (fc7)", 4096).fc("FC3 (fc8)", classes)
    return s


def alexnet_paper(classes: int = 1000) -> Net:
    """AlexNet the way it is drawn on the slide and in the 2012 paper.

    Krizhevsky, Sutskever, Hinton, NIPS 2012: input 227x227, 96-256-384-384-256
    channels and the convolutions 2, 4 and 5 split into two groups (the network
    ran on two GPUs). The paper says "60 million parameters".
    """
    s = Net(3, 227, 227, "AlexNet (2012)")
    s.conv("C1 conv 11x11/4", 11, 96, stride=4)
    s.pool("maxpool 3x3/2", 3, 2)
    s.conv("C2 conv 5x5", 5, 256, pad=2, groups=2)
    s.pool("maxpool 3x3/2", 3, 2)
    s.conv("C3 conv 3x3", 3, 384, pad=1)
    s.conv("C4 conv 3x3", 3, 384, pad=1, groups=2)
    s.conv("C5 conv 3x3", 3, 256, pad=1, groups=2)
    s.pool("maxpool 3x3/2", 3, 2)
    s.flatten()
    s.fc("FC6", 4096).fc("FC7", 4096).fc("FC8", classes)
    return s


def alexnet_library(classes: int = 1000) -> Net:
    """AlexNet the way the libraries ship it (torchvision) — the bubble in the chart.

    Channels 64-192-384-256-256 and no grouping; 61.1 M parameters and
    0.71 G-FLOPs, which are exactly the coordinates of the "AlexNet" bubble in
    the benchmark chart.
    """
    s = Net(3, 224, 224, "AlexNet (library)")
    s.conv("conv1 11x11/4", 11, 64, stride=4, pad=2)
    s.pool("maxpool 3x3/2", 3, 2)
    s.conv("conv2 5x5", 5, 192, pad=2)
    s.pool("maxpool 3x3/2", 3, 2)
    s.conv("conv3 3x3", 3, 384, pad=1)
    s.conv("conv4 3x3", 3, 256, pad=1)
    s.conv("conv5 3x3", 3, 256, pad=1)
    s.pool("maxpool 3x3/2", 3, 2)
    s.flatten()
    s.fc("fc6", 4096).fc("fc7", 4096).fc("fc8", classes)
    return s


def unet(size: int = 572, in_channels: int = 1, classes: int = 2) -> Net:
    """U-Net — Ronneberger, Fischer, Brox, MICCAI 2015, Fig. 1.

    The convolutions are "valid" (no padding), so every 3x3 removes 2 pixels in
    each direction; max pooling halves. The size chain 572 -> 570 -> 568 -> 284
    -> ... -> 388 is therefore not copied from the figure but follows from the
    arithmetic.
    """
    s = Net(in_channels, size, size, "U-Net")
    skip = []
    for channels in (64, 128, 256, 512):
        s.conv(f"conv 3x3 -> {channels}", 3, channels)
        s.conv(f"conv 3x3 -> {channels}", 3, channels)
        skip.append((channels, s.h))
        s.pool("max pool 2x2")
    s.conv("conv 3x3 -> 1024 (bottleneck)", 3, 1024)
    s.conv("conv 3x3 -> 1024 (bottleneck)", 3, 1024)
    for channels, skip_size in reversed(skip):
        s.upconv(f"up-conv 2x2 -> {channels}", 2, channels)
        s.concat(f"copy and crop ({skip_size} -> {s.h})", channels)
        s.conv(f"conv 3x3 -> {channels}", 3, channels)
        s.conv(f"conv 3x3 -> {channels}", 3, channels)
    s.conv("conv 1x1 -> classes", 1, classes)
    return s


def _resnet_block_basic(s: Net, cout: int, stride: int, first: bool) -> None:
    cin, h_in, w_in = s.c, s.h, s.w
    s.conv("conv 3x3", 3, cout, stride=stride, pad=1, bias=False).bn("bn")
    s.conv("conv 3x3", 3, cout, pad=1, bias=False).bn("bn")
    if first and (stride != 1 or cin != cout):
        h, w = s.h, s.w
        s.c, s.h, s.w = cin, h_in, w_in
        s.conv("downsample 1x1", 1, cout, stride=stride, bias=False).bn("bn")
        s.h, s.w = h, w


def _resnet_block_bottleneck(s: Net, mid: int, stride: int, first: bool) -> None:
    cin, h_in, w_in = s.c, s.h, s.w
    cout = 4 * mid
    s.conv("conv 1x1", 1, mid, bias=False).bn("bn")
    s.conv("conv 3x3", 3, mid, stride=stride, pad=1, bias=False).bn("bn")
    s.conv("conv 1x1", 1, cout, bias=False).bn("bn")
    if first:
        h, w = s.h, s.w
        s.c, s.h, s.w = cin, h_in, w_in
        s.conv("downsample 1x1", 1, cout, stride=stride, bias=False).bn("bn")
        s.h, s.w = h, w


def resnet(depth: int = 50, classes: int = 1000) -> Net:
    """ResNet-18 / ResNet-50 — He et al., CVPR 2016."""
    plan = {18: ((2, 2, 2, 2), "basic"), 50: ((3, 4, 6, 3), "bottleneck")}[depth]
    blocks, kind = plan
    s = Net(3, 224, 224, f"ResNet-{depth}")
    s.conv("conv1 7x7/2", 7, 64, stride=2, pad=3, bias=False).bn("bn1")
    s.pool("maxpool 3x3/2", 3, 2, pad=1)
    for i, (count, width) in enumerate(zip(blocks, (64, 128, 256, 512))):
        for j in range(count):
            stride = 2 if (i > 0 and j == 0) else 1
            if kind == "basic":
                _resnet_block_basic(s, width, stride, j == 0)
            else:
                _resnet_block_bottleneck(s, width, stride, j == 0)
    s.adaptive_pool("avgpool").flatten()
    s.fc("fc", classes)
    return s


def mobilenet_v2(classes: int = 1000) -> Net:
    """MobileNet-v2 — Sandler et al., CVPR 2018 (width 1.0)."""
    s = Net(3, 224, 224, "MobileNet-v2")
    s.conv("conv 3x3/2", 3, 32, stride=2, pad=1, bias=False).bn("bn")
    # (expansion t, output channels c, number of repeats n, stride s)
    plan = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
            (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    for t, c, n, stride in plan:
        for i in range(n):
            step = stride if i == 0 else 1
            hidden = s.c * t
            if t != 1:
                s.conv("expand 1x1", 1, hidden, bias=False).bn("bn")
            s.conv("depthwise 3x3", 3, hidden, stride=step, pad=1,
                   groups=hidden, bias=False).bn("bn")
            s.conv("project 1x1", 1, c, bias=False).bn("bn")
    s.conv("conv 1x1", 1, 1280, bias=False).bn("bn")
    s.adaptive_pool("avgpool").flatten()
    s.fc("classifier", classes)
    return s


# =========================================================================
# Recurrent networks — the parameter counts follow from the formulas
# =========================================================================

def params_rnn(d_x: int, d_h: int, d_y: int) -> int:
    """h_t = tanh(W_h h_{t-1} + W_x x_t + b_h), y_t = g(W_y h_t + b_y), one bias per kernel."""
    return d_h * (d_h + d_x) + d_h + d_y * d_h + d_y


def params_lstm(d_x: int, d_h: int, d_y: int) -> int:
    """An LSTM has three gates and a candidate, four kernels of the shape of a plain RNN kernel."""
    return 4 * (d_h * (d_h + d_x) + d_h) + d_y * d_h + d_y


def params_gru(d_x: int, d_h: int, d_y: int) -> int:
    """A GRU: two gates (reset, update) and a candidate — three kernels."""
    return 3 * (d_h * (d_h + d_x) + d_h) + d_y * d_h + d_y


def params_fc_window(length: int, d_x: int, d_h: int, d_y: int) -> int:
    """A fully connected network over a whole window of `length` — for comparison with an RNN."""
    return d_h * (length * d_x) + d_h + d_y * d_h + d_y


# =========================================================================
# Check against the published values
# =========================================================================

PUBLISHED_PARAMS = {
    "AlexNet (library)": 61_100_840,
    "VGG-16": 138_357_544,
    "ResNet-18": 11_689_512,
    "ResNet-50": 25_557_032,
    "MobileNet-v2": 3_504_872,
}


def check() -> None:
    nets = [alexnet_library(), vgg16(), resnet(18), resnet(50), mobilenet_v2()]
    for s in nets:
        expected = PUBLISHED_PARAMS[s.name]
        assert s.params == expected, (s.name, s.params, expected)

    # The AlexNet from the paper has slightly fewer parameters (grouped
    # convolutions); the paper itself speaks about "60 million parameters".
    a = alexnet_paper()
    assert a.params == 60_965_224, a.params
    assert a.layers[0].shape == (96, 55, 55), a.layers[0].shape

    # U-Net: the size chain and the total parameter count according to the paper.
    u = unet()
    sizes = [v.shape[1] for v in u.layers if len(v.shape) == 3]
    assert sizes[:6] == [570, 568, 284, 282, 280, 140], sizes[:6]
    assert u.shape() == (2, 388, 388), u.shape()
    assert u.params == 31_030_658, u.params
    # the slide "where the 31 million parameters sit"
    first_two = u.layers[:2]
    bottleneck = [v for v in u.layers if "bottleneck" in v.name]
    assert sum(v.params for v in first_two) == 37_568
    assert sum(v.params for v in bottleneck) == 14_157_824
    assert round(100 * sum(v.params for v in bottleneck) / u.params) == 46
    assert (round(sum(v.macs for v in first_two) / 1e9, 2),
            round(sum(v.macs for v in bottleneck) / 1e9, 2)) == (12.08, 11.65)
    level512 = [v for v in u.layers if v.name.endswith("-> 512")] + bottleneck
    assert round(sum(v.params for v in level512) / 1e6, 1) == 26.9
    assert round(100 * sum(v.params for v in level512) / u.params) == 87

    # VGG-16: the shape after the last block and the vector length after flatten.
    v = vgg16()
    shapes = [x.shape for x in v.layers if x.name.startswith("maxpool")]
    assert [t[1] for t in shapes] == [112, 56, 28, 14, 7], shapes
    assert 15.4e9 < v.macs < 15.6e9, v.macs
    print("check OK")


def main() -> None:
    check()
    for net in (vgg16(), alexnet_paper(), unet()):
        print()
        print(net.summary())

    print()
    print(f"{'model':<22}{'params [M]':>12}{'GMAC':>10}")
    for net in (alexnet_library(), vgg16(), resnet(18), resnet(50),
                mobilenet_v2()):
        print(f"{net.name:<22}{net.params / 1e6:>12.2f}{net.macs / 1e9:>10.2f}")

    print()
    d_x, d_h, d_y = 3, 64, 1
    print(f"RNN  ({d_x} inputs, {d_h} hidden): {params_rnn(d_x, d_h, d_y):,} params")
    print(f"LSTM ({d_x} inputs, {d_h} hidden): {params_lstm(d_x, d_h, d_y):,} params")
    print(f"GRU  ({d_x} inputs, {d_h} hidden): {params_gru(d_x, d_h, d_y):,} params")
    for length in (100, 1000):
        print(f"FC over a window of length {length:>5}: "
              f"{params_fc_window(length, d_x, d_h, d_y):,} params")


if __name__ == "__main__":
    main()
