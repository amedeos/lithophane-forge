"""Colour mode: find the filaments and the heights of the filament changes for a photo.

A printer with one extruder changes colour only between layers (M600). The panel prints
lying, and every point of it is a stack: filament 1 from the bed up to z1, filament 2 up
to z2, and so on, cut at the point's own height. So the colours a point can show lie on
**one path** (filament 1 alone, then 1 + 2, then 1 + 2 + 3 ...). Lit from behind (a
lithophane), the light crosses the whole stack and the path goes from light to dark. Lit
from the front (a picture you look at in room light), you see the top of the stack, and
the path goes from the base colour to the colour of the last filament. We look for the
path that fits the photo best:

1. The ideal colours: k-means in CIELAB gives a start; then an optimiser moves the colours
   and the change heights to bring the path as close as it can to the photo's pixels.
   These colours do not depend on any filament you own.
2. The real filaments (Prusament PETG, from Prusa's published hex and TD values, or your
   own list with --filaments): the ones that bring the whole print closest to the photo,
   searched from the ideal colours and from the k-means colours. The closest filament to
   each ideal colour is also reported.
3. The print: the change heights fitted again for those real filaments, on layer tops;
   filaments that no point holds are dropped.

Light from behind (Beer-Lambert, per channel): a filament with linear colour c and
transmission distance td lets through exp(-t * a) per channel, with
a = ln(10) / td * ln(c) / ln(Y(c)). For a grey, a is the same on every channel and t = td
lets 10 % through; a coloured filament lets its own colour through more than the others.

Light from the front (see `coverage`): a layer of filament of thickness t hides a part
of what lies under it, sqrt(t / (0.1 td)) and all of it from a tenth of td on. The colour
you see moves that far (in sRGB) from the colour under it to the filament's own colour.
The td values in the catalog come from Prusa's HueForge table, which is measured for this
kind of print.
"""

import math
import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np

from .image import linear_to_srgb, srgb_to_linear

LN10 = np.log(10)
WHITE = "Prusament PETG Signal White"  # the reference for "fully bright" (see white())
HEX = re.compile(r"#?[0-9A-Fa-f]{6}")
D65 = np.array([0.95047, 1.0, 1.08883])
M_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def hex_to_linear(colour: str) -> np.ndarray:
    h = colour.strip().lstrip("#")[:6]
    rgb = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255
    return srgb_to_linear(rgb)


def linear_to_hex(lin: np.ndarray) -> str:
    rgb = np.round(linear_to_srgb(np.clip(lin, 0, 1)) * 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def lab(lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (..., 3) -> CIELAB (D65)."""
    xyz = np.clip(lin, 0, None) @ M_XYZ.T / D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz * (29 / 6) ** 2 / 3 + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1
    )


def delta_e2000(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from skimage.color import deltaE_ciede2000

    return deltaE_ciede2000(a, b)


def luminance(lin: np.ndarray) -> np.ndarray:
    return lin @ np.array([0.2126, 0.7152, 0.0722])


@dataclass(frozen=True)
class Filament:
    name: str
    hex: str
    td: float  # mm that let through about 10 % of the light
    abrasive: bool = False  # needs a hardened steel nozzle (for example tungsten-filled)

    @property
    def linear(self) -> np.ndarray:
        return hex_to_linear(self.hex)


def _arrays(filaments: list["Filament"]) -> tuple[np.ndarray, np.ndarray]:
    """Linear colours (k, 3) and td (k,) of filaments."""
    return np.array([f.linear for f in filaments]), np.array([f.td for f in filaments])


def attenuation(lin: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Per-channel attenuation (1/mm) of filaments with linear colours (k, 3) and td (k,)."""
    c = np.clip(lin, 0.01, 0.98)
    y = np.clip(luminance(c), 0.01, 0.98)
    return LN10 / np.asarray(td)[:, None] * np.log(c) / np.log(y)[:, None]


def ideal_td(lin: np.ndarray) -> np.ndarray:
    """A plausible td for a free colour, from its lightness (dark filaments are opaque).

    Fitted by eye to Prusa's PETG table: black about 0.3 mm, mid tones about 1-2.5 mm,
    white 3.4 mm."""
    lstar = lab(lin)[..., 0]
    return np.interp(lstar, [15, 35, 55, 75, 95], [0.3, 0.6, 1.5, 2.6, white().td])


def heights(t_min: float, t_max: float, first: float, layer: float) -> np.ndarray:
    """Layer tops from t_min to t_max (the heights a point can have)."""
    n0 = max(0, int(np.ceil((t_min - first) / layer - 1e-9)))
    n1 = int(np.floor((t_max - first) / layer + 1e-9))
    return first + layer * np.arange(n0, n1 + 1)


@dataclass
class Fit:
    """A path of colours: filaments, where each one starts, and how well it fits."""

    filaments: list[Filament]
    starts: np.ndarray  # z where each filament starts (starts[0] = 0)
    heights: np.ndarray  # candidate heights (layer tops)
    colours: np.ndarray = field(repr=False)  # (n, 3) linear colour seen at each height
    mean_de: float = 0.0  # CIEDE2000 between the photo and the print, over the pixels
    p95_de: float = 0.0

    def colour_changes(self, layer: float) -> list[tuple[float, str]]:
        """PrusaSlicer's M600 list: the top of the first layer of every new filament."""
        return [
            (round(float(z) + layer, 4), f.hex) for z, f in zip(self.starts[1:], self.filaments[1:])
        ]


def _exposure(h_min: float) -> float:
    """The viewer's white: light through the thinnest stack of white PETG."""
    return float(np.exp(-h_min * LN10 / white().td))


LIGHTS = ("front", "back")  # front: a picture in room light; back: a lithophane
COVER = 0.1  # light from the front: a filament hides what lies under it from COVER x td


def coverage(t: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Light from the front: how much of what lies under it a layer of filament of
    thickness t hides (0 to 1).

    Front-lit prints made with HueForge's td values hide the colour below far faster than
    td suggests: white PETG (td 3.4) hides black under about 0.34 mm. The curve
    sqrt(t / (0.1 td)) is the calibrated model of AutoForge
    (https://github.com/hvoss-techfak/AutoForge), without its slower curve for a dark
    filament on a light one. It is an estimate until a test print checks it."""
    return np.minimum(np.sqrt(np.maximum(t, 0) / (COVER * td)), 1.0)


def _depths(starts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """How much of each filament points of height h hold: starts (n, k), h (m,) ->
    (n, m, k). Filament k fills z from starts[k] to starts[k+1]."""
    ends = np.concatenate([starts[:, 1:], np.full((len(starts), 1), np.inf)], axis=1)
    return np.clip(h[None, :, None] - starts[:, None, :], 0, (ends - starts)[:, None, :])


class Backlight:
    """A lithophane: the light crosses the whole stack (Beer-Lambert). The base lies
    behind every point and is all you see in the brightest parts, so it is a light,
    neutral, translucent filament."""

    name = "back"
    grey_base = True  # the ideal base is a grey: only its level moves

    def __init__(self, h: np.ndarray):
        self.white = _exposure(h[0])
        self.base_td = BASE_TD

    def props(self, lin: np.ndarray, td: np.ndarray) -> np.ndarray:
        """What the model needs of filaments with linear colours (k, 3) and td (k,)."""
        return attenuation(lin, td)

    def seen(self, depth: np.ndarray, props: np.ndarray) -> np.ndarray:
        """Linear colours (n, m, 3) of stacks with depths (n, m, k) of filaments (n, k, P),
        relative to the viewer's white."""
        return np.exp(-np.einsum("nhk,nkc->nhc", depth, props)) / self.white

    def is_base(self, f: Filament) -> bool:
        return is_base(f)


class Frontlight:
    """A picture in room light: you see the top of the stack (see `coverage`). The base
    must hide what is behind the panel at its thinnest point, so its td is at most the
    thinnest stack / COVER. Signal White, thick, is the viewer's white."""

    name = "front"
    grey_base = False  # the base can have any colour

    def __init__(self, h: np.ndarray):
        self.white = float(luminance(white().linear))
        self.base_td = (TD_RANGE[0], min(TD_RANGE[1], float(h[0]) / COVER))

    def props(self, lin: np.ndarray, td: np.ndarray) -> np.ndarray:
        """The sRGB colour and the td of each filament: (k, 4)."""
        srgb = linear_to_srgb(np.clip(lin, 0, 1))
        return np.concatenate([srgb, np.asarray(td, float).reshape(-1, 1)], axis=1)

    def seen(self, depth: np.ndarray, props: np.ndarray) -> np.ndarray:
        """From the bed up, each filament moves the colour from the one under it toward its
        own by its coverage. Unrolled: filament k shows with weight cover_k times what the
        filaments above it leave uncovered. Behind the panel is dark (the base hides it)."""
        srgb, td = props[..., :3], props[..., 3]
        cover = coverage(depth, td[:, None, :])  # (n, m, k)
        left = np.cumprod((1 - cover)[..., ::-1], axis=-1)[..., ::-1]  # uncovered from k up
        above = np.concatenate([left[..., 1:], np.ones_like(left[..., :1])], axis=-1)
        out = np.einsum("nmk,nkc->nmc", cover * above, srgb)
        return srgb_to_linear(out) / self.white

    def is_base(self, f: Filament) -> bool:
        return f.td <= self.base_td[1] + 1e-9


def light_model(light: str, h: np.ndarray) -> Backlight | Frontlight:
    if light not in LIGHTS:
        raise ValueError(f"unknown light {light!r}; choose from {', '.join(LIGHTS)}")
    return Frontlight(h) if light == "front" else Backlight(h)


def curve(filaments: list[Filament], starts: np.ndarray, h: np.ndarray, light="back"):
    """The linear colour (m, 3) of the stack at every height h."""
    model = light_model(light, h)
    props = model.props(*_arrays(filaments))
    return model.seen(_depths(np.asarray(starts, float)[None], h), props[None])[0]


def _snapped(starts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """A plan's starts with every change on a layer top (the base keeps 0)."""
    return np.concatenate([[0.0], _snap(np.asarray(starts, float)[1:], h)])


def _snap(z: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Change heights on layer tops, increasing, inside the height range."""
    out = []
    for v in z:
        k = int(np.argmin(np.abs(h - v)))
        if out:
            k = max(k, int(np.searchsorted(h, out[-1])) + 1)
        out.append(h[min(k, len(h) - 1)])
    return np.array(out)


class Fitter:
    """Fits paths of colours to a sample of the photo's pixels."""

    def __init__(
        self,
        target_lab: np.ndarray,
        h: np.ndarray,
        sample: int = 3000,
        seed: int = 0,
        light: str = "back",
    ):
        flat = target_lab.reshape(-1, 3)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), min(sample, len(flat)), replace=False)
        self.sample = flat[idx]
        self._w = self.sample * WEIGHTS
        self._w2 = (self._w**2).sum(1)
        self.h = h
        self.light = light_model(light, h)

    def props(self, filaments: list[Filament]) -> np.ndarray:
        """What the light model needs of each filament (attenuation, or colour and td)."""
        return self.light.props(*_arrays(filaments))

    def cost(self, props: np.ndarray, starts: np.ndarray) -> float:
        """Mean distance from the sampled pixels to the path (the same weighted CIELAB
        distance that `assign` uses to pick each point's height)."""
        starts = np.asarray(starts, float)[None]
        best = self._nearest_d2(props[None], starts, self.h, np.float64)
        return float(np.sqrt(np.maximum(best, 0)).mean())

    def costs(
        self, props: np.ndarray, starts: np.ndarray, h: np.ndarray | None = None, below=None
    ) -> np.ndarray:
        """`cost` for many plans at once: filaments (n, k, P) or one set (k, P) for all,
        and change heights (n, k). With `h`, only those heights are compared, and `below` (per
        sampled pixel) is the smallest squared distance over the other heights. The
        distances are float32: half the memory traffic, which matters in `choose`; `cost`
        (used by the ideal fit) stays float64."""
        h = self.h if h is None else h
        props = np.broadcast_to(props, (len(starts), *props.shape[-2:]))
        rows = max(1, CHUNK // max(1, len(h) * len(self.sample)))
        best = np.empty((len(starts), len(self.sample)))
        for a in range(0, len(starts), rows):
            best[a : a + rows] = self._nearest_d2(props[a : a + rows], starts[a : a + rows], h)
        if below is not None:
            best = np.minimum(best, below[None])
        return np.sqrt(np.maximum(best, 0)).mean(1)

    def _nearest_d2(
        self, props: np.ndarray, starts: np.ndarray, h: np.ndarray, dtype=np.float32
    ) -> np.ndarray:
        """For each plan, each sampled pixel's squared distance to its closest height."""
        seen = self.light.seen(_depths(starts, h), props)
        curve = (lab(seen.reshape(-1, 3)) * WEIGHTS).reshape(-1, 3).astype(dtype)
        w, w2 = self._w.astype(dtype, copy=False), self._w2.astype(dtype, copy=False)
        d2 = (curve**2).sum(1)[:, None] + w2[None, :] - 2 * (curve @ w.T)
        return d2.reshape(len(starts), len(h), -1).min(1).astype(np.float64)

    def cost_of(self, filaments: list[Filament], starts: np.ndarray) -> float:
        return self.cost(self.props(filaments), starts)

    def _starts(self, steps: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], self.h[0] + np.cumsum(np.abs(steps))])

    def _steps(self, starts: np.ndarray) -> np.ndarray:
        return np.maximum(np.diff(np.concatenate([[self.h[0]], starts[1:]])), 0.05)

    def ideal(
        self, lin0: np.ndarray, td0: np.ndarray, starts0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Free colours (linear), transparencies (td) and change heights, starting from the
        given ones; never worse than the start.

        Light from behind: the base (filament 1) lies behind every point and is all you see
        in the brightest parts, so it stays a light, neutral, translucent filament (white or
        natural): only its grey level and its td move. Light from the front: the base can
        have any colour, but it must hide what is behind the panel (`Frontlight`). The other
        filaments are free."""
        from scipy.optimize import minimize

        k = len(lin0)
        span = self.h[-1] - self.h[0]
        nb = 1 if self.light.grey_base else 3  # numbers for the base colour
        base_td = self.light.base_td

        def unpack(x):
            base = np.full((1, 3), x[0]) if nb == 1 else np.clip(x[:3], 0.01, 0.99)[None]
            rest = np.clip(x[nb : nb + 3 * (k - 1)].reshape(k - 1, 3), 0.01, 0.99)
            o = nb + 3 * (k - 1)
            td = np.exp(x[o : o + k])
            return np.concatenate([base, rest]), td, self._starts(x[o + k :])

        def f(x):
            lin, td, starts = unpack(x)
            return self.cost(self.light.props(lin, td), starts)

        td0 = np.clip(np.asarray(td0, dtype=float), TD_RANGE[0], TD_RANGE[1])
        td0[0] = np.clip(td0[0], base_td[0], base_td[1])
        if nb == 1:
            base0 = [float(np.clip(luminance(lin0[0]), BASE_GREY[0], BASE_GREY[1]))]
            base_bounds = [BASE_GREY]
        else:
            base0 = list(np.clip(lin0[0], 0.01, 0.99))
            base_bounds = [(0.01, 0.99)] * 3
        x0 = np.concatenate(
            [base0, np.clip(lin0[1:], 0.01, 0.99).ravel(), np.log(td0), self._steps(starts0)]
        )
        bounds = (
            base_bounds
            + [(0.01, 0.99)] * (3 * (k - 1))
            + [tuple(np.log(base_td))]
            + [tuple(np.log(TD_RANGE))] * (k - 1)
            + [(0.05, span)] * (k - 1)
        )
        res = minimize(
            f,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxfev": 6000, "xtol": 1e-3, "ftol": 1e-5},
        )
        return unpack(res.x if res.fun < f(x0) else x0)

    def choose(
        self,
        targets: list[tuple[list[Filament], np.ndarray]],
        choices: list[Filament],
        rounds: int = 6,
    ) -> tuple[list[Filament], np.ndarray]:
        """Real filaments for target colours (with their change heights).

        Start from the closest colours, with the changes on layer tops. Then take one
        position at a time. Try every filament there, at every layer top between the
        changes below and above it, and keep the best pair. Repeat until nothing changes.
        Every change stays on a layer top, so every cost we compare is the cost of a plan
        that can print. (Choosing the filament with the heights fixed got stuck: a dark
        filament low in the stack hides every change above it.) The search runs once per
        target and keeps the best. The base comes from the filaments that can be one
        (`Backlight.is_base`, `Frontlight.is_base`), if the list has any."""
        bases = [c for c in choices if self.light.is_base(c)] or choices
        found = []
        for wanted, starts in targets:
            options = [bases] + [choices] * (len(wanted) - 1)
            picked = [nearest(f.linear, opts)[0] for f, opts in zip(wanted, options)]
            starts = _snapped(starts, self.h)
            best = float(self.costs(self.props(picked), starts[None])[0])  # as the moves
            for _ in range(rounds):
                moved = False
                for i, opts in enumerate(options):
                    found_i = self._best_move(i, opts, picked, starts)
                    if found_i is not None and found_i[0] < best - 1e-9:
                        best, picked, starts = found_i
                        moved = True
                if not moved:
                    break
            found.append((best, picked, starts))
        _, picked, starts = min(found, key=lambda f: f[0])
        return picked, starts

    def _best_move(self, i: int, options: list[Filament], picked: list[Filament], starts):
        """The best (cost, filaments, starts) with any filament of `options` at position i,
        starting at any layer top it may use. Heights at or below the change under i do not
        depend on that choice, so their distances are computed once."""
        heights = self._heights(i, starts)
        if not len(heights):
            return None
        floor = starts[i - 1] + 1e-9 if i > 0 else -np.inf
        low, high = self.h[self.h <= floor], self.h[self.h > floor]
        now = self.props(picked)
        below = self._nearest_d2(now[None], starts[None], low)[0] if len(low) else None
        props = np.repeat(now[None], len(options) * len(heights), axis=0)
        props[:, i] = np.repeat(self.props(options), len(heights), axis=0)
        batch = np.tile(starts, (len(options) * len(heights), 1))
        batch[:, i] = np.tile(heights, len(options))
        costs = self.costs(props, batch, high, below)
        j = int(np.argmin(costs))
        trial = [*picked[:i], options[j // len(heights)], *picked[i + 1 :]]
        return float(costs[j]), trial, batch[j].copy()

    def _heights(self, i: int, starts: np.ndarray) -> np.ndarray:
        """The layer tops where position i can start: between the changes below and above
        it, never the top layer (a change there would print nothing). The base starts at 0."""
        if i == 0:
            return np.array([0.0])
        above = starts[i + 1] if i + 1 < len(starts) else np.inf
        tops = self.h[:-1]
        return tops[(tops > starts[i - 1] + 1e-9) & (tops < above - 1e-9)]


WEIGHTS = np.array([1.5, 1.0, 1.0])  # lightness counts a little more than hue
CHUNK = 4_000_000  # distances computed at once in Fitter.costs (about 32 MB)
TD_RANGE = (0.2, 12.0)  # mm: from black to very translucent filament
BASE_TD = (2.0, 12.0)  # the base lets the light through
BASE_GREY = (0.5, 0.99)  # linear grey level of the base: white to light grey


LIGHT_L = 80  # L* from which a filament counts as light


def is_base(f: Filament) -> bool:
    """A light, neutral, translucent filament: it can be the base of a lithophane."""
    lab_ = lab(f.linear)
    return bool(lab_[0] >= LIGHT_L and np.hypot(lab_[1], lab_[2]) <= 12 and f.td >= BASE_TD[0])


def lab_to_linear(lab_: np.ndarray) -> np.ndarray:
    lab_ = np.asarray(lab_, dtype=float)
    fy = (lab_[..., 0] + 16) / 116
    fx = fy + lab_[..., 1] / 500
    fz = fy - lab_[..., 2] / 200

    def inv(f):
        return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))

    xyz = np.stack([inv(fx), inv(fy), inv(fz)], -1) * D65
    return xyz @ np.linalg.inv(M_XYZ).T


def kmeans_lab(sample: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k cluster centres in CIELAB, lightest first. A photo with fewer than k distinct
    colours repeats some of them (scipy warns on empty clusters; the user must not see
    that)."""
    import warnings

    from scipy.cluster.vq import kmeans2

    m = min(k, len(np.unique(np.round(sample, 3), axis=0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        centres, _ = kmeans2(sample, m, seed=seed, minit="++")
    centres = np.concatenate([centres, centres[np.arange(k - m) % m]])
    return centres[np.argsort(-centres[:, 0], kind="stable")]


def evaluate(filaments, starts, h, target_lab, light: str = "back") -> Fit:
    colours = curve(filaments, starts, h, light)
    idx = assign(target_lab, lab(colours))
    de = delta_e2000(target_lab.reshape(-1, 3), lab(colours)[idx.ravel()])
    return Fit(filaments, starts, h, colours, float(de.mean()), float(np.percentile(de, 95)))


def assign(target_lab: np.ndarray, curve_lab: np.ndarray) -> np.ndarray:
    """Index of the best height for every pixel. Lightness counts a little more than hue:
    when two heights give similar colours, the one with the right lightness wins, and the
    surface stays smooth."""
    flat = target_lab.reshape(-1, 3) * WEIGHTS
    curve = curve_lab * WEIGHTS
    out = np.empty(len(flat), dtype=np.int64)
    for i in range(0, len(flat), 200_000):  # in chunks: n x heights distances
        d = np.linalg.norm(flat[i : i + 200_000, None, :] - curve[None, :, :], axis=2)
        out[i : i + 200_000] = d.argmin(1)
    return out.reshape(target_lab.shape[:-1])


def load_filaments(path: Path | None = None) -> list[Filament]:
    """The catalog (Prusament PETG) or a TOML file with [[filament]] name, hex, td, and
    optionally abrasive = true."""
    if path is None:
        text = (
            resources.files("lithophane_forge")
            .joinpath("data/petg_catalog.toml")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    rows = tomllib.loads(text).get("filament", [])
    out = []
    for row in rows:
        try:
            abrasive = row.get("abrasive", False)
            if not isinstance(abrasive, bool):
                raise TypeError("abrasive must be true or false")
            colour = str(row["hex"]).strip()
            if not HEX.fullmatch(colour):
                raise ValueError(f"hex must be #RRGGBB (got {colour!r})")
            # one spelling everywhere: the preview, the M600 list and PrusaSlicer read it
            colour = "#" + colour.lstrip("#").upper()
            td = float(row["td"])
            if not (math.isfinite(td) and td > 0):
                raise ValueError(f"td must be a positive number of mm (got {row['td']!r})")
            out.append(Filament(str(row["name"]), colour, td, abrasive))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"bad filament entry {row!r}: {e}") from None
    if not out:
        raise ValueError("no filaments in the list")
    return out


@cache
def white() -> Filament:
    """Prusament PETG Signal White from the catalog: the viewer's "fully bright" and the
    colour of white PETG prints."""
    return next(f for f in load_filaments() if f.name == WHITE)


CATALOG = "Prusament PETG (Prusa's TD table)"


def choices(path: Path | None = None, allow_abrasive: bool = False) -> tuple[list[Filament], str]:
    """The filaments to pick from (the catalog or your list), and where they come from.

    Abrasive filaments need a hardened steel nozzle, and most printers come with brass:
    they are left out unless allow_abrasive."""
    found = load_filaments(path)
    source = str(path) if path else CATALOG
    if allow_abrasive:
        return found, source
    kept = [f for f in found if not f.abrasive]
    if not kept:
        raise ValueError(
            "every filament in the list is abrasive; pass --allow-abrasive if your nozzle "
            "is hardened steel"
        )
    left_out = [f.name for f in found if f.abrasive]
    if left_out:
        source += (
            f"; left out as abrasive (they need a hardened steel nozzle, --allow-abrasive "
            f"uses them): {', '.join(left_out)}"
        )
    return kept, source


def nearest(colour_lin: np.ndarray, choices: list[Filament]) -> tuple[Filament, float]:
    """The filament closest to a colour (CIEDE2000), and the distance."""
    target = lab(colour_lin)[None, :]
    found = lab(np.array([f.linear for f in choices]))
    de = delta_e2000(np.repeat(target, len(choices), 0), found)
    i = int(np.argmin(de))
    return choices[i], float(de[i])


@dataclass
class Plan:
    """Everything the colour mode decides."""

    ideal: Fit
    real: Fit
    matches: list[tuple[Filament, float]]  # per ideal colour: closest filament, Delta E
    source: str  # where the real filaments come from
    choices: list[Filament] = field(default_factory=list)  # the filaments it could use

    @property
    def available(self) -> int:
        return len(self.choices)


def _merge(filaments: list[Filament], starts: np.ndarray, by_colour: bool = False):
    """The same filament twice in a row is one filament (with by_colour, the same hex
    and td, whatever the name)."""

    def key(f):
        return (f.hex, f.td) if by_colour else f

    keep = [0] + [i for i in range(1, len(filaments)) if key(filaments[i]) != key(filaments[i - 1])]
    return [filaments[i] for i in keep], np.asarray(starts)[keep]


def _used(starts: np.ndarray, h: np.ndarray) -> list[int]:
    """The filaments some point of the print holds. A filament that starts at the top layer
    (snapping can put it there) would only get an M600 above the print, and one that the
    next filament replaces at the same height has no layer at all."""
    idx = [0] + [i for i in range(1, len(starts)) if starts[i] < h[-1] - 1e-9]
    return [
        i for j, i in enumerate(idx) if j + 1 == len(idx) or starts[idx[j + 1]] > starts[i] + 1e-9
    ]


def _on_layers(filaments: list[Filament], starts: np.ndarray, h: np.ndarray):
    """Changes snapped to layer tops, without the filaments that no point holds."""
    starts = _snapped(starts, h)
    keep = _used(starts, h)
    return [filaments[i] for i in keep], starts[keep], keep


def plan(
    target_lab: np.ndarray,
    k: int,
    h: np.ndarray,
    choices: list[Filament],
    source: str,
    light: str = "back",
) -> Plan:
    """Ideal colours for the photo, the closest real filaments, and their change heights.

    The ideal colours come from the photo alone: `choices` is used only after them, to
    pick the real filaments. Fewer than k filaments may come back when the photo does not
    need them all.

    The ideal fit moves colours and heights, not the order of the filaments, so it starts
    from k-means colours in a few orders: lightest first for light from behind (the base
    is light), lightest first and darkest first for light from the front. Its heights are
    free, and they can lose quality when they are snapped to layer tops (in room light the
    fit also stops early: a run thicker than its coverage depth does not depend on its td).
    So the same search that picks real filaments also runs on the photo's own colours
    (`photo_colours`), on layer tops, and Powell refines its result. The ideal is the best
    of all these, compared on layer tops."""
    fitter = Fitter(target_lab, h, light=light)
    span = h[-1] - h[0]
    even = np.concatenate([[0.0], h[0] + span * np.arange(1, k) / k])
    orders = (False, True) if light == "front" else (False,)  # True: darkest first

    def printed_cost(found):  # the cost with the changes on layer tops
        lin, td, starts = found
        return fitter.cost(fitter.light.props(lin, td), _snapped(starts, h))

    found, kmeans = [], []
    for seed in (0, 1):
        centres = kmeans_lab(fitter.sample, k, seed)
        for dark_first in orders:
            lin0 = np.clip(lab_to_linear(centres[::-1] if dark_first else centres), 0.01, 0.99)
            kmeans.append((_ideal(lin0, ideal_td(lin0)), even))
            found.append(fitter.ideal(lin0, ideal_td(lin0), even))
    picked, on_tops = fitter.choose(kmeans[:2], photo_colours(fitter.sample))
    lin, td = _arrays(picked)
    found += [(lin, td, on_tops), fitter.ideal(lin, td, on_tops)]
    lin, td, starts = min(found, key=printed_cost)
    # the real filaments: from the ideal colours, and from the photo's own (k-means) colours
    picked, real_starts = fitter.choose([(_ideal(lin, td), starts), *kmeans[:2]], choices)
    ideal, starts = _merge(_ideal(lin, np.round(td, 2)), starts, by_colour=True)
    ideal, starts, _ = _on_layers(ideal, starts, h)
    ideal, starts = _merge(ideal, starts, by_colour=True)
    ideal = _ideal(*_arrays(ideal))
    matches = [nearest(f.linear, choices) for f in ideal]
    real, real_starts = _merge(picked, real_starts)
    real, real_starts, _ = _on_layers(real, real_starts, h)
    real, real_starts = _merge(real, real_starts)
    ideal_fit = evaluate(ideal, starts, h, target_lab, light)
    real_fit = evaluate(real, real_starts, h, target_lab, light)
    return Plan(ideal_fit, real_fit, matches, source, list(choices))


def _ideal(lin: np.ndarray, td: np.ndarray) -> list[Filament]:
    """Free colours as filaments named "ideal 1", "ideal 2" ..."""
    return [
        Filament(f"ideal {i + 1}", linear_to_hex(c), float(t))
        for i, (c, t) in enumerate(zip(lin, td))
    ]


PHOTO_COLOURS = 8  # k-means colours of the photo for the search of the ideal colours
GREYS = (0.03, 0.2, 0.6, 0.85)  # linear neutral greys added to them
TD_STEPS = (0.5, 1.5)  # each colour at these multiples of its ideal_td


def photo_colours(sample: np.ndarray, n: int = PHOTO_COLOURS) -> list[Filament]:
    """Candidate colours from the photo alone (CIELAB pixels), for the search of the ideal
    colours: its k-means colours and a few neutral greys, each at a few td values. No
    filament list comes in, so the ideal colours still depend on the photo alone."""
    centres = np.clip(lab_to_linear(kmeans_lab(sample, n)), 0.01, 0.99)
    lin = np.concatenate([centres, np.repeat(np.array(GREYS)[:, None], 3, axis=1)])
    return [
        Filament("photo", linear_to_hex(c), float(np.clip(td * f, *TD_RANGE)))
        for c, td in zip(lin, ideal_td(lin))
        for f in TD_STEPS
    ]


def target(rgb: np.ndarray, levels=(0.5, 99.5), gamma: float = 1.0) -> np.ndarray:
    """The photo as the print should show it, in CIELAB: lightness stretched by the levels
    and the gamma, colour kept."""
    from .image import lstar_to_y, y_to_lstar

    lin = srgb_to_linear(rgb)
    y = np.clip(luminance(lin), 1e-6, None)
    lstar = y_to_lstar(y)
    lo, hi = np.percentile(lstar, levels)
    stretched = np.clip((lstar - lo) / max(hi - lo, 1e-6), 0, 1)
    new_y = lstar_to_y(stretched * 100) ** gamma
    return lab(lin * (new_y / y)[..., None])
