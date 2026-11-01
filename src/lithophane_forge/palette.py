"""Colour mode: find the filaments and the heights of the filament changes for a photo.

A printer with one extruder changes colour only between layers (M600). The panel prints
lying, and every point of it is a stack: filament 1 from the bed up to z1, filament 2 up
to z2, and so on, cut at the point's own height. Lit from behind, the light crosses the
whole stack, so the colours a point can show lie on **one path** (filament 1 alone, then
1 + 2, then 1 + 2 + 3 ...), from light to dark. We look for the path that fits the photo
best:

1. The ideal colours: k-means in CIELAB gives a start; then an optimiser moves the colours
   and the change heights to bring the path as close as it can to the photo's pixels.
   These colours do not depend on any filament you own.
2. The real filaments (Prusament PETG, from Prusa's published hex and TD values, or your
   own list with --filaments): the ones that bring the whole print closest to the photo,
   searched from the ideal colours and from the k-means colours. The closest filament to
   each ideal colour is also reported.
3. The print: the change heights fitted again for those real filaments, on layer tops;
   filaments that no point holds are dropped.

Light model (Beer-Lambert, per channel): a filament with linear colour c and transmission
distance td lets through exp(-t * a) per channel, with a = ln(10) / td * ln(c) / ln(Y(c)).
For a grey, a is the same on every channel and t = td lets 10 % through; a coloured
filament lets its own colour through more than the others.
"""

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

from .image import linear_to_srgb, srgb_to_linear

LN10 = np.log(10)
WHITE_TD = 3.4  # Prusament PETG Signal White: the reference for "fully bright"
WHITE_HEX = "#E5E0E1"  # the same white: the filament colour of mono prints in the project
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
    return np.interp(lstar, [15, 35, 55, 75, 95], [0.3, 0.6, 1.5, 2.6, WHITE_TD])


def heights(t_min: float, t_max: float, first: float, layer: float) -> np.ndarray:
    """Layer tops from t_min to t_max (the heights a point can have)."""
    n0 = max(0, int(np.ceil((t_min - first) / layer - 1e-9)))
    n1 = int(np.floor((t_max - first) / layer + 1e-9))
    return first + layer * np.arange(n0, n1 + 1)


def stack(att: np.ndarray, starts: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Light through stacks of height h (n,) -> (n, 3), relative to the LED.

    Filament k fills z from starts[k] to starts[k+1] (starts[0] = 0)."""
    ends = np.append(starts[1:], np.inf)
    depth = np.clip(h[:, None] - starts[None, :], 0, (ends - starts)[None, :])
    return np.exp(-depth @ att)


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
    return float(np.exp(-h_min * LN10 / WHITE_TD))


def _curve(att, starts, h, white) -> np.ndarray:
    return stack(att, starts, h) / white


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

    def __init__(self, target_lab: np.ndarray, h: np.ndarray, sample: int = 3000, seed: int = 0):
        flat = target_lab.reshape(-1, 3)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(flat), min(sample, len(flat)), replace=False)
        self.sample = flat[idx]
        self._w = self.sample * WEIGHTS
        self._w2 = (self._w**2).sum(1)
        self.h = h
        self.white = _exposure(h[0])

    def cost(self, att: np.ndarray, starts: np.ndarray) -> float:
        """Mean distance from the sampled pixels to the path (the same weighted CIELAB
        distance that `assign` uses to pick each point's height)."""
        curve = lab(_curve(att, starts, self.h, self.white)) * WEIGHTS
        d2 = self._w2[:, None] + (curve**2).sum(1)[None, :] - 2 * self._w @ curve.T
        return float(np.sqrt(np.maximum(d2.min(1), 0)).mean())

    def cost_of(self, filaments: list[Filament], starts: np.ndarray) -> float:
        lin = np.array([f.linear for f in filaments])
        return self.cost(attenuation(lin, np.array([f.td for f in filaments])), starts)

    def _starts(self, steps: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], self.h[0] + np.cumsum(np.abs(steps))])

    def _steps(self, starts: np.ndarray) -> np.ndarray:
        return np.maximum(np.diff(np.concatenate([[self.h[0]], starts[1:]])), 0.05)

    def ideal(
        self, lin0: np.ndarray, td0: np.ndarray, starts0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Free colours (linear), transparencies (td) and change heights, starting from the
        given ones; never worse than the start.

        The base (filament 1) lies behind every point and is all you see in the brightest
        parts, so it stays a light, neutral, translucent filament (white or natural): only
        its grey level and its td move. The other filaments are free."""
        from scipy.optimize import minimize

        k = len(lin0)
        span = self.h[-1] - self.h[0]

        def unpack(x):
            base = np.full((1, 3), x[0])
            rest = np.clip(x[1 : 1 + 3 * (k - 1)].reshape(k - 1, 3), 0.01, 0.99)
            td = np.exp(x[3 * k - 2 : 4 * k - 2])
            return np.concatenate([base, rest]), td, self._starts(x[4 * k - 2 :])

        def f(x):
            lin, td, starts = unpack(x)
            return self.cost(attenuation(lin, td), starts)

        td0 = np.clip(np.asarray(td0, dtype=float), TD_RANGE[0], TD_RANGE[1])
        td0[0] = np.clip(td0[0], BASE_TD[0], BASE_TD[1])
        grey0 = float(np.clip(luminance(lin0[0]), BASE_GREY[0], BASE_GREY[1]))
        x0 = np.concatenate(
            [[grey0], np.clip(lin0[1:], 0.01, 0.99).ravel(), np.log(td0), self._steps(starts0)]
        )
        bounds = (
            [BASE_GREY]
            + [(0.01, 0.99)] * (3 * (k - 1))
            + [tuple(np.log(BASE_TD))]
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

    def heights_for(self, filaments: list[Filament], starts0: np.ndarray) -> np.ndarray:
        """Change heights for fixed filaments; never worse than starts0."""
        from scipy.optimize import minimize

        if len(filaments) == 1:
            return np.array([0.0])
        lin = np.array([f.linear for f in filaments])
        att = attenuation(lin, np.array([f.td for f in filaments]))
        span = self.h[-1] - self.h[0]
        x0 = self._steps(starts0)
        res = minimize(
            lambda x: self.cost(att, self._starts(x)),
            x0,
            method="Powell",
            bounds=[(0.05, span)] * (len(filaments) - 1),
            options={"maxfev": 2000, "xtol": 1e-3, "ftol": 1e-5},
        )
        best = res.x if res.fun < self.cost(att, self._starts(x0)) else x0
        return self._starts(best)

    def choose(
        self,
        targets: list[tuple[list[Filament], np.ndarray]],
        choices: list[Filament],
        rounds: int = 2,
    ) -> tuple[list[Filament], np.ndarray]:
        """Real filaments for target colours (with their change heights): start from the
        closest colours, then change one position at a time to the filament that makes the
        whole print closest to the photo, and fit the heights again. That can get stuck
        (a dark filament low in the stack hides every change above it), so it runs once per
        target and keeps the best. The base comes from the light, neutral, translucent
        filaments, if the list has any."""
        bases = [c for c in choices if is_base(c)] or choices
        found = []
        for wanted, starts in targets:
            options = [bases] + [choices] * (len(wanted) - 1)
            picked = [nearest(f.linear, opts)[0] for f, opts in zip(wanted, options)]
            for _ in range(rounds):
                for i, opts in enumerate(options):
                    costs = [self.cost_of(picked[:i] + [c] + picked[i + 1 :], starts) for c in opts]
                    picked[i] = opts[int(np.argmin(costs))]
                starts = self.heights_for(picked, starts)
            found.append((self.cost_of(picked, starts), picked, starts))
        _, picked, starts = min(found, key=lambda f: f[0])
        return picked, starts


WEIGHTS = np.array([1.5, 1.0, 1.0])  # lightness counts a little more than hue
TD_RANGE = (0.2, 12.0)  # mm: from black to very translucent filament
BASE_TD = (2.0, 12.0)  # the base lets the light through
BASE_GREY = (0.5, 0.99)  # linear grey level of the base: white to light grey


def is_base(f: Filament) -> bool:
    """A light, neutral, translucent filament: it can be the base of a colour print."""
    lab_ = lab(f.linear)
    return bool(lab_[0] >= 80 and np.hypot(lab_[1], lab_[2]) <= 12 and f.td >= BASE_TD[0])


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
    """Cluster centres in CIELAB, lightest first."""
    from scipy.cluster.vq import kmeans2

    centres, _ = kmeans2(sample, k, seed=seed, minit="++")
    return centres[np.argsort(-centres[:, 0])]


def evaluate(filaments, starts, h, target_lab) -> Fit:
    lin = np.array([f.linear for f in filaments])
    att = attenuation(lin, np.array([f.td for f in filaments]))
    colours = _curve(att, starts, h, _exposure(h[0]))
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
            f = Filament(str(row["name"]), str(row["hex"]), float(row["td"]), abrasive)
            hex_to_linear(f.hex)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"bad filament entry {row!r}: {e}") from None
        if f.td <= 0:
            raise ValueError(f"{f.name}: td must be positive")
        out.append(f)
    if not out:
        raise ValueError("no filaments in the list")
    return out


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


def _merge(filaments: list[Filament], starts: np.ndarray):
    """The same filament twice in a row is one filament."""
    keep = [0] + [i for i in range(1, len(filaments)) if filaments[i] != filaments[i - 1]]
    return [filaments[i] for i in keep], starts[keep]


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
    starts = np.concatenate([[0.0], _snap(starts[1:], h)])
    keep = _used(starts, h)
    return [filaments[i] for i in keep], starts[keep], keep


def plan(
    target_lab: np.ndarray, k: int, h: np.ndarray, choices: list[Filament], source: str
) -> Plan:
    """Ideal colours for the photo, the closest real filaments, and their change heights.

    The ideal colours come from the photo alone: `choices` is used only after them, to
    pick the real filaments. Fewer than k filaments may come back when the photo does not
    need them all."""
    fitter = Fitter(target_lab, h)
    span = h[-1] - h[0]
    even = np.concatenate([[0.0], h[0] + span * np.arange(1, k) / k])

    def cost(lin, td, starts):
        return fitter.cost(attenuation(lin, td), starts)

    def as_filaments(lin, td):
        return [
            Filament(f"ideal {i + 1}", linear_to_hex(c), float(t))
            for i, (c, t) in enumerate(zip(lin, td))
        ]

    best, kmeans = None, []
    for seed in (0, 1):
        lin0 = np.clip(lab_to_linear(kmeans_lab(fitter.sample, k, seed)), 0.01, 0.99)
        kmeans.append((as_filaments(lin0, ideal_td(lin0)), even))
        found = fitter.ideal(lin0, ideal_td(lin0), even)
        if best is None or cost(*found) < cost(*best):
            best = found
    lin, td, starts = best
    # the real filaments: from the ideal colours, and from the photo's own (k-means) colours
    picked, real_starts = fitter.choose([(as_filaments(lin, td), starts), *kmeans], choices)
    ideal = [
        Filament(f"ideal {i + 1}", linear_to_hex(c), round(float(t), 2))
        for i, (c, t) in enumerate(zip(lin, td))
    ]
    ideal, starts, _ = _on_layers(ideal, starts, h)
    matches = [nearest(f.linear, choices) for f in ideal]
    real, real_starts = _merge(picked, real_starts)
    real, real_starts, _ = _on_layers(real, real_starts, h)
    real, real_starts = _merge(real, real_starts)
    ideal_fit = evaluate(ideal, starts, h, target_lab)
    return Plan(ideal_fit, evaluate(real, real_starts, h, target_lab), matches, source)


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
