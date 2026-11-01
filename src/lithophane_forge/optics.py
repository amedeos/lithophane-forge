"""How much light goes through the plastic, and the thickness that gives a wanted brightness.

Light that crosses t mm of filament falls off exponentially (Beer-Lambert):
T(t) = exp(-t / a). We describe a filament by `td`, the thickness (mm) that lets 10 % of
the light through, so a = td / ln(10). A thicker td means a more translucent filament.

The numbers below are estimates for white filament. Print the calibration strip and
measure your own spool: the right t_min, t_max and td matter more than any image
processing.
"""

from dataclasses import dataclass, replace
from math import exp, log

import numpy as np

LN10 = log(10)


@dataclass(frozen=True)
class Optics:
    t_min: float  # mm, the brightest parts (thin, but still strong enough)
    t_max: float  # mm, the darkest parts
    td: float  # mm of filament that lets 10 % of the light through

    @property
    def attenuation(self) -> float:
        """Thickness (mm) that divides the light by e."""
        return self.td / LN10

    @property
    def floor(self) -> float:
        """Light through t_max relative to t_min: the darkest black the print can show."""
        return exp(-(self.t_max - self.t_min) / self.attenuation)

    def contrast(self) -> float:
        return 1 / self.floor

    def check(self) -> None:
        if not 0.3 <= self.t_min < self.t_max:
            raise ValueError(f"need 0.3 <= t_min < t_max (got {self.t_min}, {self.t_max})")
        if self.td <= 0:
            raise ValueError(f"td must be positive (got {self.td})")

    def thickness(self, light: np.ndarray) -> np.ndarray:
        """Wanted light (0 = black, 1 = brightest; linear light) -> thickness in mm."""
        light = np.clip(light, 0.0, 1.0)
        transmit = self.floor + (1 - self.floor) * light
        return np.clip(self.t_min - self.attenuation * np.log(transmit), self.t_min, self.t_max)

    def light(self, thickness: np.ndarray) -> np.ndarray:
        """Light through `thickness`, relative to t_min (1 at t_min)."""
        return np.exp(-(np.asarray(thickness) - self.t_min) / self.attenuation)


# Estimates for white filament (to be replaced by your calibration).
DEFAULTS = {
    "petg": Optics(t_min=0.8, t_max=3.6, td=2.3),
    "pla": Optics(t_min=0.8, t_max=3.0, td=1.6),
}


def optics(material: str, t_min=None, t_max=None, td=None) -> Optics:
    o = DEFAULTS[material]
    changes = {k: v for k, v in (("t_min", t_min), ("t_max", t_max), ("td", td)) if v is not None}
    o = replace(o, **changes)
    o.check()
    return o
