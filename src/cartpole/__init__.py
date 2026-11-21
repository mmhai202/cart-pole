"""Cart-pole physics and control."""

from .controller import control, design_lqr
from .model import Plant, DoublePlant, dynamics, linearize

__all__ = ["Plant", "DoublePlant", "dynamics", "linearize", "control", "design_lqr"]
