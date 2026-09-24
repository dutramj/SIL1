import numpy as np
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Loads:
    fx : np.float64
    fy : np.float64
    fz : np.float64
    l : np.float64
    m : np.float64
    n : np.float64

    @property
    def force(self) -> np.ndarray:
        return np.array([[self.fx], [self.fy], [self.fz]])
    
    @property
    def moment(self) -> np.ndarray:
        return np.array([[self.l], [self.m], [self.n]])

    @property
    def vector(self) -> np.ndarray:
        return np.array([[self.fx], [self.fy], [self.fz], [self.l], [self.m], [self.n]])
    
    @property
    def force_norm(self) -> np.float64:
        return np.linalg.norm([self.fx, self.fy, self.fz])

    def __add__(self, other):
        return Loads(
            self.fx + other.fx,
            self.fy + other.fy,
            self.fz + other.fz,
            self.l + other.l,
            self.m + other.m,
            self.n + other.n,
        )