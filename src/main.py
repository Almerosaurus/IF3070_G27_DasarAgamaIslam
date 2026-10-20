import random
import math
import utils

HORIZONTAL = "Horizontal"
VERTICAL = "Vertical"

@dataclass
class Vehicle:
    id: str
    w: int
    l: int
    shipping_fee: int
    weight: int
    eta: int


@dataclass
class Ship:
    w: int
    l: int
    max_capacity: int


@dataclass
class Placement:
    x: int = 0
    y: int = 0
    orientation: str = VERTICAL

    @property
    def inside(self) -> bool:
        return self.x is not None and self.y is not None
