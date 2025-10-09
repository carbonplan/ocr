from enum import Enum
from typing import Literal


class Environment(str, Enum):
    QA = 'qa'
    STAGING = 'staging'
    PRODUCTION = 'production'


class Platform(str, Enum):
    COILED = 'coiled'
    LOCAL = 'local'


class RiskType(str, Enum):
    """Available risk types for calculation."""

    FIRE = 'fire'


RegionType = Literal['tract', 'county']
