from enum import StrEnum
from typing import Literal


class Environment(StrEnum):
    QA = 'qa'
    STAGING = 'staging'
    PRODUCTION = 'production'


class Platform(StrEnum):
    COILED = 'coiled'
    LOCAL = 'local'


class RiskType(StrEnum):
    """Available risk types for calculation."""

    FIRE = 'fire'


RegionType = Literal['tract', 'county']
