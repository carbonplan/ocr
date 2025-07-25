from enum import Enum


class Branch(str, Enum):
    QA = 'QA'
    PROD = 'prod'


class Platform(str, Enum):
    COILED = 'coiled'
    LOCAL = 'local'


class RiskType(str, Enum):
    """Available risk types for calculation."""

    WIND = 'wind'
