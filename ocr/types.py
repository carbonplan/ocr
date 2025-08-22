from enum import Enum


class Environment(str, Enum):
    QA = 'QA'
    STAGING = 'staging'
    PROD = 'prod'


class Platform(str, Enum):
    COILED = 'coiled'
    LOCAL = 'local'


class RiskType(str, Enum):
    """Available risk types for calculation."""

    FIRE = 'fire'
