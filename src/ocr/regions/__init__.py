from enum import StrEnum

from ocr.config import StorageConfig

TIGER_VINTAGE = '2026'


class RegionLevel(StrEnum):
    """Census TIGER level. The value is its folder under input/regions/census-tiger/ (written by ocr.regions.tiger)."""

    NATION = 'nation'
    STATE = 'states'
    COUNTY = 'counties'
    TRACT = 'tracts'
    BLOCK = 'blocks'

    @property
    def key(self) -> str:
        return f'input/regions/census-tiger/{self}/{self}.parquet'

    @property
    def uri(self) -> str:
        return StorageConfig().uri(self.key)

    @property
    def group_key(self) -> str:
        """SQL for this level's GEOID from a block GEOID. Each level is a prefix of the block:

        06 001 000100 1000[A]
        |  |   |      +- block (+ optional suffix) -> GEOID
        |  |   +- tract                             -> left(GEOID, 11)
        |  +- county                                -> left(GEOID, 5)
        +- state                                    -> left(GEOID, 2)
        """
        return {
            RegionLevel.NATION: "'US'",
            RegionLevel.STATE: 'left(GEOID, 2)',
            RegionLevel.COUNTY: 'left(GEOID, 5)',
            RegionLevel.TRACT: 'left(GEOID, 11)',
            RegionLevel.BLOCK: 'GEOID',
        }[self]
