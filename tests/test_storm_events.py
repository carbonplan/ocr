import gzip
import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import geoparquet_lod
import pytest

from ocr.duck import duckdb_connect
from ocr.historic import storm_events

COLUMNS = [
    'BEGIN_YEARMONTH', 'BEGIN_DAY', 'END_YEARMONTH', 'END_DAY', 'EPISODE_ID', 'EVENT_ID', 'STATE', 'YEAR',
    'EVENT_TYPE', 'CZ_NAME', 'INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT',
    'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'FLOOD_CAUSE', 'BEGIN_LOCATION', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT',
    'END_LON', 'EVENT_NARRATIVE',
]  # fmt: skip
REPORT = {
    'BEGIN_YEARMONTH': '201905',
    'BEGIN_DAY': '9',
    'END_YEARMONTH': '201905',
    'END_DAY': '10',
    'EPISODE_ID': '137295',
    'EVENT_ID': '824116',
    'STATE': '"TEXAS"',
    'YEAR': '2019',
    'EVENT_TYPE': '"Flash Flood"',
    'CZ_NAME': '"BEXAR"',
    'INJURIES_DIRECT': '"1"',
    'INJURIES_INDIRECT': '"2"',
    'DEATHS_DIRECT': '"0"',
    'DEATHS_INDIRECT': '"1"',
    'DAMAGE_PROPERTY': '"10.00K"',
    'DAMAGE_CROPS': '"2.5M"',
    'FLOOD_CAUSE': '"Heavy Rain"',
    'BEGIN_LOCATION': '"LEON SPGS"',
    'BEGIN_LAT': '29.7158',
    'BEGIN_LON': '-98.7744',
    'END_LAT': '29.7898',
    'END_LON': '-98.6406',
    'EVENT_NARRATIVE': '"Water over the road, which stayed closed."',
}


def row(**changes):
    return ','.join((REPORT | changes)[column] for column in COLUMNS)


ROWS = {
    'flash_flood': row(),
    'flood_without_amounts': row(
        EVENT_ID='2',
        EVENT_TYPE='"Flood"',
        STATE='"MARYLAND"',
        CZ_NAME='"PRINCE GEORGE\'S"',
        BEGIN_LOCATION='',
        FLOOD_CAUSE='',
        DAMAGE_PROPERTY='',
        DAMAGE_CROPS='',
        BEGIN_LAT='38.8',
        BEGIN_LON='-76.9',
    ),
    'tornado': row(EVENT_ID='3', EVENT_TYPE='"Tornado"'),
    'unlocated': row(EVENT_ID='4', BEGIN_LAT='', BEGIN_LON='', END_LAT='', END_LON=''),
    'puerto_rico': row(EVENT_ID='5', STATE='"PUERTO RICO"', BEGIN_LAT='18.2', BEGIN_LON='-66.1'),
    'quotes_left_unescaped': row(
        EVENT_ID='6', EVENT_TYPE='"Waterspout"', BEGIN_LOCATION='"S OF "7 MILE" BRIDGE"'
    ),
    'after_unescaped_quotes': row(EVENT_ID='7'),
}
NEXT_YEAR = row(EVENT_ID='8', YEAR='2020', BEGIN_YEARMONTH='202003', END_YEARMONTH='202003')
KEPT = [2, 7, 8, 824116]


def write_details(path, rows):
    with gzip.open(path, 'wt') as f:
        f.write('\n'.join([','.join(COLUMNS), *rows]) + '\n')
    return path


def details_name(year, stamp):
    return f'StormEvents_details-ftp_v1.0_d{year}_c{stamp}.csv.gz'


@pytest.fixture
def csv_paths(tmp_path):
    return [
        write_details(tmp_path / details_name(2019, 20260323), ROWS.values()),
        write_details(tmp_path / details_name(2020, 20260918), [NEXT_YEAR]),
    ]


def test_latest_files_take_each_year_from_1996_at_its_newest_stamp():
    listed = [
        details_name(1995, 20260323),
        details_name(2019, 20260323),
        details_name(2020, 20260323),
        details_name(2020, 20260918),
        details_name(2020, 20260625),
        'StormEvents_fatalities-ftp_v1.0_d2020_c20260918.csv.gz',
        'StormEvents_locations-ftp_v1.0_d2020_c20260918.csv.gz',
    ]
    index = ''.join(f'<td><a href="{name}">{name}</a></td>\n' for name in listed)

    assert storm_events.latest_files(index) == {
        2019: details_name(2019, 20260323),
        2020: details_name(2020, 20260918),
    }


def test_output_key_carries_the_newest_creation_stamp():
    names = [
        details_name(2019, 20260323),
        details_name(2020, 20260918),
        details_name(2021, 20260625),
    ]

    day = storm_events.source_date(names)

    assert day == date(2026, 9, 18)
    assert (
        storm_events.key_for(day)
        == 'input/historic/storm_events/20260918/storm_events_floods.parquet'
    )
    assert storm_events.raw_key(names[0]) == f'input/historic/storm_events/raw/{names[0]}'


def test_reports_fields(csv_paths):
    with duckdb_connect() as con:
        rel = storm_events.reports(con, csv_paths)
        fields = rel.columns
        report = rel.filter('event_id = 824116').project(
            '* EXCLUDE (geometry, bbox), ST_AsText(geometry), bbox'
        ).fetchall()  # fmt: skip

    assert fields == [
        'event_id', 'episode_id', 'type', 'begin_date', 'end_date', 'year', 'state', 'county', 'location',
        'cause', 'damage', 'deaths', 'injuries', 'end_lon', 'end_lat', 'geometry', 'bbox',
    ]  # fmt: skip
    assert report == [
        (
            824116,
            137295,
            'Flash Flood',
            '2019-05-09',
            '2019-05-10',
            2019,
            'Texas',
            'Bexar',
            'Leon Spgs',
            'Heavy Rain',
            2_510_000,
            1,
            3,
            -98.6406,
            29.7898,
            'POINT (-98.7744 29.7158)',
            {
                'xmin': pytest.approx(-98.7744),
                'ymin': pytest.approx(29.7158),
                'xmax': pytest.approx(-98.7744),
                'ymax': pytest.approx(29.7158),
            },
        )
    ]


def test_reports_are_the_located_conus_floods_of_every_file(csv_paths):
    with duckdb_connect() as con:
        ids = storm_events.reports(con, csv_paths).project('event_id').order('event_id').fetchall()

    assert [event_id for (event_id,) in ids] == KEPT


def test_reports_leave_text_and_amounts_that_were_not_entered_null(csv_paths):
    with duckdb_connect() as con:
        report = (
            storm_events.reports(con, csv_paths)
            .filter('event_id = 2')
            .project('type, state, county, location, cause, damage')
            .fetchall()
        )

    assert report == [('Flood', 'Maryland', "Prince George's", None, None, None)]


@pytest.mark.parametrize(
    ('amount', 'dollars'),
    [
        ('10.00K', 10_000),
        ('2.5M', 2_500_000),
        ('1.2b', 1_200_000_000),
        ('3H', 300),
        ('500', 500),
        ('0.00K', 0),
        ('K', 0),
        ('.5K', 500),
        ('5.', 5),
        ('', None),
        (None, None),
        ('about 5K', None),
        ('.', None),
        ('.K', None),
    ],
)
def test_dollars_sql_scales_an_amount_by_its_suffix(amount, dollars):
    with duckdb_connect() as con:
        got = con.execute(
            f'SELECT {storm_events.dollars_sql("amount")} FROM (SELECT $a::VARCHAR AS amount)', {'a': amount}
        ).fetchone()[0]  # fmt: skip

    assert got == dollars


@pytest.mark.parametrize(
    ('property_amount', 'crop_amount', 'damage'),
    [('"1K"', '"0.50K"', 1_500), ('"1K"', '', 1_000), ('', '"2M"', 2_000_000), ('', '', None)],
)
def test_damage_adds_the_amounts_that_were_entered(tmp_path, property_amount, crop_amount, damage):
    path = write_details(
        tmp_path / details_name(2019, 20260323),
        [row(DAMAGE_PROPERTY=property_amount, DAMAGE_CROPS=crop_amount)],
    )
    with duckdb_connect() as con:
        assert storm_events.reports(con, [path]).project('damage').fetchall() == [(damage,)]


@pytest.mark.parametrize(
    ('name', 'titled'),
    [
        ("PRINCE GEORGE'S", "Prince George's"),
        ("O'BRIEN", "O'Brien"),
        ('ST. PETERSBURG', 'St. Petersburg'),
        ('WINSTON-SALEM', 'Winston-Salem'),
        ('CAÑON CITY', 'Cañon City'),
        (None, None),
    ],
)
def test_title_sql_capitalises_words_and_keeps_a_possessive_s_lower(name, titled):
    with duckdb_connect() as con:
        got = con.execute(
            f'SELECT {storm_events.title_sql("name")} FROM (SELECT $n::VARCHAR AS name)', {'n': name}
        ).fetchone()[0]  # fmt: skip

    assert got == titled


@pytest.mark.parametrize(
    ('changes', 'error', 'message'),
    [
        (
            {'END_YEARMONTH': '201902', 'END_DAY': '30'},
            duckdb.ConversionException,
            'Date out of range: 2019-2-30',
        ),
        ({'BEGIN_DAY': '100'}, duckdb.InvalidInputException, 'Could not parse string "201905100"'),
    ],
)
def test_reports_raise_on_a_day_that_is_no_date(tmp_path, changes, error, message):
    path = write_details(tmp_path / details_name(2019, 20260323), [row(**changes)])
    with duckdb_connect() as con:
        with pytest.raises(error, match=message):
            storm_events.reports(con, [path]).fetchall()


def test_write_local_builds_geoparquet_and_overviews_with_provenance(csv_paths, tmp_path):
    files = {path.name: f'sha-{i}' for i, path in enumerate(csv_paths)}
    meta = storm_events.source_metadata(files, retrieved=date(2026, 11, 16))

    out = storm_events.write_local(csv_paths, tmp_path / 'out', metadata=meta)

    with duckdb_connect() as con:
        ids = con.execute(
            'SELECT list(event_id ORDER BY event_id) FROM read_parquet($p)', {'p': str(out)}
        ).fetchone()[0]
        dates = con.execute(
            'SELECT begin_date, end_date FROM read_parquet($p) WHERE event_id = 824116',
            {'p': str(out)},
        ).fetchone()
        kv = dict(
            con.execute(
                'SELECT decode(key), decode(value) FROM parquet_kv_metadata($p)', {'p': str(out)}
            ).fetchall()
        )
    lod_meta = json.loads(kv['geo'])[geoparquet_lod.LOD_KEY]
    geoparquet_lod.validate(str(out))
    assert out.name == 'storm_events_floods.parquet'
    assert ids == KEPT
    assert dates == ('2019-05-09', '2019-05-10')
    assert {k: v for k, v in kv.items() if k.startswith('storm_events_')} == {
        'storm_events_index': 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/',
        'storm_events_files': json.dumps(files),
        'storm_events_raw_mirror': 'input/historic/storm_events/raw/',
        'storm_events_retrieved': '2026-11-16',
        'storm_events_attribution': 'NOAA NCEI Storm Events Database',
        'storm_events_license': 'Public domain (US Government work)',
        'storm_events_ocr_version': storm_events.ocr_version(),
    }
    assert [a['name'] for a in lod_meta['attributes']] == ['year', 'type']
    assert lod_meta['zooms'][-1] == 10


def test_write_local_raises_when_two_files_hold_one_report(tmp_path):
    paths = [
        write_details(tmp_path / details_name(2019, stamp), [ROWS['flash_flood']])
        for stamp in (20260323, 20260918)
    ]

    with pytest.raises(ValueError, match='1 reports repeat an event_id'):
        storm_events.write_local(paths, tmp_path / 'out')


def test_read_mirror_copies_a_mirrored_file_and_digests_it(tmp_path, monkeypatch):
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path / 'store'))
    name = details_name(2019, 20260323)
    mirrored = tmp_path / 'store' / storm_events.raw_key(name)
    mirrored.parent.mkdir(parents=True)
    mirrored.write_bytes(b'gzipped reports')

    sha256 = storm_events.read_mirror(name, tmp_path / name)

    assert (tmp_path / name).read_bytes() == b'gzipped reports'
    assert sha256 == hashlib.sha256(b'gzipped reports').hexdigest()


OLD, NEW = details_name(2019, 20260323), details_name(2020, 20260918)
OUT = 'input/historic/storm_events/20260918/storm_events_floods.parquet'
RAW_OLD, RAW_NEW = storm_events.raw_key(OLD), storm_events.raw_key(NEW)
BUILD = 'build'


@pytest.mark.parametrize(
    ('case', 'stored', 'overwrite', 'build_fails', 'events', 'message'),
    [
        ('absent', set(), False, False,
         [('download', OLD), ('download', NEW), BUILD, RAW_OLD, RAW_NEW, OUT], 'wrote'),
        ('output_present', {OUT}, False, False, [], 'skip'),
        ('one_file_mirrored', {RAW_OLD}, False, False,
         [('read mirror', OLD), ('download', NEW), BUILD, RAW_NEW, OUT], 'wrote'),
        ('overwrite_reads_the_mirrors', {OUT, RAW_OLD, RAW_NEW}, True, False,
         [('read mirror', OLD), ('read mirror', NEW), BUILD, OUT], 'wrote'),
        ('build_failure_uploads_nothing', set(), False, True,
         [('download', OLD), ('download', NEW), BUILD], 'boom'),
    ],
)  # fmt: skip
def test_main_builds_from_each_years_newest_file_and_mirrors_after_the_build(
    monkeypatch, capsys, case, stored, overwrite, build_fails, events, message
):
    log = []

    def fetch(how):
        def copy(name, dest):
            log.append((how, name))
            dest.write_bytes(b'x')
            return f'sha of {name}'

        return copy

    def head(store, key):
        if key not in stored:
            raise FileNotFoundError(key)
        return {'size': 1}

    def write_local(csv_paths, out_dir, *, metadata, temp_dir):
        log.append(BUILD)
        if build_fails:
            raise RuntimeError('boom')
        assert [path.name for path in csv_paths] == [OLD, NEW], case
        assert json.loads(metadata['storm_events_files']) == {
            OLD: f'sha of {OLD}',
            NEW: f'sha of {NEW}',
        }, case
        return Path(out_dir) / 'storm_events_floods.parquet'

    index = '\n'.join([OLD, details_name(2020, 20260323), NEW])
    monkeypatch.setattr(storm_events, 'fetch_index', lambda: index)
    monkeypatch.setattr(storm_events, 'download', fetch('download'))
    monkeypatch.setattr(storm_events, 'read_mirror', fetch('read mirror'))
    monkeypatch.setattr(storm_events, 'write_local', write_local)
    monkeypatch.setattr(storm_events.obstore, 'head', head)
    monkeypatch.setattr(storm_events.obstore, 'put', lambda store, key, path: log.append(key))
    monkeypatch.setattr(
        storm_events, 'StorageConfig', lambda: SimpleNamespace(object_store=lambda: None)
    )

    if build_fails:
        with pytest.raises(RuntimeError, match=message):
            storm_events.main(overwrite=overwrite)
    else:
        storm_events.main(overwrite=overwrite)
        assert message in capsys.readouterr().out, case
    assert log == events, case


def test_main_raises_when_the_listing_names_no_details_file(monkeypatch):
    monkeypatch.setattr(storm_events, 'fetch_index', lambda: '<html>moved</html>')

    with pytest.raises(ValueError, match='lists no details files'):
        storm_events.main(overwrite=False)
