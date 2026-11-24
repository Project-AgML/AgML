import pytest
from .nass_fixtures import CORN_GRAIN, CORN_SILAGE, SOYBEANS


# ── Fixture dataset ──────────────────────────────────────────────────────────
# A few rows per table, shaped like the real NASS parquet files.


_CROPS = {
    "crop_id": [CORN_GRAIN, CORN_SILAGE, SOYBEANS],
    "commodity": ["CORN", "CORN", "SOYBEANS"],
    "class": ["ALL CLASSES"] * 3,
    "practice": ["ALL PRODUCTION PRACTICES"] * 3,
    "utilization": ["GRAIN", "SILAGE", "ALL UTILIZATION PRACTICES"],
    "crop_label": ["CORN, GRAIN", "CORN, SILAGE", "SOYBEANS"],
}

_STATES = {
    "state_fips": ["19", "17", "53"],
    "state_alpha": ["IA", "IL", "WA"],
    "state_name": ["Iowa", "Illinois", "Washington"],
    "area_km2": [145746.0, 143793.0, 172119.0],
    "centroid_lon": [-93.5, -89.2, -120.4],
    "centroid_lat": [42.1, 40.0, 47.4],
}

_COUNTIES = {
    "county_fips": ["19169", "19183", "17183", "17189", "53071"],
    "state_fips": ["19", "19", "17", "17", "53"],
    "county_code": ["169", "183", "183", "189", "071"],
    "county_name": ["Story", "Washington", "Vermilion", "Washington", "Walla Walla"],
    "state_alpha": ["IA", "IA", "IL", "IL", "WA"],
    "area_km2": [1490.0, 1474.0, 2329.0, 1458.0, 3349.0],
    "centroid_lon": [-93.4, -91.7, -87.7, -89.4, -118.5],
    "centroid_lat": [42.0, 41.3, 40.2, 38.4, 46.2],
}

_DISTRICTS = {
    "state_fips": ["19", "19", "17"],
    "state_alpha": ["IA", "IA", "IL"],
    "asd_code": ["50", "90", "60"],
    "asd_desc": ["CENTRAL", "SOUTHEAST", "EAST"],
    "asd_id": ["1950", "1990", "1760"],
}

# An old NASS county code mapped to its 2020 code.
_REMAP = {"fips_nass": ["19999"], "fips_2020": ["19169"], "note": ["Old Story County code"]}


def _fact_rows(years):
    """One row per crop, place, and year, at county, district, state, and national level."""
    places = [
        ("COUNTY", "19", "IA", "50", "19169", False),
        ("COUNTY", "19", "IA", "90", "19183", False),
        ("COUNTY", "17", "IL", "60", "17183", False),
        ("COUNTY", "19", "IA", "90", None, True),
        ("AGRICULTURAL DISTRICT", "19", "IA", "50", None, False),
        ("STATE", "19", "IA", None, None, False),
        ("STATE", "17", "IL", None, None, False),
        ("NATIONAL", "99", "US", None, None, False),
    ]
    rows = []
    for crop_id, commodity in [(CORN_GRAIN, "CORN"), (CORN_SILAGE, "CORN"), (SOYBEANS, "SOYBEANS")]:
        for level, state_fips, state_alpha, asd, county, other in places:
            for year in years:
                rows.append(
                    {
                        "crop_id": crop_id,
                        "commodity": commodity,
                        "agg_level": level,
                        "state_fips": state_fips,
                        "state_alpha": state_alpha,
                        "asd_code": asd,
                        "county_fips": county,
                        "is_other_counties": other,
                        "year": year,
                        "value": float(year - 1900),
                    }
                )
    return rows


def _write(path, data):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(data) if isinstance(data, list) else pa.table(data)
    pq.write_table(table, path)


@pytest.fixture(scope="session")
def nass_root(tmp_path_factory):
    """A directory laid out like the real dataset: `<root>/parquet/*.parquet`."""
    # Skipped here, not at module level: a skip while loading a conftest aborts the whole run.
    pytest.importorskip("pyarrow")
    root = tmp_path_factory.mktemp("nass")
    parquet = root / "parquet"
    parquet.mkdir()

    _write(parquet / "dim_crop.parquet", _CROPS)
    _write(parquet / "dim_state.parquet", _STATES)
    _write(parquet / "dim_county.parquet", _COUNTIES)
    _write(parquet / "dim_asd.parquet", _DISTRICTS)
    _write(parquet / "county_fips_remap.parquet", _REMAP)

    years = [2018, 2019, 2020, 2021]
    _write(parquet / "yield.parquet", _fact_rows(years))
    _write(
        parquet / "area_production.parquet",
        [dict(r, statistic=s) for r in _fact_rows(years) for s in ("area_planted", "production")],
    )
    county_rows = [r for r in _fact_rows(years) if r["agg_level"] == "COUNTY" and r["county_fips"]]
    _write(
        parquet / "v_county_yield_main.parquet",
        [{k: r[k] for k in ("county_fips", "state_alpha", "year", "crop_id", "commodity", "value")} for r in county_rows],
    )
    state_rows = [r for r in _fact_rows(years) if r["agg_level"] in ("STATE", "NATIONAL")]
    _write(
        parquet / "crop_progress.parquet",
        [dict(r, crop_year=r["year"], stage=s) for r in state_rows for s in ("PLANTED", "HARVESTED")],
    )
    _write(
        parquet / "yield_forecast.parquet",
        [dict(r, forecast_month=m) for r in state_rows for m in ("NOV", "AUG", "OCT", "SEP")],
    )
    _write(
        parquet / "crop_condition.parquet",
        [{**{k: r[k] for k in ("crop_id", "agg_level", "state_fips", "state_alpha")}, "crop_year": r["year"], "good": 50.0} for r in state_rows],
    )

    boundaries = root / "boundaries"
    boundaries.mkdir()
    for level, geoid in (("county", "19169"), ("state", "19")):
        attributes = {"GEOID": [geoid], "NAME": ["Story" if level == "county" else "Iowa"]}
        _write(boundaries / f"us_{level}_2020_attributes.parquet", attributes)
        _write(boundaries / f"us_{level}_2020.parquet", dict(attributes, geometry=[b"\x01"]))
    return root
