import os
import pickle
import shutil

import pytest

pytest.importorskip("pyarrow")

import pandas as pd
import pyarrow as pa

from agml.ag_stats import DEFAULT_REPO_ID, NASSDataLoader

from nass_fixtures import CORN_GRAIN, CORN_SILAGE


@pytest.fixture(scope="module")
def nass(nass_root):
    return NASSDataLoader(str(nass_root))


# ── Initialisation ────────────────────────────────────────────────────────────

class TestInit:
    def test_accepts_parent_or_parquet_dir(self, nass_root):
        a = NASSDataLoader(str(nass_root))
        b = NASSDataLoader(str(nass_root / "parquet"))
        assert a.root == b.root

    def test_reads_env_var(self, nass_root, monkeypatch):
        monkeypatch.setenv("AGML_NASS_PATH", str(nass_root))
        assert NASSDataLoader().root.endswith("parquet")

    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NASSDataLoader(str(tmp_path / "nope"))

    def test_pickle_round_trip(self, nass):
        clone = pickle.loads(pickle.dumps(nass))
        assert clone.root == nass.root
        assert len(clone.yield_data(state="IA", years=2020)) > 0

    def test_tables_listing(self):
        assert "yield" in NASSDataLoader.tables()


# ── Filters ───────────────────────────────────────────────────────────────────

class TestFilters:
    def test_one_county(self, nass):
        df = nass.yield_data(crop="corn", utilization="grain", county="Story", state="IA")
        assert set(df["county_fips"]) == {"19169"}
        assert set(df["crop_id"]) == {CORN_GRAIN}
        assert sorted(df["year"]) == [2018, 2019, 2020, 2021]

    def test_year_forms(self, nass):
        def years(spec):
            return sorted(nass.yield_data(county="19169", crop="corn, grain", years=spec)["year"])

        assert years(2020) == [2020]
        assert years((2019, 2020)) == [2019, 2020]
        assert years([2018, 2021]) == [2018, 2021]
        assert years(range(2020, 2022)) == [2020, 2021]

    def test_level(self, nass):
        df = nass.yield_data(crop="corn", utilization="grain", level="state")
        assert set(df["agg_level"]) == {"STATE"}
        assert set(df["state_fips"]) == {"19", "17"}

    def test_state_without_level_keeps_all_levels(self, nass):
        df = nass.yield_data(crop="corn", utilization="grain", state="IA", years=2020)
        assert set(df["agg_level"]) == {"COUNTY", "AGRICULTURAL DISTRICT", "STATE"}

    def test_other_counties_dropped_by_default(self, nass):
        kept = nass.yield_data(level="county", state="IA", crop=CORN_GRAIN, years=2020)
        assert kept["county_fips"].notna().all()
        every = nass.yield_data(level="county", state="IA", crop=CORN_GRAIN, years=2020, include_other_counties=True)
        assert len(every) == len(kept) + 1

    def test_district(self, nass):
        df = nass.yield_data(district="central", state="IA", level="district", crop=CORN_GRAIN)
        assert set(df["asd_code"]) == {"50"}

    def test_where_and_wrapper(self, nass):
        df = nass.area_production(statistic="production", crop="soybeans", state="IL", level="state")
        assert set(df["statistic"]) == {"production"}
        assert len(df) == 4

    @pytest.mark.parametrize("stage", ["planted", "PLANTED", "PCT PLANTED"])
    def test_progress_stage(self, nass, stage):
        df = nass.progress(stage=stage, state="IA", years=2021)
        assert set(df["stage"]) == {"PLANTED"}

    def test_forecast_months_in_calendar_order(self, nass):
        df = nass.forecast(state="IA", crop=CORN_GRAIN, years=2020)
        assert list(df["forecast_month"]) == ["AUG", "SEP", "OCT", "NOV"]
        assert "_month" not in df.columns

    def test_results_are_sorted(self, nass):
        df = nass.yield_data(crop=CORN_GRAIN, level="county")
        keys = list(zip(df["state_fips"], df["county_fips"], df["year"]))
        assert keys == sorted(keys)

    def test_condition_uses_crop_year(self, nass):
        df = nass.condition(state="IA", years=2019)
        assert set(df["crop_year"]) == {2019}

    def test_county_yield_filters_by_state_alpha(self, nass):
        df = nass.county_yield(state="Iowa", crop=CORN_SILAGE)
        assert set(df["state_alpha"]) == {"IA"}

    def test_columns(self, nass):
        df = nass.yield_data(county="19169", crop=CORN_GRAIN, columns=["year", "value"])
        assert list(df.columns) == ["year", "value"]


# ── Errors ────────────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_table(self, nass):
        with pytest.raises(ValueError, match="Unknown table"):
            nass.query("weather")

    def test_county_on_state_only_table(self, nass):
        with pytest.raises(ValueError, match="county-level"):
            nass.progress(county="19169")

    def test_level_on_county_only_table(self, nass):
        with pytest.raises(ValueError, match="state-level"):
            nass.county_yield(level="state")

    def test_unknown_where_column(self, nass):
        with pytest.raises(ValueError, match="no column 'colour'"):
            nass.yield_data(where={"colour": "red"})

    def test_unknown_columns(self, nass):
        with pytest.raises(ValueError, match="no column"):
            nass.yield_data(columns=["colour"])

    def test_unknown_format(self, nass):
        with pytest.raises(ValueError, match="Unknown format"):
            nass.yield_data(format="xlsx")

    def test_file_format_needs_path(self, nass):
        with pytest.raises(ValueError, match="needs an output"):
            nass.yield_data(format="csv")


# ── Boundaries ────────────────────────────────────────────────────────────────

class TestBoundaries:
    def test_county_with_geometry(self, nass):
        df = nass.boundaries("county")
        assert list(df["GEOID"]) == ["19169"] and "geometry" in df.columns

    def test_state_without_geometry(self, nass):
        df = nass.boundaries("state", geometry=False)
        assert list(df["GEOID"]) == ["19"] and "geometry" not in df.columns

    def test_unknown_level(self, nass):
        with pytest.raises(ValueError, match="boundary level"):
            nass.boundaries("district")


# ── Hugging Face Hub ──────────────────────────────────────────────────────────

class FakeHub:
    """Stands in for `huggingface_hub`, serving the fixture dataset as one commit."""

    COMMIT = "0123456789abcdef0123456789abcdef01234567"

    def __init__(self, nass_root, tmp_path):
        self.folder = tmp_path / self.COMMIT
        shutil.copytree(nass_root, self.folder)
        self.downloads = []

    def snapshot_download(self, repo_id, repo_type, revision, cache_dir, token, allow_patterns):
        assert repo_type == "dataset" and all(p.startswith("parquet/") for p in allow_patterns)
        self.downloads.extend(allow_patterns)
        return str(self.folder)

    def hf_hub_download(self, repo_id, repo_type, filename, revision, cache_dir, token):
        assert revision == self.COMMIT
        self.downloads.append(filename)
        return str(self.folder / filename)


@pytest.fixture
def hub(nass_root, tmp_path, monkeypatch):
    import huggingface_hub

    fake = FakeHub(nass_root, tmp_path)
    monkeypatch.delenv("AGML_NASS_PATH", raising=False)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    return fake


class TestHub:
    def test_defaults_to_hub(self, hub):
        nass = NASSDataLoader()
        assert nass.repo_id == DEFAULT_REPO_ID
        assert nass.revision == FakeHub.COMMIT

    def test_downloads_only_queried_tables(self, hub):
        nass = NASSDataLoader()
        assert not any(d.endswith("yield.parquet") for d in hub.downloads)
        assert len(nass.yield_data(state="IA", years=2020)) > 0
        assert "parquet/yield.parquet" in hub.downloads
        assert "parquet/area_production.parquet" not in hub.downloads

    def test_boundaries(self, hub):
        assert list(NASSDataLoader().boundaries("county")["GEOID"]) == ["19169"]
        assert "boundaries/us_county_2020.parquet" in hub.downloads

    def test_pickle_keeps_commit(self, hub):
        clone = pickle.loads(pickle.dumps(NASSDataLoader()))
        assert clone.revision == FakeHub.COMMIT
        assert len(clone.yield_data(state="IA", years=2020)) > 0


# ── Output formats ────────────────────────────────────────────────────────────

class TestFormats:
    QUERY = dict(county="19169", crop=CORN_GRAIN, years=2020)

    def test_pandas(self, nass):
        assert isinstance(nass.yield_data(**self.QUERY), pd.DataFrame)

    def test_arrow(self, nass):
        assert isinstance(nass.yield_data(format="arrow", **self.QUERY), pa.Table)

    def test_records(self, nass):
        records = nass.yield_data(format="records", **self.QUERY)
        assert records[0]["county_fips"] == "19169"

    def test_polars(self, nass):
        pl = pytest.importorskip("polars")
        assert isinstance(nass.yield_data(format="polars", **self.QUERY), pl.DataFrame)

    @pytest.mark.parametrize("fmt", ["csv", "parquet"])
    def test_files(self, nass, tmp_path, fmt):
        path = nass.yield_data(format=fmt, path=str(tmp_path / f"out.{fmt}"), **self.QUERY)
        assert os.path.getsize(path) > 0


# ── Real dataset (only when AGML_NASS_PATH points at it) ──────────────────────

@pytest.mark.skipif("AGML_NASS_PATH" not in os.environ, reason="AGML_NASS_PATH is not set")
def test_real_dataset_story_county_corn():
    nass = NASSDataLoader()
    df = nass.yield_data(crop="corn, grain", state="IA", county="Story", years=(2015, 2020), level="county")
    assert sorted(df["year"]) == list(range(2015, 2021))
    assert (df["value"] > 100).all()
