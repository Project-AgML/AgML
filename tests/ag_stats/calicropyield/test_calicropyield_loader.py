import os
import pickle
import shutil

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("xarray")

from shapely.geometry import box

from agml.ag_stats.calicropyield import DEFAULT_REPO_ID, CaliCropYieldLoader

from .calicropyield_fixtures import CDL, LANDSAT_BANDS, RES

try:
    import zarr  # noqa: F401
except ImportError:
    zarr = None
needs_zarr = pytest.mark.skipif(zarr is None, reason="zarr is not installed")


@pytest.fixture(scope="module")
def cali(cali_root):
    return CaliCropYieldLoader(str(cali_root))


# ── Initialisation ────────────────────────────────────────────────────────────

class TestInit:
    def test_reads_env_var(self, cali_root, monkeypatch):
        monkeypatch.setenv("AGML_CALICROPYIELD_PATH", str(cali_root))
        assert CaliCropYieldLoader().root == os.path.realpath(cali_root)

    def test_defaults_to_hub(self, monkeypatch):
        monkeypatch.delenv("AGML_CALICROPYIELD_PATH", raising=False)
        loader = CaliCropYieldLoader()
        assert loader.repo_id == DEFAULT_REPO_ID and loader.root is None

    def test_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CaliCropYieldLoader(str(tmp_path / "nope"))

    def test_pickle_round_trip(self, cali):
        clone = pickle.loads(pickle.dumps(cali))
        assert clone.root == cali.root
        assert clone.load_cdl("Yolo", 2020).shape == CDL.shape

    def test_listings(self):
        assert len(CaliCropYieldLoader.counties()) == 58
        assert "landsat" in CaliCropYieldLoader.layers()
        assert CaliCropYieldLoader.cdl_classes()[54] == "Tomatoes"


# ── Rasters ───────────────────────────────────────────────────────────────────

class TestRasters:
    def test_cdl(self, cali):
        cdl = cali.load_cdl("yolo", 2020)
        assert cdl.dims == ("y", "x")
        np.testing.assert_array_equal(cdl.values, CDL)
        assert cdl.attrs["crs"] == "EPSG:4326"

    def test_cdl_keeps_only_crops(self, cali):
        cdl = cali.load_cdl("Yolo", 2020, crops=["Tomatoes", "Almonds"])
        assert set(np.unique(cdl.values)) == {0, 54, 75}
        assert cdl.dtype == CDL.dtype

    def test_coordinates_are_pixel_centers(self, cali):
        cdl = cali.load_cdl("Yolo", 2020)
        assert cdl["x"].values[0] == pytest.approx(-122.0 + RES / 2)
        assert cdl["y"].values[0] == pytest.approx(38.6 - RES / 2)

    def test_crop_mask_default_is_cropland(self, cali):
        mask = cali.crop_mask("Yolo", 2020)
        assert mask.dtype == bool
        assert mask.values.sum() == ((CDL == 54) | (CDL == 75) | (CDL == 36)).sum()

    def test_irrigation(self, cali):
        assert cali.load_irrigation("Yolo", 2021).dims == ("y", "x")

    def test_landsat(self, cali):
        landsat = cali.load_landsat("Yolo", 2020, months=[6, 7])
        assert landsat.dims == ("month", "band", "y", "x")
        assert list(landsat["month"].values) == [6, 7]
        assert list(landsat["band"].values) == list(LANDSAT_BANDS)
        assert float(landsat.sel(month=7, band="SR_B1").mean()) == pytest.approx(7.0)

    def test_et(self, cali):
        et = cali.load_et("Yolo", 2020)
        assert et.dims == ("month", "y", "x") and et.sizes["month"] == 12
        assert float(et.sel(month=3).mean()) == pytest.approx(30.0)

    def test_geometry_crops(self, cali):
        # Kept inside the pixel edges; a box exactly on an edge may include the next row.
        cdl = cali.load_cdl("Yolo", 2020, geometry=box(-122.0 + 0.1 * RES, 38.6 - 1.9 * RES, -122.0 + 1.9 * RES, 38.6 - 0.1 * RES))
        assert cdl.shape == (2, 2)
        assert (cdl.values == 54).all()

    def test_geometry_fills_floats_with_nan(self, cali):
        triangle = {"type": "Polygon", "coordinates": [[[-122.0, 38.6], [-122.0 + 4 * RES, 38.6], [-122.0, 38.6 - 4 * RES], [-122.0, 38.6]]]}
        et = cali.load_et("Yolo", 2020, months=[1], geometry=triangle)
        assert np.isnan(et.values).any() and not np.isnan(et.values).all()

    def test_bad_month(self, cali):
        with pytest.raises(ValueError, match="Months"):
            cali.load_landsat("Yolo", 2020, months=[13])

    def test_year_not_in_layer(self, cali):
        with pytest.raises(ValueError, match="2012"):
            cali.load_cdl("Yolo", 2012)

    def test_file_missing_from_copy(self, cali):
        with pytest.raises(FileNotFoundError):
            cali.load_cdl("Fresno", 2020)


# ── Zarr layers ───────────────────────────────────────────────────────────────

@needs_zarr
class TestZarr:
    def test_climate_one_year(self, cali):
        ds = cali.load_climate("Yolo", years=2020)
        assert set(ds.data_vars) == {"prcp", "tmax", "tmin"}
        assert ds.sizes["time"] == 366

    def test_climate_years_are_joined(self, cali):
        ds = cali.load_climate("Yolo", years=[2020, 2021], variables="prcp")
        assert list(ds.data_vars) == ["prcp"]
        assert ds.sizes["time"] == 366 + 365

    def test_climate_monthly(self, cali):
        ds = cali.load_climate("Yolo", years=2020, monthly=True)
        assert ds.sizes["time"] == 12
        assert float(ds["prcp"].isel(time=0).mean()) == pytest.approx(15.0)

    def test_climate_geometry_bbox(self, cali):
        ds = cali.load_climate("Yolo", years=2020, geometry=box(-122.0, 38.58, -121.98, 38.6))
        assert ds.sizes["y"] == 2 and ds.sizes["x"] == 2

    def test_soil(self, cali):
        ds = cali.load_soil("Yolo", variables=["hydgrpdcd"])
        assert list(ds.data_vars) == ["hydgrpdcd"]
        assert ds["hydgrpdcd"].attrs["categories"] == "1=A,2=B"

    def test_crs_variable_is_a_coordinate(self, cali):
        ds = cali.load_soil("Yolo")
        assert set(ds.data_vars) == {"soc0_100", "hydgrpdcd"}
        assert "spatial_ref" in ds.coords

    def test_unknown_variable(self, cali):
        with pytest.raises(ValueError, match="soc0_100"):
            cali.load_soil("Yolo", variables="carbon")


# ── Tables ────────────────────────────────────────────────────────────────────

class TestTables:
    def test_yield_filters(self, cali):
        df = cali.load_yield(county="yolo", crop="tomatoes")
        assert list(df["year"]) == [2019, 2020]

    def test_yield_county_with_space(self, cali):
        df = cali.load_yield(county="ContraCosta")
        assert list(df["county"]) == ["Contra Costa"]

    def test_yield_year_range(self, cali):
        df = cali.load_yield(years=(2020, 2021))
        assert set(df["year"]) == {2020, 2021}

    def test_yield_detail(self, cali):
        df = cali.load_yield(crop="tomatoes processing", detail=True)
        assert list(df["Yield_Tons_Per_Hectare"]) == [95.0]

    def test_yield_crops(self, cali):
        assert cali.yield_crops() == ["ALMONDS", "HAY", "TOMATOES"]

    def test_crosswalk_codes(self, cali):
        assert cali.yield_crop_cdl_codes("hay") == [36, 37]
        assert cali.yield_crop_cdl_codes("ANISE") == []
        with pytest.raises(ValueError, match="yield crop"):
            cali.yield_crop_cdl_codes("kiwi")

    def test_crosswalk_feeds_crop_mask(self, cali):
        mask = cali.crop_mask("Yolo", 2020, crops=cali.yield_crop_cdl_codes("HAY"))
        assert mask.values.sum() == (CDL == 36).sum()

    def test_county_boundaries(self, cali):
        pytest.importorskip("geopandas")
        assert list(cali.load_boundaries()["NAME"]) == ["Yolo County"]


# ── Downloads ─────────────────────────────────────────────────────────────────

class TestDownload:
    def test_monthly_paths(self, cali):
        paths = cali.download("et", "Yolo", years=2020, months=[1, 2])
        assert [os.path.basename(p) for p in paths] == ["Yolo_ET_2020_01.tif", "Yolo_ET_2020_02.tif"]

    def test_layer_methods(self, cali):
        assert len(cali.download_landsat("Yolo", years=2020, months=[7])) == 1
        assert len(cali.download_irrigation("Yolo", years=(2020, 2021))) == 2
        assert [os.path.basename(p) for p in cali.download_yield()] == [
            "county_crop_yield.parquet", "county_crop_yield_detail.parquet", "cdl_crop_crosswalk.csv",
        ]

    @needs_zarr
    def test_zarr_layer_methods(self, cali):
        assert os.path.isdir(cali.download_soil("Yolo")[0])
        assert len(cali.download_climate("Yolo", years=[2020, 2021])) == 2

    def test_missing_files_are_skipped(self, cali):
        paths = cali.download("cdl", ["Yolo", "Fresno"], years=(2020, 2021))
        assert len(paths) == 2


# ── Hugging Face Hub ──────────────────────────────────────────────────────────

class FakeHub:
    """Stands in for `huggingface_hub`, serving the fixture dataset from a folder."""

    def __init__(self, cali_root, tmp_path):
        self.folder = tmp_path / "snapshot"
        shutil.copytree(cali_root, self.folder)
        self.downloads = []

    def hf_hub_download(self, repo_id, repo_type, filename, revision, cache_dir, token):
        from huggingface_hub.errors import EntryNotFoundError

        assert repo_id == DEFAULT_REPO_ID and repo_type == "dataset"
        if not (self.folder / filename).is_file():
            raise EntryNotFoundError(filename)
        self.downloads.append(filename)
        return str(self.folder / filename)

    def snapshot_download(self, repo_id, repo_type, revision, cache_dir, token, allow_patterns):
        self.downloads.extend(allow_patterns)
        return str(self.folder)


@pytest.fixture
def hub(cali_root, tmp_path, monkeypatch):
    import huggingface_hub

    fake = FakeHub(cali_root, tmp_path)
    monkeypatch.delenv("AGML_CALICROPYIELD_PATH", raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    return fake


class TestHub:
    def test_downloads_only_what_is_read(self, hub):
        CaliCropYieldLoader().load_et("Yolo", 2020, months=[7])
        assert hub.downloads == ["counties/Yolo/et/2020/Yolo_ET_2020_07.tif"]

    def test_yield(self, hub):
        assert len(CaliCropYieldLoader().load_yield(county="Yolo")) == 3
        assert hub.downloads == ["yield/county_crop_yield.parquet"]

    @needs_zarr
    def test_zarr_store(self, hub):
        ds = CaliCropYieldLoader().load_soil("Yolo")
        assert "soc0_100" in ds
        assert hub.downloads == ["counties/Yolo/soil/2020/Yolo_soil.zarr/*"]

    def test_missing_file(self, hub):
        with pytest.raises(FileNotFoundError, match="not in the dataset"):
            CaliCropYieldLoader().load_cdl("Fresno", 2020)

    def test_pickle_keeps_repo(self, hub):
        clone = pickle.loads(pickle.dumps(CaliCropYieldLoader(revision="abc123")))
        assert clone.repo_id == DEFAULT_REPO_ID and clone.revision == "abc123"
