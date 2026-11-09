import pytest

pytest.importorskip("rasterio")
pytest.importorskip("xarray")

from agml.ag_stats.calicropyield import catalog


class TestCounties:
    @pytest.mark.parametrize("name", ["Contra Costa", "contra costa", "ContraCosta", "Contra Costa County"])
    def test_resolve(self, name):
        assert catalog.resolve_county(name) == "Contra Costa"
        assert catalog.county_slug(name) == "ContraCosta"

    def test_unknown_suggests(self):
        with pytest.raises(ValueError, match="Yolo"):
            catalog.resolve_county("Yollo")

    def test_count(self):
        assert len(catalog.COUNTIES) == 58


class TestYears:
    def test_all(self):
        years = catalog.resolve_years("landsat", None)
        assert years[0] == 2008 and years[-1] == 2023 and 2012 not in years

    def test_range_skips_2012(self):
        assert catalog.resolve_years("cdl", (2011, 2013)) == [2011, 2013]

    def test_layer_specific_years(self):
        assert catalog.resolve_years("cdl", 2024) == [2024]
        assert catalog.LAYERS["climate"].years[-1] == 2022

    @pytest.mark.parametrize("year", [2012, 2030])
    def test_missing_year(self, year):
        with pytest.raises(ValueError, match=str(year)):
            catalog.resolve_years("et", year)

    def test_unknown_layer(self):
        with pytest.raises(ValueError, match="Unknown layer"):
            catalog.resolve_years("ndvi", 2020)


class TestPaths:
    def test_monthly(self):
        assert catalog.LAYERS["landsat"].path("landsat", "Yolo", 2021, 7) == "counties/Yolo/landsat/2021/Yolo_LT_2021_07.tif"

    def test_zarr(self):
        assert catalog.LAYERS["climate"].path("climate", "Yolo", 2020) == "counties/Yolo/climate/2020/DayMet_Yolo_2020.zarr"


class TestCdlCodes:
    def test_names_and_codes(self):
        assert catalog.cdl_codes(["tomatoes", 75]) == [54, 75]

    def test_shared_name_returns_every_code(self):
        assert set(catalog.cdl_codes("Non-agricultural")) == {62, 63, 64, 65, *range(81, 92)}

    def test_default_is_cropland_only(self):
        codes = catalog.cdl_codes(None)
        assert 54 in codes and 121 not in codes and 141 not in codes and 0 not in codes

    def test_unknown(self):
        with pytest.raises(ValueError, match="Tomatoes"):
            catalog.cdl_codes("tomatos")
