import pytest

pytest.importorskip("pyarrow")

from agml.ag_stats.nass import NASSCatalog

from .nass_fixtures import CORN_GRAIN, CORN_SILAGE, SOYBEANS


@pytest.fixture(scope="module")
def catalog(nass_root):
    return NASSCatalog(str(nass_root / "parquet"))


# ── Listings ──────────────────────────────────────────────────────────────────

class TestListings:
    def test_commodities(self, catalog):
        assert catalog.commodities() == ["CORN", "SOYBEANS"]

    def test_crops_for_one_commodity(self, catalog):
        assert set(catalog.crops("corn")["crop_id"]) == {CORN_GRAIN, CORN_SILAGE}

    def test_counties_for_one_state(self, catalog):
        assert list(catalog.counties("IA")["county_fips"]) == ["19169", "19183"]

    def test_districts_for_one_state(self, catalog):
        assert list(catalog.districts("Illinois")["asd_id"]) == ["1760"]

    def test_missing_dimension_table(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NASSCatalog(str(tmp_path))


# ── States ────────────────────────────────────────────────────────────────────

class TestStates:
    @pytest.mark.parametrize("value", ["IA", "ia", "Iowa", "IOWA", "19", 19])
    def test_accepted_forms(self, catalog, value):
        assert catalog.resolve_states(value) == ["19"]

    def test_list(self, catalog):
        assert catalog.resolve_states(["IA", "Illinois"]) == ["19", "17"]

    def test_unknown_suggests(self, catalog):
        with pytest.raises(ValueError, match="IOWA") as error:
            catalog.resolve_states("Iowaa")
        assert str(error.value).count("IOWA") == 1


# ── Counties ──────────────────────────────────────────────────────────────────

class TestCounties:
    def test_by_fips(self, catalog):
        assert catalog.resolve_counties("19169") == ["19169"]
        assert catalog.resolve_counties(19169) == ["19169"]

    def test_by_name(self, catalog):
        assert catalog.resolve_counties("Story") == ["19169"]
        assert catalog.resolve_counties("story county") == ["19169"]

    def test_ambiguous_name_needs_state(self, catalog):
        with pytest.raises(ValueError, match="several states"):
            catalog.resolve_counties("Washington")
        assert catalog.resolve_counties("Washington", state="IL") == ["17189"]

    def test_old_nass_code_is_remapped(self, catalog):
        assert catalog.resolve_counties("19999") == ["19169"]

    def test_unknown_fips(self, catalog):
        with pytest.raises(ValueError, match="Unknown county FIPS"):
            catalog.resolve_counties("99999")

    def test_unknown_name(self, catalog):
        with pytest.raises(ValueError, match="Unknown county"):
            catalog.resolve_counties("Atlantis", state="IA")


# ── Districts ─────────────────────────────────────────────────────────────────

class TestDistricts:
    def test_by_asd_id(self, catalog):
        assert catalog.resolve_districts("1950") == ["1950"]

    def test_by_code_with_state(self, catalog):
        assert catalog.resolve_districts("50", state="IA") == ["1950"]

    def test_code_without_state(self, catalog):
        with pytest.raises(ValueError, match="Pass `state`"):
            catalog.resolve_districts("50")

    def test_by_name(self, catalog):
        assert catalog.resolve_districts("southeast", state="IA") == ["1990"]


# ── Crops ─────────────────────────────────────────────────────────────────────

class TestCrops:
    def test_commodity_gives_every_variant(self, catalog):
        assert set(catalog.resolve_crops("corn")) == {CORN_GRAIN, CORN_SILAGE}

    def test_label(self, catalog):
        assert catalog.resolve_crops("Corn, Grain") == [CORN_GRAIN]

    def test_crop_id(self, catalog):
        assert catalog.resolve_crops(SOYBEANS) == [SOYBEANS]

    def test_narrow_by_utilization(self, catalog):
        assert catalog.resolve_crops("corn", utilization="silage") == [CORN_SILAGE]

    def test_several_crops(self, catalog):
        assert set(catalog.resolve_crops(["corn", "soybeans"], utilization=["grain", "all utilization practices"])) == {
            CORN_GRAIN,
            SOYBEANS,
        }

    def test_unknown_crop(self, catalog):
        with pytest.raises(ValueError, match="Unknown crop"):
            catalog.resolve_crops("unobtainium")

    def test_unknown_utilization(self, catalog):
        with pytest.raises(ValueError, match="Unknown utilization"):
            catalog.resolve_crops("corn", utilization="fiber")
