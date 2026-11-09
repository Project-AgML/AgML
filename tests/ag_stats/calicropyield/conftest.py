import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("xarray")

import rasterio
import xarray as xr
from rasterio.transform import from_origin

from .calicropyield_fixtures import CDL, LANDSAT_BANDS, RES

# A 4x6 pixel grid at the CDL resolution, in the corner of Yolo County.
_TRANSFORM = from_origin(-122.0, 38.6, RES, RES)
_HEIGHT, _WIDTH = CDL.shape


def _write_tif(path, data, nodata=None, descriptions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=_HEIGHT, width=_WIDTH, count=data.shape[0],
        dtype=data.dtype, crs="EPSG:4326", transform=_TRANSFORM, nodata=nodata,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(descriptions or []):
            dst.set_band_description(i + 1, name)


def _grid_coords(n_y, n_x, step):
    return {"y": 38.6 - step * (np.arange(n_y) + 0.5), "x": -122.0 + step * (np.arange(n_x) + 0.5)}


@pytest.fixture(scope="session")
def cali_root(tmp_path_factory):
    """A directory laid out like the Hub repo, with Yolo 2020 and 2021 data."""
    root = tmp_path_factory.mktemp("calicropyield")
    county = root / "counties" / "Yolo"

    for year in (2020, 2021):
        _write_tif(county / "cdl" / str(year) / f"Yolo_{year}.tif", CDL, nodata=0)
        _write_tif(county / "irrigation" / str(year) / f"Yolo_{year}.tif", (CDL > 0).astype("uint8"))
        for month in range(1, 13):
            landsat = np.full((6, _HEIGHT, _WIDTH), month, dtype="float32") + np.arange(6, dtype="float32")[:, None, None] / 10
            _write_tif(county / "landsat" / str(year) / f"Yolo_LT_{year}_{month:02d}.tif", landsat, descriptions=LANDSAT_BANDS)
            et = np.full((_HEIGHT, _WIDTH), 10.0 * month, dtype="float32")
            _write_tif(county / "et" / str(year) / f"Yolo_ET_{year}_{month:02d}.tif", et, nodata=np.nan)

    try:
        import zarr  # noqa: F401
    except ImportError:
        pass
    else:
        for year in (2020, 2021):
            time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            coords = {"time": time, **_grid_coords(3, 4, 0.01)}
            values = np.broadcast_to(np.arange(len(time), dtype="float32")[:, None, None], (len(time), 3, 4))
            climate = xr.Dataset({v: (("time", "y", "x"), values.copy()) for v in ("prcp", "tmax", "tmin")}, coords=coords)
            climate.to_zarr(county / "climate" / str(year) / f"DayMet_Yolo_{year}.zarr", consolidated=False)
        soil = xr.Dataset(
            {
                "soc0_100": (("y", "x"), np.full((_HEIGHT, _WIDTH), 5000.0, dtype="float32")),
                "hydgrpdcd": (("y", "x"), np.ones((_HEIGHT, _WIDTH), dtype="float32"), {"categories": "1=A,2=B"}),
            },
            coords=_grid_coords(_HEIGHT, _WIDTH, RES),
        )
        # The real stores carry their CRS in a `spatial_ref` variable.
        soil["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": "EPSG:4326"})
        soil.to_zarr(county / "soil" / "2020" / "Yolo_soil.zarr", consolidated=False)

    (root / "yield" / "metadata").mkdir(parents=True)
    pd.DataFrame(
        {
            "state": "CA",
            "county": ["Yolo", "Yolo", "Yolo", "Contra Costa", "Fresno"],
            "year": [2019, 2020, 2020, 2020, 2021],
            "crop": ["TOMATOES", "TOMATOES", "ALMONDS", "TOMATOES", "HAY"],
            "group_desc": ["VEGETABLE CROPS", "VEGETABLE CROPS", "TREE NUTS & FRUITS", "VEGETABLE CROPS", "FIELD CROPS"],
            "yield_t_per_ha": [90.0, 95.0, 2.5, 80.0, 15.0],
            "yield_std": 0.0,
            "n_subtypes_merged": 1,
            "contains_flagged_outlier": False,
        }
    ).to_parquet(root / "yield" / "county_crop_yield.parquet")
    pd.DataFrame(
        {
            "Year": [2020, 2020],
            "County": ["Yolo", "Yolo"],
            "Crop_Name_Clean": ["Tomatoes Processing", "Almonds All"],
            "Yield_Tons_Per_Hectare": [95.0, 2.5],
        }
    ).to_parquet(root / "yield" / "county_crop_yield_detail.parquet")
    pd.DataFrame(
        {
            "yield_crop": ["TOMATOES", "HAY", "ANISE"],
            "cdl_codes": ["54", "36;37", None],
            "cdl_names": ["Tomatoes", "Alfalfa;Other Hay/Non Alfalfa", None],
            "confidence": ["high", "manual", "unmatched"],
            "note": [None, None, "no CDL class"],
        }
    ).to_csv(root / "yield" / "metadata" / "cdl_crop_crosswalk.csv", index=False)

    (root / "reference").mkdir()
    square = [[-122.0, 38.6], [-121.9, 38.6], [-121.9, 38.5], [-122.0, 38.5], [-122.0, 38.6]]
    feature = {"type": "Feature", "properties": {"NAME": "Yolo County"}, "geometry": {"type": "Polygon", "coordinates": [square]}}
    (root / "reference" / "California_Counties.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return root
