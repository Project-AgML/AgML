"""Downloads and reads the California Crop Yield Benchmark."""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.mask
import xarray as xr

from agml.framework import AgMLSerializable
from agml.utils.logging import log
from agml.ag_stats.calicropyield import catalog
from agml.ag_stats.calicropyield.catalog import DEFAULT_REPO_ID

# The environment variable read when no `root` is passed to the loader.
_ROOT_ENV_VAR = "AGML_CALICROPYIELD_PATH"

StrOrList = Union[str, Iterable[str], None]
YearSpec = Union[None, int, Tuple[int, int], Iterable[int]]

# Files downloaded at the same time from the Hub.
_DOWNLOAD_WORKERS = 8


class DownloadError(RuntimeError):
    """Raised when files that are in the dataset fail to download.

    `failed` maps each repo path that failed to the exception it raised.
    Files that finished are kept in the cache, so running the same call
    again only downloads the rest.
    """

    def __init__(self, failed: Dict[str, Exception]):
        self.failed = failed
        first_path, first_error = next(iter(failed.items()))
        super().__init__(
            f"{len(failed)} file(s) failed to download, e.g. '{first_path}' "
            f"({type(first_error).__name__}: {first_error}). Files that finished are cached, "
            f"so running the same call again only downloads the rest. See `.failed` for every file."
        )


class CaliCropYieldLoader(AgMLSerializable):
    """Downloads and reads the California Crop Yield Benchmark.

    The benchmark covers the 58 California counties from 2008 to 2023
    (2012 is missing) with these layers:

    - `cdl`: USDA Cropland Data Layer, yearly, 30 m.
    - `landsat`: Landsat surface reflectance, monthly, 30 m, 6 bands.
    - `et`: OpenET evapotranspiration, monthly, 30 m.
    - `irrigation`: IrrMapper irrigated land, yearly, on the CDL grid.
    - `climate`: DayMet weather, daily, 1 km (Zarr).
    - `soil`: 17 gNATSGO/SSURGO soil attributes, 2020 vintage, 30 m (Zarr).

    and county crop yield (t/ha) for over 100 crops from the CDFA county
    agricultural commissioner reports.

    By default the files come from the Hugging Face Hub. Each file is
    downloaded the first time it is read and kept in the Hugging Face
    cache. All rasters are in EPSG:4326, and `geometry` arguments are
    shapely geometries or GeoJSON-like dicts in that CRS.

    Parameters
    ----------
    root : str, optional
        A local copy of the dataset, laid out like the Hub repo (e.g. made
        with `hf download --local-dir`). Defaults to the
        `AGML_CALICROPYIELD_PATH` environment variable. When neither is
        set, the dataset is read from `repo_id`.
    repo_id : str, optional
        The Hugging Face dataset repo to read from.
    revision : str, optional
        A branch, tag, or commit of `repo_id`. Defaults to the latest
        version. A branch or tag is resolved to its commit when the loader
        is created, so every file comes from the same version of the
        dataset. Pass a commit hash to make results reproducible.
    cache_dir : str, optional
        Where downloaded files are kept. Defaults to the Hugging Face cache.
    token : str, optional
        A Hugging Face token. Not needed for the public dataset.

    Examples
    --------
    >>> cali = CaliCropYieldLoader()
    >>> cali.load_yield(county="Yolo", crop="tomatoes", years=(2015, 2020))
    >>> cdl = cali.load_cdl("Yolo", 2020, crops="Tomatoes")
    >>> landsat = cali.load_landsat("Yolo", 2020, months=[6, 7, 8])
    >>> weather = cali.load_climate("Yolo", years=2020, monthly=True)
    """

    serializable = frozenset(("root", "repo_id", "revision", "cache_dir"))

    def __init__(
        self,
        root: Optional[str] = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        root = root or os.environ.get(_ROOT_ENV_VAR)
        if root is not None:
            root = os.path.realpath(os.path.expanduser(root))
            if not os.path.isdir(root):
                raise FileNotFoundError(f"The California Crop Yield Benchmark directory '{root}' does not exist.")
        self._root = root
        # `repo_id` and `revision` are only kept in Hub mode; `None` marks a local copy.
        self._repo_id = None if root is not None else repo_id
        self._cache_dir = cache_dir
        self._token = token
        self._revision = None if root is not None else self._resolve_commit(revision)

    def __repr__(self):
        if self._repo_id is not None:
            return f"<CaliCropYieldLoader: {self._repo_id}@{self._revision[:7]}>"
        return f"<CaliCropYieldLoader: {self._root}>"

    @property
    def root(self) -> Optional[str]:
        """The local copy of the dataset, or `None` when reading from the Hub."""
        return self._root

    @property
    def repo_id(self) -> Optional[str]:
        """The Hugging Face repo the data comes from, or `None` for a local copy."""
        return self._repo_id

    @property
    def revision(self) -> Optional[str]:
        """The commit of `repo_id` the data comes from, or `None` for a local copy."""
        return self._revision

    # Listings.

    @staticmethod
    def counties() -> List[str]:
        """Returns the names of the 58 California counties."""
        return list(catalog.COUNTIES)

    @staticmethod
    def layers() -> Dict[str, str]:
        """Returns the name and a one-line description of every gridded layer."""
        return {name: spec.description for name, spec in catalog.LAYERS.items()}

    @staticmethod
    def years(layer: str) -> List[int]:
        """Returns the years a layer covers."""
        return list(catalog.layer(layer).years)

    @staticmethod
    def cdl_classes() -> Dict[int, str]:
        """Returns the CDL class name of every code."""
        return dict(catalog.CDL_CLASSES)

    # Downloads.

    def download(
        self,
        layer: str,
        counties: StrOrList,
        years: YearSpec = None,
        months: Optional[Iterable[int]] = None,
    ) -> List[str]:
        """Downloads layer files without loading them, and returns their local paths.

        Use this to fetch data ahead of time, e.g. before going offline.
        Files are downloaded in parallel. Files that are not in the dataset
        are skipped with a warning. Files that are in the dataset but fail
        to download raise a `DownloadError` that lists them. Zarr layers
        return the path of each store (a directory).

        Parameters
        ----------
        layer : str
            One of the names in `CaliCropYieldLoader.layers()`.
        counties : str or list of str
            County names, e.g. `"Yolo"` or `["Fresno", "Contra Costa"]`.
        years : int, tuple, or list of int, optional
            A year, an inclusive `(start, end)` range, or a list of
            years. Defaults to every year of the layer.
        months : list of int, optional
            Months of the monthly layers (`landsat`, `et`). Defaults to all.
        """
        spec = catalog.layer(layer)
        months = _check_months(months) if spec.monthly else [None]
        targets = [
            spec.path(layer, catalog.county_slug(county), year, month)
            for county in _as_list(counties)
            for year in catalog.resolve_years(layer, years)
            for month in months
        ]
        if self._repo_id is None:
            found = {t: os.path.join(self._root, t) for t in targets if os.path.exists(os.path.join(self._root, t))}
            missing = [t for t in targets if t not in found]
        else:
            found, missing = self._fetch(targets)
        if missing:
            log(f"{len(missing)} file(s) are not in the dataset and were skipped: {missing[:5]}")
        return [found[t] for t in targets if t in found]

    def download_cdl(self, counties: StrOrList, years: YearSpec = None) -> List[str]:
        """Downloads Cropland Data Layer GeoTIFFs. See `download`."""
        return self.download("cdl", counties, years)

    def download_landsat(self, counties: StrOrList, years: YearSpec = None, months: Optional[Iterable[int]] = None) -> List[str]:
        """Downloads monthly Landsat GeoTIFFs. See `download`."""
        return self.download("landsat", counties, years, months)

    def download_et(self, counties: StrOrList, years: YearSpec = None, months: Optional[Iterable[int]] = None) -> List[str]:
        """Downloads monthly OpenET GeoTIFFs. See `download`."""
        return self.download("et", counties, years, months)

    def download_irrigation(self, counties: StrOrList, years: YearSpec = None) -> List[str]:
        """Downloads irrigated land GeoTIFFs. See `download`."""
        return self.download("irrigation", counties, years)

    def download_climate(self, counties: StrOrList, years: YearSpec = None) -> List[str]:
        """Downloads DayMet Zarr stores, one per county and year. See `download`."""
        return self.download("climate", counties, years)

    def download_soil(self, counties: StrOrList) -> List[str]:
        """Downloads soil Zarr stores, one per county. See `download`."""
        return self.download("soil", counties)

    def download_yield(self) -> List[str]:
        """Downloads the yield tables and the CDL crosswalk, and returns their local paths."""
        return [self._file(path) for path in (catalog.YIELD_TABLE, catalog.YIELD_DETAIL_TABLE, catalog.CDL_CROSSWALK)]

    # Loading.

    def load_cdl(self, county: str, year: int, crops=None, geometry=None) -> xr.DataArray:
        """Reads the Cropland Data Layer of a county as a `(y, x)` array of class codes.

        `crops` keeps only the given CDL classes, as names (`"Tomatoes"`)
        or codes (`54`), and sets every other pixel to 0. Look up names
        with `cdl_classes()`, or use `yield_crop_cdl_codes` to go from a
        yield crop to its CDL classes.
        """
        cdl = self._read_raster("cdl", county, year, geometry=geometry)[0]
        if crops is not None:
            cdl = cdl.where(cdl.isin(catalog.cdl_codes(crops)), 0).astype(cdl.dtype)
        return cdl.assign_attrs(classes=catalog.CDL_CLASSES)

    def crop_mask(self, county: str, year: int, crops=None, geometry=None) -> xr.DataArray:
        """Returns a boolean `(y, x)` mask of the CDL pixels of the given crops.

        `crops` takes CDL names or codes. `None` means every cropland
        class (no urban, water, or natural land cover).
        """
        cdl = self._read_raster("cdl", county, year, geometry=geometry)[0]
        return cdl.isin(catalog.cdl_codes(crops)).rename("crop_mask")

    def load_irrigation(self, county: str, year: int, geometry=None) -> xr.DataArray:
        """Reads irrigated land as a `(y, x)` array: 1 irrigated, 0 not irrigated.

        With `geometry`, pixels outside it are 255, the value in `attrs["nodata"]`.
        """
        return self._read_raster("irrigation", county, year, geometry=geometry)[0]

    def load_landsat(self, county: str, year: int, months: Optional[Iterable[int]] = None, geometry=None) -> xr.DataArray:
        """Reads monthly Landsat reflectance as a `(month, band, y, x)` array.

        Values are returned as stored in the dataset, without rescaling to
        reflectance. A full county is large (about 2 GB for one year of a mid-sized
        county), so pass `months` or `geometry` to read less.
        """
        return self._read_monthly("landsat", county, year, months, geometry)

    def load_et(self, county: str, year: int, months: Optional[Iterable[int]] = None, geometry=None) -> xr.DataArray:
        """Reads monthly OpenET evapotranspiration as a `(month, y, x)` array.

        The ET grid can differ from the CDL and Landsat grid by a pixel at
        the edges, so align the arrays before stacking them.
        """
        return self._read_monthly("et", county, year, months, geometry).isel(band=0, drop=True)

    # Zarr layers.

    def load_climate(
        self,
        county: str,
        years: YearSpec = None,
        variables: StrOrList = None,
        monthly: bool = False,
        geometry=None,
    ) -> xr.Dataset:
        """Reads DayMet weather as an xarray Dataset with `(time, y, x)` variables.

        Variables are `prcp`, `tmax`, `tmin`, `srad`, `vp`, `dayl`, `pet`,
        `swe`, and `snow`. Several years are joined along `time`.
        `monthly=True` returns monthly means instead of daily values.
        DayMet years have 365 days, so leap years end on December 30.
        `geometry` keeps the grid cells inside its bounding box.
        """
        slug = catalog.county_slug(county)
        datasets = [
            self._open_zarr("climate", slug, year, variables, geometry)
            for year in catalog.resolve_years("climate", years)
        ]
        ds = datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")
        if monthly:
            ds = ds.resample(time="MS").mean()
        return ds

    def load_soil(self, county: str, variables: StrOrList = None, geometry=None) -> xr.Dataset:
        """Reads the soil attributes as an xarray Dataset with `(y, x)` variables.

        Categorical variables carry their code labels in
        `ds[name].attrs["categories"]`. `geometry` keeps the grid cells
        inside its bounding box.
        """
        slug = catalog.county_slug(county)
        return self._open_zarr("soil", slug, catalog.LAYERS["soil"].years[0], variables, geometry)

    # Tables.

    def load_yield(
        self,
        county: StrOrList = None,
        crop: StrOrList = None,
        years: YearSpec = None,
        detail: bool = False,
    ) -> pd.DataFrame:
        """Returns county crop yield as a pandas DataFrame.

        The default table has one row per county, crop, and year, with
        `yield_t_per_ha` and the crop group. Crops are merged groups such
        as `"TOMATOES"` or `"GRAPES"`, matched ignoring case. `detail=True`
        returns the per-commodity report rows instead, with harvested
        area, production, price, and value; there `crop` matches
        `Crop_Name_Clean` (e.g. `"Tomatoes Processing"`).
        """
        if detail:
            df = pd.read_parquet(self._file(catalog.YIELD_DETAIL_TABLE))
            county_col, crop_col, year_col = "County", "Crop_Name_Clean", "Year"
        else:
            df = pd.read_parquet(self._file(catalog.YIELD_TABLE))
            county_col, crop_col, year_col = "county", "crop", "year"

        keep = pd.Series(True, index=df.index)
        if county is not None:
            names = [catalog.resolve_county(c) for c in _as_list(county)]
            keep &= df[county_col].isin(names)
        if crop is not None:
            crops = {c.strip().lower() for c in _as_list(crop)}
            keep &= df[crop_col].str.lower().isin(crops)
        if years is not None:
            keep &= df[year_col].isin(catalog.check_years(years, df[year_col].unique(), "The yield table"))
        return df[keep].sort_values([county_col, crop_col, year_col]).reset_index(drop=True)

    def yield_crops(self) -> List[str]:
        """Returns the crop names used in the yield table."""
        return sorted(pd.read_parquet(self._file(catalog.YIELD_TABLE), columns=["crop"])["crop"].unique())

    def cdl_crosswalk(self) -> pd.DataFrame:
        """Returns the table that maps each yield crop to its CDL classes."""
        return pd.read_csv(self._file(catalog.CDL_CROSSWALK))

    def yield_crop_cdl_codes(self, crop: str) -> List[int]:
        """Returns the CDL codes of a yield crop, e.g. `"HAY"` -> `[36, 37]`.

        Returns an empty list for crops that have no CDL class.
        """
        crosswalk = self.cdl_crosswalk()
        row = crosswalk[crosswalk["yield_crop"].str.lower() == crop.strip().lower()]
        if row.empty:
            raise ValueError(f"Unknown yield crop '{crop}'. See `yield_crops()`.")
        codes = row["cdl_codes"].iloc[0]
        return [] if pd.isna(codes) else [int(c) for c in str(codes).split(";")]

    def load_boundaries(self):
        """Returns the California county boundaries as a geopandas GeoDataFrame (EPSG:4326)."""
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("County boundaries need geopandas. Install it with `pip install geopandas`.")
        return gpd.read_file(self._file(catalog.COUNTY_BOUNDARIES))

    # Internals.

    def _file(self, path: str) -> str:
        """Returns the local path of a dataset file, downloading it from the Hub if needed."""
        if self._repo_id is None:
            local = os.path.join(self._root, path)
            if not os.path.exists(local):
                raise FileNotFoundError(f"The dataset file '{path}' is missing: '{local}'.")
            return local

        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return hf_hub_download(
                repo_id=self._repo_id,
                repo_type="dataset",
                filename=path,
                revision=self._revision,
                cache_dir=self._cache_dir,
                token=getattr(self, "_token", None),  # not serialized
            )
        except EntryNotFoundError:
            raise FileNotFoundError(f"The file '{path}' is not in the dataset '{self._repo_id}'.")

    def _store(self, path: str) -> str:
        """Returns the local path of a Zarr store, downloading it from the Hub if needed."""
        if self._repo_id is None:
            local = os.path.join(self._root, path)
            if not os.path.isdir(local):
                raise FileNotFoundError(f"The Zarr store '{path}' is missing: '{local}'.")
            return local
        found, missing = self._fetch([path])
        if missing:
            raise FileNotFoundError(f"The Zarr store '{path}' is not in the dataset '{self._repo_id}'.")
        return found[path]

    def _resolve_commit(self, revision: Optional[str]) -> str:
        """Returns the commit hash of `revision`, so every file comes from one version of the dataset."""
        if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
            return revision
        from huggingface_hub import hf_hub_download

        # The Hugging Face cache names each snapshot folder after its commit. This
        # also works offline once the dataset card is cached.
        card = hf_hub_download(
            repo_id=self._repo_id,
            repo_type="dataset",
            filename=catalog.DATASET_CARD,
            revision=revision,
            cache_dir=self._cache_dir,
            token=self._token,
        )
        return os.path.basename(os.path.dirname(card))

    def _fetch(self, targets: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """Downloads files and Zarr stores from the Hub in one batch.

        Returns the local path of every target in the dataset, and the
        targets that are not in it. Raises a `DownloadError` that lists
        every file that is in the dataset but failed to download.
        """
        from huggingface_hub import hf_hub_download, list_repo_tree
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFile

        token = getattr(self, "_token", None)  # not serialized

        # List each layer folder (`counties/<County>/<layer>`) once to see which files exist.
        listed = set()
        for folder in sorted({"/".join(t.split("/")[:3]) for t in targets}):
            try:
                entries = list_repo_tree(
                    self._repo_id, folder, recursive=True, repo_type="dataset", revision=self._revision, token=token
                )
                listed.update(entry.path for entry in entries if isinstance(entry, RepoFile))
            except EntryNotFoundError:
                pass  # The whole folder is missing.

        # A target is either one file or a Zarr store, i.e. every file under it.
        files = {t: [t] if t in listed else sorted(f for f in listed if f.startswith(t + "/")) for t in targets}
        wanted = [f for names in files.values() for f in names]

        def download(filename):
            return hf_hub_download(
                repo_id=self._repo_id,
                repo_type="dataset",
                filename=filename,
                revision=self._revision,
                cache_dir=self._cache_dir,
                token=token,
            )

        local, failed = {}, {}
        with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
            futures = {f: pool.submit(download, f) for f in wanted}
            for f, future in futures.items():
                try:
                    local[f] = future.result()
                except Exception as error:
                    failed[f] = error
        if failed:
            raise DownloadError(failed)

        found = {}
        for target, names in files.items():
            if names:
                # The cache keeps the repo layout, so a store's folder sits above its files.
                first = local[names[0]]
                found[target] = first if names == [target] else first[: len(first) - len(names[0])] + target
        missing = [t for t, names in files.items() if not names]
        return found, missing

    def _layer_path(self, layer: str, slug: str, year: int, month: Optional[int]) -> str:
        spec = catalog.LAYERS[layer]
        path = spec.path(layer, slug, year, month)
        return self._store(path) if spec.zarr else self._file(path)

    def _read_raster(self, layer: str, county: str, year: int, month: Optional[int] = None, geometry=None) -> xr.DataArray:
        """Reads one GeoTIFF as a `(band, y, x)` array, optionally cropped to a geometry."""
        catalog.resolve_years(layer, year)
        path = self._layer_path(layer, catalog.county_slug(county), year, month)
        with rasterio.open(path) as src:
            nodata = src.nodata
            if geometry is None:
                data, transform = src.read(), src.transform
            else:
                # Pixels outside the geometry get the file's no-data value. Files without
                # one get NaN or the dtype's maximum, never a value that could be a class.
                if nodata is None:
                    dtype = np.dtype(src.dtypes[0])
                    nodata = np.nan if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).max
                data, transform = rasterio.mask.mask(src, [geometry], crop=True, nodata=nodata)
            bands = list(src.descriptions) if all(src.descriptions) else list(range(1, src.count + 1))
            attrs = {"crs": src.crs.to_string(), "transform": tuple(transform)[:6], "nodata": nodata}

        height, width = data.shape[1:]
        x = transform.c + transform.a * (np.arange(width) + 0.5)
        y = transform.f + transform.e * (np.arange(height) + 0.5)
        return xr.DataArray(
            data, dims=("band", "y", "x"), coords={"band": bands, "y": y, "x": x}, name=layer, attrs=attrs
        )

    def _read_monthly(self, layer, county, year, months, geometry) -> xr.DataArray:
        months = _check_months(months)
        arrays = [self._read_raster(layer, county, year, month, geometry) for month in months]
        return xr.concat(arrays, dim=pd.Index(months, name="month"), join="override")

    def _open_zarr(self, layer: str, slug: str, year: int, variables: StrOrList, geometry) -> xr.Dataset:
        try:
            import zarr  # noqa: F401
        except ImportError:
            raise ImportError("The climate and soil layers need zarr 3. Install it with `pip install 'zarr>=3'`.")
        path = self._layer_path(layer, slug, year, None)
        ds = xr.open_dataset(path, engine="zarr", consolidated=False, chunks=None, decode_coords="all")
        # The CRS is stored in a `spatial_ref` variable; keep it as a coordinate, not data.
        if "spatial_ref" in ds.data_vars:
            ds = ds.set_coords("spatial_ref")
        if variables is not None:
            names = list(_as_list(variables))
            unknown = [v for v in names if v not in ds.data_vars]
            if unknown:
                raise ValueError(f"Layer '{layer}' has no variable(s) {unknown}. It has {list(ds.data_vars)}.")
            ds = ds[names]
        if geometry is not None:
            minx, miny, maxx, maxy = rasterio.features.bounds(geometry)
            y_slice = slice(maxy, miny) if ds["y"][0] > ds["y"][-1] else slice(miny, maxy)
            ds = ds.sel(x=slice(minx, maxx), y=y_slice)
        return ds.load()


def _as_list(value) -> list:
    return [value] if isinstance(value, str) else list(value)


def _check_months(months: Optional[Iterable[int]]) -> List[int]:
    months = list(months) if months is not None else list(range(1, 13))
    if any(m not in range(1, 13) for m in months):
        raise ValueError(f"Months must be between 1 and 12, got {months}.")
    return months
