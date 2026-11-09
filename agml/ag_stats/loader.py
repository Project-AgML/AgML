"""Query interface for the USDA NASS crop statistics."""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from agml.framework import AgMLSerializable
from agml.ag_stats.catalog import _DIM_FILES, NASSCatalog, StrOrList, _as_list

# The environment variable read when no `root` is passed to the loader.
_ROOT_ENV_VAR = "AGML_NASS_PATH"

# The Hugging Face dataset repo used when no local copy is given.
DEFAULT_REPO_ID = "Project-AgML/usda_nass_crop_statistics"

# The 2020 Census boundary files, in `boundaries/` next to `parquet/`.
_BOUNDARY_FILES = {
    ("county", True): "us_county_2020.parquet",
    ("county", False): "us_county_2020_attributes.parquet",
    ("state", True): "us_state_2020.parquet",
    ("state", False): "us_state_2020_attributes.parquet",
}

# Maps the `level` argument to the NASS `agg_level` values.
_LEVELS = {
    "national": "NATIONAL",
    "state": "STATE",
    "district": "AGRICULTURAL DISTRICT",
    "county": "COUNTY",
}


# Results are sorted by these columns (when present), then by year and time within the year.
_SORT_COLUMNS = ("crop_id", "agg_level", "state_fips", "state_alpha", "asd_code", "county_fips")
_TIME_COLUMNS = ("forecast_month", "week_ending", "stage", "rating", "statistic")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

YearSpec = Optional[Union[int, Tuple[int, int], Iterable[int]]]


@dataclass(frozen=True)
class _Table:
    """Describes where a fact table lives and which filters it supports."""

    filename: str
    year_column: str = "year"
    has_agg_level: bool = True
    has_state_fips: bool = True
    has_county: bool = True
    has_district: bool = True
    has_other_counties: bool = True
    description: str = ""


# Every fact table in the dataset. The keys are the names used in `query`.
TABLES: Dict[str, _Table] = {
    "yield": _Table(
        "yield.parquet",
        description="Yield at national, state, district, and county level, with a t/ha column.",
    ),
    "county_yield": _Table(
        "v_county_yield_main.parquet",
        has_agg_level=False,
        has_state_fips=False,
        has_district=False,
        has_other_counties=False,
        description="County yield for the main crop variants, on 2020 county FIPS codes, with QA flags.",
    ),
    "area_production": _Table(
        "area_production.parquet",
        description="Area planted, area harvested, and production (`statistic` column).",
    ),
    "census": _Table(
        "census_crops.parquet",
        description="Census of Agriculture crop records, every five years from 1997.",
    ),
    "forecast": _Table(
        "yield_forecast.parquet",
        has_county=False,
        has_district=False,
        has_other_counties=False,
        description="In-season yield forecasts by `forecast_month` (state and national).",
    ),
    "progress": _Table(
        "crop_progress.parquet",
        has_county=False,
        has_district=False,
        has_other_counties=False,
        description="Weekly crop progress by `stage`, as percent (state and national).",
    ),
    "condition": _Table(
        "crop_condition.parquet",
        year_column="crop_year",
        has_county=False,
        has_district=False,
        has_other_counties=False,
        description="Weekly crop condition, one column per rating (state and national).",
    ),
    "condition_long": _Table(
        "crop_condition_long.parquet",
        has_county=False,
        has_district=False,
        has_other_counties=False,
        description="Weekly crop condition, one row per rating (state and national).",
    ),
}

FORMATS = ("pandas", "polars", "arrow", "records", "csv", "parquet")


class NASSDataLoader(AgMLSerializable):
    """Queries USDA NASS crop statistics by crop, place, and year.

    The loader reads the AgML NASS dataset, a set of parquet files built
    from the USDA NASS Quick Stats database. By default the files come
    from the Hugging Face Hub: each table is downloaded the first time it
    is queried and read from the local Hugging Face cache after that.
    Filters are pushed down to the parquet reader, so a query for one
    county reads only a small part of each file.

    Parameters
    ----------
    root : str, optional
        A local copy of the dataset. Either the `parquet` folder itself or
        its parent. Defaults to the `AGML_NASS_PATH` environment variable.
        When neither is set, the dataset is read from `repo_id`.
    repo_id : str, optional
        The Hugging Face dataset repo to read from.
    revision : str, optional
        A branch, tag, or commit of `repo_id`. Pin a commit to make
        results reproducible. Defaults to the latest version.
    cache_dir : str, optional
        Where downloaded files are kept. Defaults to the Hugging Face cache.
    token : str, optional
        A Hugging Face token. Not needed for the public dataset.

    Examples
    --------
    >>> nass = NASSDataLoader()  # reads from the Hugging Face Hub
    >>> nass = NASSDataLoader("/path/to/nass-usda-full-data")  # a local copy
    >>> df = nass.yield_data(crop="corn", utilization="grain", state="IA",
    ...                      county="Story", years=(2000, 2024))
    >>> nass.query("progress", crop="soybeans", state="IL", years=2023,
    ...            where={"stage": "PLANTED"}, format="polars")
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
        self._cache_dir = cache_dir
        self._token = token
        root = root or os.environ.get(_ROOT_ENV_VAR)
        if root is None:
            # `repo_id` is only kept in Hub mode; `None` marks a local copy.
            self._repo_id = repo_id
            root = self._download_dimension_tables(revision)
        else:
            self._repo_id = None
            self._revision = None
            root = os.path.realpath(os.path.expanduser(root))
            if os.path.isdir(os.path.join(root, "parquet")):
                root = os.path.join(root, "parquet")
            if not os.path.isdir(root):
                raise FileNotFoundError(f"The NASS dataset directory '{root}' does not exist.")
        self._root = root
        self._catalog = NASSCatalog(root)

    def __repr__(self):
        if self._repo_id is not None:
            return f"<NASSDataLoader: {self._repo_id}@{self._revision[:7]}>"
        return f"<NASSDataLoader: {self.root}>"

    @property
    def root(self) -> str:
        """The directory holding the NASS parquet files."""
        return self._root

    @property
    def repo_id(self) -> Optional[str]:
        """The Hugging Face repo the data comes from, or `None` for a local copy."""
        return self._repo_id

    @property
    def revision(self) -> Optional[str]:
        """The commit of `repo_id` the data comes from, or `None` for a local copy."""
        return self._revision

    @property
    def catalog(self) -> NASSCatalog:
        """The `NASSCatalog` used to resolve crop and place names."""
        # Only `root` is serialized, so the catalog is rebuilt after unpickling.
        if getattr(self, "_catalog", None) is None:
            self._catalog = NASSCatalog(self._root)
        return self._catalog

    @staticmethod
    def tables() -> Dict[str, str]:
        """Returns the name and a one-line description of every queryable table."""
        return {name: table.description for name, table in TABLES.items()}

    def schema(self, table: str) -> pa.Schema:
        """Returns the column names and types of a table."""
        return self._dataset(table).schema

    def query(
        self,
        table: str,
        crop: StrOrList = None,
        crop_class: StrOrList = None,
        practice: StrOrList = None,
        utilization: StrOrList = None,
        state: StrOrList = None,
        county: StrOrList = None,
        district: StrOrList = None,
        years: YearSpec = None,
        level: Optional[str] = None,
        where: Optional[Dict[str, Union[str, int, float, bool, Sequence]]] = None,
        columns: Optional[Sequence[str]] = None,
        include_other_counties: bool = False,
        format: str = "pandas",
        path: Optional[str] = None,
    ):
        """Returns the rows of a table that match every given filter.

        Parameters
        ----------
        table : str
            One of the names in `NASSDataLoader.tables()`.
        crop, crop_class, practice, utilization : str or list, optional
            The crop, resolved with `NASSCatalog.resolve_crops`. `crop`
            matches a commodity (`"corn"`), a crop label (`"CORN, GRAIN"`),
            or a `crop_id`.
        state : str, int, or list, optional
            Postal codes, names, or FIPS codes.
        county : str, int, or list, optional
            5-digit FIPS codes or county names (names need `state` when
            they are not unique).
        district : str or list, optional
            Agricultural statistics districts, as 4-digit `asd_id` codes or
            as codes or names together with `state`.
        years : int, tuple, or list, optional
            One year, an inclusive `(start, end)` range, or a list of years.
        level : str, optional
            One of `national`, `state`, `district`, or `county`.
        where : dict, optional
            Extra equality filters on any column, e.g.
            `{"statistic": "production"}`. A list value matches any of its items.
        columns : list of str, optional
            The columns to return. Defaults to every column.
        include_other_counties : bool
            Whether to keep the NASS "other (combined) counties" rows.
            These have no county FIPS code and are dropped by default.
        format : str
            One of `pandas`, `polars`, `arrow`, `records`, `csv`, or `parquet`.
        path : str, optional
            The output file, required for the `csv` and `parquet` formats.

        Returns
        -------
        The matching rows in the requested format, sorted by crop, place,
        and time, or the output file path for the `csv` and `parquet` formats.
        """
        spec = self._table_spec(table)
        self._check_format(format, path)

        dataset = self._dataset(table)
        expression = self._build_filter(
            table, spec, dataset.schema, crop, crop_class, practice, utilization,
            state, county, district, years, level, where, include_other_counties,
        )
        if columns is not None:
            missing = [c for c in columns if c not in dataset.schema.names]
            if missing:
                raise ValueError(f"Table '{table}' has no column(s) {missing}.")
        result = dataset.to_table(filter=expression, columns=list(columns) if columns else None)
        return self._convert(self._sort(result, spec), format, path)

    # Convenience wrappers, one per table.

    def yield_data(self, **kwargs):
        """Queries the `yield` table. Takes the same arguments as `query`."""
        return self.query("yield", **kwargs)

    def county_yield(self, **kwargs):
        """Queries the `county_yield` table. Takes the same arguments as `query`."""
        return self.query("county_yield", **kwargs)

    def area_production(self, statistic: StrOrList = None, **kwargs):
        """Queries the `area_production` table.

        `statistic` is one or more of `area_planted`, `area_harvested`, and
        `production`. The other arguments are the same as for `query`.
        """
        where = self._merge_where(kwargs, statistic=statistic)
        return self.query("area_production", where=where, **kwargs)

    def census(self, statistic: StrOrList = None, **kwargs):
        """Queries the `census` table. See `area_production` for `statistic`."""
        where = self._merge_where(kwargs, statistic=statistic)
        return self.query("census", where=where, **kwargs)

    def forecast(self, **kwargs):
        """Queries the `forecast` table. Takes the same arguments as `query`."""
        return self.query("forecast", **kwargs)

    def progress(self, stage: StrOrList = None, **kwargs):
        """Queries the `progress` table, e.g. `stage="planted"`.

        `stage` ignores letter case and an optional `PCT ` prefix, so
        `"planted"`, `"PLANTED"`, and `"PCT PLANTED"` are the same.
        """
        if stage is not None:
            stage = [s.strip().upper().removeprefix("PCT ") for s in _as_list(stage)]
        where = self._merge_where(kwargs, stage=stage)
        return self.query("progress", where=where, **kwargs)

    def condition(self, long: bool = False, **kwargs):
        """Queries the weekly condition ratings, wide (default) or long."""
        return self.query("condition_long" if long else "condition", **kwargs)

    def boundaries(self, level: str = "county", geometry: bool = True, format: str = "pandas", path: Optional[str] = None):
        """Returns the 2020 Census county or state boundaries.

        Each row has the FIPS code (`GEOID`), names, area, and centroid.
        With `geometry=True` it also has a `geometry` column of WKB
        polygons, e.g. for `geopandas.GeoSeries.from_wkb`.
        """
        if level not in ("county", "state"):
            raise ValueError(f"Unknown boundary level '{level}'. Choose 'county' or 'state'.")
        self._check_format(format, path)
        filename = _BOUNDARY_FILES[(level, geometry)]
        result = ds.dataset(self._file("boundaries", filename), format="parquet").to_table()
        return self._convert(result, format, path)

    # Internals.

    def _download_dimension_tables(self, revision: Optional[str]) -> str:
        """Downloads the small lookup tables and returns their local directory.

        The commit they come from is kept in `_revision`, so every table
        downloaded later comes from the same version of the dataset.
        """
        from huggingface_hub import snapshot_download

        folder = snapshot_download(
            repo_id=self._repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=self._cache_dir,
            token=self._token,
            allow_patterns=[f"parquet/{name}" for name in _DIM_FILES.values()],
        )
        # The Hugging Face cache names each snapshot folder after its commit.
        self._revision = os.path.basename(folder)
        return os.path.join(folder, "parquet")

    def _file(self, folder: str, filename: str) -> str:
        """Returns the local path of a dataset file, downloading it from the Hub if needed."""
        if self._repo_id is None:
            base = self.root if folder == "parquet" else os.path.join(os.path.dirname(self.root), folder)
            path = os.path.join(base, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"The dataset file '{filename}' is missing: '{path}'.")
            return path

        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self._repo_id,
            repo_type="dataset",
            filename=f"{folder}/{filename}",
            revision=self._revision,
            cache_dir=self._cache_dir,
            token=getattr(self, "_token", None),  # not serialized
        )

    @staticmethod
    def _table_spec(table: str) -> _Table:
        if table not in TABLES:
            raise ValueError(f"Unknown table '{table}'. Choose one of {list(TABLES)}.")
        return TABLES[table]

    def _dataset(self, table: str) -> ds.Dataset:
        return ds.dataset(self._file("parquet", self._table_spec(table).filename), format="parquet")

    @staticmethod
    def _sort(result: pa.Table, spec: _Table) -> pa.Table:
        """Sorts rows by crop, place, and time. Forecast months sort in calendar order."""
        names = result.schema.names
        order = [c for c in (*_SORT_COLUMNS, spec.year_column, *_TIME_COLUMNS) if c in names]
        if not order:
            return result
        if "forecast_month" in names:
            month = pc.index_in(result["forecast_month"], value_set=pa.array(_MONTHS))
            result = result.append_column("_month", month)
            order = ["_month" if c == "forecast_month" else c for c in order]
        result = result.sort_by([(c, "ascending") for c in order])
        return result.drop_columns(["_month"]) if "_month" in result.schema.names else result

    @staticmethod
    def _merge_where(kwargs: dict, **extra) -> Optional[dict]:
        """Combines a wrapper's own column filter with any `where` the caller passed."""
        where = dict(kwargs.pop("where", None) or {})
        where.update({k: v for k, v in extra.items() if v is not None})
        return where or None

    @staticmethod
    def _years_filter(column: str, years: YearSpec) -> ds.Expression:
        field = pc.field(column)
        if isinstance(years, int):
            return field == years
        if isinstance(years, tuple) and len(years) == 2:
            start, end = years
            return (field >= start) & (field <= end)
        return field.isin([int(y) for y in years])

    def _build_filter(
        self, table, spec, schema, crop, crop_class, practice, utilization,
        state, county, district, years, level, where, include_other_counties,
    ) -> Optional[ds.Expression]:
        """Turns the query arguments into one pyarrow filter expression."""

        def unsupported(argument, what):
            return ValueError(f"Table '{table}' has no {what} data, so `{argument}` cannot be used with it.")

        filters = []

        if any(v is not None for v in (crop, crop_class, practice, utilization)):
            crop_ids = self.catalog.resolve_crops(crop, crop_class, practice, utilization)
            filters.append(pc.field("crop_id").isin(crop_ids))

        state_fips = self.catalog.resolve_states(state) if state is not None else None
        if state_fips is not None:
            if spec.has_state_fips:
                filters.append(pc.field("state_fips").isin(state_fips))
            else:
                filters.append(pc.field("state_alpha").isin(self.catalog.state_alphas(state_fips)))

        if county is not None:
            if not spec.has_county:
                raise unsupported("county", "county-level")
            filters.append(pc.field("county_fips").isin(self.catalog.resolve_counties(county, state=state)))

        if district is not None:
            if not spec.has_district:
                raise unsupported("district", "district-level")
            asd_ids = self.catalog.resolve_districts(district, state=state)
            by_state = {}
            for asd_id in asd_ids:
                by_state.setdefault(asd_id[:2], []).append(asd_id[2:])
            district_filter = None
            for fips, codes in by_state.items():
                part = (pc.field("state_fips") == fips) & pc.field("asd_code").isin(codes)
                district_filter = part if district_filter is None else district_filter | part
            filters.append(district_filter)

        if years is not None:
            filters.append(self._years_filter(spec.year_column, years))

        if level is not None:
            if level not in _LEVELS:
                raise ValueError(f"Unknown level '{level}'. Choose one of {list(_LEVELS)}.")
            if spec.has_agg_level:
                filters.append(pc.field("agg_level") == _LEVELS[level])
            elif level != "county":
                raise unsupported("level", f"{level}-level")

        if spec.has_other_counties and not include_other_counties:
            filters.append(pc.field("is_other_counties").is_null() | ~pc.field("is_other_counties"))

        for column, value in (where or {}).items():
            if column not in schema.names:
                raise ValueError(f"Table '{table}' has no column '{column}'.")
            values = value if isinstance(value, (list, tuple, set)) else [value]
            filters.append(pc.field(column).isin(list(values)))

        if not filters:
            return None
        expression = filters[0]
        for f in filters[1:]:
            expression = expression & f
        return expression

    @staticmethod
    def _check_format(format: str, path: Optional[str]):
        if format not in FORMATS:
            raise ValueError(f"Unknown format '{format}'. Choose one of {FORMATS}.")
        if format in ("csv", "parquet") and path is None:
            raise ValueError(f"The '{format}' format needs an output `path`.")

    @staticmethod
    def _convert(result: pa.Table, format: str, path: Optional[str]):
        """Converts an Arrow table into the requested output format."""
        if format == "arrow":
            return result
        if format == "pandas":
            return result.to_pandas()
        if format == "records":
            return result.to_pylist()
        if format == "polars":
            try:
                import polars as pl
            except ImportError:
                raise ImportError("The 'polars' format needs polars. Install it with `pip install polars`.")
            return pl.from_arrow(result)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if format == "csv":
            import pyarrow.csv as pacsv

            pacsv.write_csv(result, path)
        else:
            import pyarrow.parquet as pq

            pq.write_table(result, path)
        return path
