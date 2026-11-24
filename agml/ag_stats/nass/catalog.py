"""Lookup tables and name resolution for the NASS crop statistics.

The NASS fact tables are keyed by codes (a 12-character `crop_id`, 2-digit
state FIPS, 5-digit county FIPS, and agricultural statistics district
codes). The `NASSCatalog` reads the small dimension tables that ship with
the dataset and turns what a user types ("corn", "Iowa", "Story") into
those codes.
"""

import difflib
import os
from typing import Iterable, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

# Dimension tables, keyed by the name used inside the catalog.
_DIM_FILES = {
    "crop": "dim_crop.parquet",
    "state": "dim_state.parquet",
    "county": "dim_county.parquet",
    "district": "dim_asd.parquet",
    "county_remap": "county_fips_remap.parquet",
}

# Columns of `dim_crop` that a user can filter on besides the commodity.
_CROP_ATTRIBUTES = ("class", "practice", "utilization")

StrOrList = Optional[Union[str, int, Iterable[Union[str, int]]]]


def _as_list(value: StrOrList) -> List[str]:
    """Turns a scalar or an iterable of values into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [str(value)]
    return [str(v) for v in value]


def _no_match(kind: str, value: str, choices: Iterable[str]) -> ValueError:
    """Builds a `ValueError` that suggests the closest valid choices."""
    unique = sorted({str(c).upper() for c in choices})
    close = difflib.get_close_matches(value.upper(), unique, n=5, cutoff=0.6)
    hint = f" Did you mean one of: {', '.join(close)}?" if close else ""
    return ValueError(f"Unknown {kind} '{value}'.{hint}")


class NASSCatalog(object):
    """Resolves crop, state, county, and district names to NASS codes.

    Parameters
    ----------
    root : str
        The directory holding the NASS parquet files (`dim_crop.parquet`,
        `dim_state.parquet`, and so on).

    Examples
    --------
    >>> catalog = NASSCatalog("/path/to/nass-usda-full-data/parquet")
    >>> catalog.resolve_states("Iowa")
    ['19']
    >>> catalog.resolve_counties("Story", state="IA")
    ['19169']
    """

    def __init__(self, root: str):
        self.root = root
        self._dims = {}
        for name, filename in _DIM_FILES.items():
            path = os.path.join(root, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not find the NASS dimension table '{filename}' in '{root}'.")
            self._dims[name] = pq.read_table(path).to_pandas()

    # Listings.

    def crops(self, commodity: Optional[str] = None) -> pd.DataFrame:
        """Returns the crop table, optionally only the rows of one commodity."""
        df = self._dims["crop"]
        if commodity is not None:
            df = df[df["commodity"] == commodity.upper()]
        return df.reset_index(drop=True)

    def commodities(self) -> List[str]:
        """Returns the sorted list of commodity names (e.g., `CORN`)."""
        return sorted(self._dims["crop"]["commodity"].unique())

    def states(self) -> pd.DataFrame:
        """Returns the state table sorted by state FIPS code."""
        return self._dims["state"].sort_values("state_fips").reset_index(drop=True)

    def counties(self, state: StrOrList = None) -> pd.DataFrame:
        """Returns the county table, optionally only for the given states."""
        df = self._dims["county"]
        if state is not None:
            df = df[df["state_fips"].isin(self.resolve_states(state))]
        return df.sort_values("county_fips").reset_index(drop=True)

    def districts(self, state: StrOrList = None) -> pd.DataFrame:
        """Returns the agricultural statistics district table."""
        df = self._dims["district"]
        if state is not None:
            df = df[df["state_fips"].isin(self.resolve_states(state))]
        return df.sort_values("asd_id").reset_index(drop=True)

    # Name resolution.

    def resolve_states(self, state: StrOrList) -> List[str]:
        """Resolves states to 2-digit FIPS codes.

        Each value can be a postal code (`"IA"`), a name (`"Iowa"`), or a
        FIPS code (`"19"` or `19`), in any letter case.
        """
        df = self._dims["state"]
        by_alpha = dict(zip(df["state_alpha"].str.upper(), df["state_fips"]))
        by_name = dict(zip(df["state_name"].str.upper(), df["state_fips"]))
        fips = set(df["state_fips"])

        out = []
        for value in _as_list(state):
            key = value.strip().upper()
            if key.isdigit() and key.zfill(2) in fips:
                out.append(key.zfill(2))
            elif key in by_alpha:
                out.append(by_alpha[key])
            elif key in by_name:
                out.append(by_name[key])
            else:
                raise _no_match("state", value, list(by_alpha) + list(by_name))
        return out

    def state_alphas(self, state_fips: Iterable[str]) -> List[str]:
        """Maps 2-digit state FIPS codes to postal codes."""
        df = self._dims["state"]
        lookup = dict(zip(df["state_fips"], df["state_alpha"]))
        return [lookup[f] for f in state_fips]

    def resolve_counties(self, county: StrOrList, state: StrOrList = None) -> List[str]:
        """Resolves counties to 5-digit 2020 county FIPS codes.

        Each value can be a 5-digit FIPS code or a county name. Names are
        not unique across states (there are 31 Washington counties), so a
        name needs `state` unless it occurs in only one state. Old NASS
        codes for renamed or merged counties are mapped to their 2020 code.
        """
        df = self._dims["county"]
        remap = dict(zip(self._dims["county_remap"]["fips_nass"], self._dims["county_remap"]["fips_2020"]))
        fips = set(df["county_fips"])
        if state is not None:
            df = df[df["state_fips"].isin(self.resolve_states(state))]

        out = []
        for value in _as_list(county):
            key = value.strip()
            if key.isdigit():
                code = remap.get(key.zfill(5), key.zfill(5))
                if code not in fips:
                    raise ValueError(f"Unknown county FIPS code '{value}'.")
                out.append(code)
                continue

            name = key.upper().removesuffix(" COUNTY")
            matches = df[df["county_name"].str.upper() == name]
            if matches.empty:
                raise _no_match("county", value, df["county_name"])
            if matches["state_fips"].nunique() > 1:
                states = ", ".join(sorted(matches["state_alpha"]))
                raise ValueError(f"County name '{value}' exists in several states ({states}). Pass `state` as well.")
            out.extend(matches["county_fips"])
        return out

    def resolve_districts(self, district: StrOrList, state: StrOrList = None) -> List[str]:
        """Resolves agricultural statistics districts to 4-digit `asd_id` codes.

        Each value can be a 4-digit `asd_id` (state FIPS + district code),
        or a 2-digit district code or a district name together with `state`.
        """
        df = self._dims["district"]
        if state is not None:
            df = df[df["state_fips"].isin(self.resolve_states(state))]

        out = []
        for value in _as_list(district):
            key = value.strip().upper()
            if key.isdigit() and len(key) == 4:
                matches = self._dims["district"][self._dims["district"]["asd_id"] == key]
            elif key.isdigit():
                if state is None:
                    raise ValueError(f"District code '{value}' is only unique within a state. Pass `state` as well.")
                matches = df[df["asd_code"] == key.zfill(2)]
            else:
                matches = df[df["asd_desc"].str.upper() == key]
                if matches["state_fips"].nunique() > 1:
                    raise ValueError(f"District name '{value}' exists in several states. Pass `state` as well.")
            if matches.empty:
                raise _no_match("district", value, df["asd_desc"])
            out.extend(matches["asd_id"])
        return out

    def resolve_crops(
        self,
        crop: StrOrList = None,
        crop_class: StrOrList = None,
        practice: StrOrList = None,
        utilization: StrOrList = None,
    ) -> List[str]:
        """Resolves a crop description to the matching `crop_id` codes.

        `crop` is matched against the commodity first (`"corn"` gives every
        corn variant), then against the full crop label (`"CORN, GRAIN"`),
        then against `crop_id` itself. `crop_class`, `practice`, and
        `utilization` narrow the result further (e.g., `utilization="grain"`).
        All matches ignore letter case.
        """
        df = self._dims["crop"]
        if crop is not None:
            keep = pd.Series(False, index=df.index)
            for value in _as_list(crop):
                key = value.strip().upper()
                for column in ("commodity", "crop_label"):
                    hit = df[column] == key
                    if hit.any():
                        break
                else:
                    hit = df["crop_id"] == value.strip()
                if not hit.any():
                    raise _no_match("crop", value, list(df["commodity"].unique()) + list(df["crop_label"]))
                keep |= hit
            df = df[keep]

        for column, value in zip(_CROP_ATTRIBUTES, (crop_class, practice, utilization)):
            if value is None:
                continue
            wanted = [v.strip().upper() for v in _as_list(value)]
            narrowed = df[df[column].isin(wanted)]
            if narrowed.empty:
                raise _no_match(column, ", ".join(wanted), df[column].unique())
            df = narrowed
        return list(df["crop_id"])
