"""Layout of the California Crop Yield Benchmark dataset on the Hugging Face Hub.

Every county has one folder per data layer, `counties/<County>/<layer>/<year>/`,
where `<County>` is the county name without spaces (`ContraCosta`). Rasters
are GeoTIFFs, and the climate and soil layers are Zarr stores. Crop yield is
one statewide table, `yield/county_crop_yield.parquet`, with a more detailed
table and lookup files next to it.
"""

import difflib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

DEFAULT_REPO_ID = "hkaman/california-crop-yield-benchmark"

# Files outside the county folders.
YIELD_TABLE = "yield/county_crop_yield.parquet"
YIELD_DETAIL_TABLE = "yield/county_crop_yield_detail.parquet"
CDL_CROSSWALK = "yield/metadata/cdl_crop_crosswalk.csv"
COUNTY_BOUNDARIES = "reference/California_Counties.geojson"

# 2012 is missing from every layer.
_YEARS = tuple(y for y in range(2008, 2024) if y != 2012)


@dataclass(frozen=True)
class Layer:
    """Describes how one data layer is stored."""

    filename: str
    years: Tuple[int, ...]
    monthly: bool = False
    zarr: bool = False
    description: str = ""

    def path(self, name: str, slug: str, year: int, month: Optional[int] = None) -> str:
        """Returns the repo path of one file, e.g. `counties/Yolo/et/2020/Yolo_ET_2020_07.tif`."""
        filename = self.filename.format(county=slug, year=year, month=month)
        return f"counties/{slug}/{name}/{year}/{filename}"


# Every gridded layer. The keys are the names used by the loader.
LAYERS: Dict[str, Layer] = {
    "cdl": Layer(
        "{county}_{year}.tif",
        years=_YEARS + (2024,),
        description="USDA Cropland Data Layer, one crop class code per 30 m pixel.",
    ),
    "landsat": Layer(
        "{county}_LT_{year}_{month:02d}.tif",
        years=_YEARS,
        monthly=True,
        description="Monthly Landsat surface reflectance, bands SR_B1-SR_B5 and SR_B7.",
    ),
    "et": Layer(
        "{county}_ET_{year}_{month:02d}.tif",
        years=_YEARS,
        monthly=True,
        description="Monthly OpenET evapotranspiration.",
    ),
    "irrigation": Layer(
        "{county}_{year}.tif",
        years=_YEARS,
        description="IrrMapper irrigated land (1) and not irrigated (0), on the CDL grid.",
    ),
    "climate": Layer(
        "DayMet_{county}_{year}.zarr",
        years=tuple(y for y in _YEARS if y <= 2022),
        zarr=True,
        description="Daily DayMet weather at 1 km: prcp, tmax, tmin, srad, vp, dayl, pet, swe, snow.",
    ),
    "soil": Layer(
        "{county}_soil.zarr",
        years=(2020,),
        zarr=True,
        description="17 gNATSGO/SSURGO soil attributes (2020 vintage), on a 30 m grid.",
    ),
}

COUNTIES: Tuple[str, ...] = (
    "Alameda", "Alpine", "Amador", "Butte", "Calaveras", "Colusa",
    "Contra Costa", "Del Norte", "El Dorado", "Fresno", "Glenn", "Humboldt",
    "Imperial", "Inyo", "Kern", "Kings", "Lake", "Lassen", "Los Angeles",
    "Madera", "Marin", "Mariposa", "Mendocino", "Merced", "Modoc", "Mono",
    "Monterey", "Napa", "Nevada", "Orange", "Placer", "Plumas", "Riverside",
    "Sacramento", "San Benito", "San Bernardino", "San Diego",
    "San Francisco", "San Joaquin", "San Luis Obispo", "San Mateo",
    "Santa Barbara", "Santa Clara", "Santa Cruz", "Shasta", "Sierra",
    "Siskiyou", "Solano", "Sonoma", "Stanislaus", "Sutter", "Tehama",
    "Trinity", "Tulare", "Tuolumne", "Ventura", "Yolo", "Yuba",
)

# USDA CDL class codes used in California. Codes 62-65, 81-91, and 94-199
# are non-agricultural or NLCD land cover.
CDL_CLASSES: Dict[int, str] = {
    0: "Other", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Sorghum", 5: "Soybeans",
    6: "Sunflower", 10: "Peanuts", 11: "Tobacco", 12: "Sweet Corn",
    13: "Pop or Orn Corn", 14: "Mint", 21: "Barley", 22: "Durum Wheat",
    23: "Spring Wheat", 24: "Winter Wheat", 25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans", 27: "Rye", 28: "Oats", 29: "Millet",
    30: "Speltz", 31: "Canola", 32: "Flaxseed", 33: "Safflower", 34: "Rape Seed",
    35: "Mustard", 36: "Alfalfa", 37: "Other Hay/Non Alfalfa", 38: "Camelina",
    39: "Buckwheat", 41: "Sugarbeets", 42: "Dry Beans", 43: "Potatoes",
    44: "Other Crops", 45: "Sugarcane", 46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits", 48: "Watermelons", 49: "Onions", 50: "Cucumbers",
    51: "Chick Peas", 52: "Lentils", 53: "Peas", 54: "Tomatoes",
    55: "Caneberries", 56: "Hops", 57: "Herbs", 58: "Clover/Wildflowers",
    59: "Sod/Grass Seed", 60: "Switchgrass", 61: "Fallow/Idle Cropland",
    **{code: "Non-agricultural" for code in (62, 63, 64, 65, *range(81, 92))},
    66: "Cherries", 67: "Peaches", 68: "Apples", 69: "Grapes",
    70: "Christmas Trees", 71: "Other Tree Crops", 72: "Citrus", 74: "Pecans",
    75: "Almonds", 76: "Walnuts", 77: "Pears",
    **{code: "NLCD-sampled categories" for code in range(93, 200)},
    121: "Urban", 141: "Water",
    204: "Pistachios", 205: "Triticale", 206: "Carrots", 207: "Asparagus",
    208: "Garlic", 209: "Cantaloupes", 210: "Prunes", 211: "Olives",
    212: "Oranges", 213: "Honeydew Melons", 214: "Broccoli", 215: "Avocados",
    216: "Peppers", 217: "Pomegranates", 218: "Nectarines", 219: "Greens",
    220: "Plums", 221: "Strawberries", 222: "Squash", 223: "Apricots",
    224: "Vetch", 225: "Dbl Crop WinWht/Corn", 226: "Dbl Crop Oats/Corn",
    227: "Lettuce", 228: "Dbl Crop Triticale/Corn", 229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht", 231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton", 233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum", 235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum", 237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton", 239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats", 241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries", 243: "Cabbage", 244: "Cauliflower", 245: "Celery",
    246: "Radishes", 247: "Turnips", 248: "Eggplants", 249: "Gourds",
    250: "Cranberries", 254: "Dbl Crop Barley/Soybeans",
}

# CDL classes that are not cropland.
NON_CROP_CLASSES = frozenset(("Other", "Non-agricultural", "NLCD-sampled categories", "Urban", "Water"))


def layer(name: str) -> Layer:
    """Returns the description of a layer, or raises a helpful error."""
    if name not in LAYERS:
        raise ValueError(f"Unknown layer '{name}'. Choose one of {list(LAYERS)}.")
    return LAYERS[name]


def resolve_county(county: str) -> str:
    """Returns the canonical county name, ignoring case, spaces, and a ` County` suffix."""
    key = _county_key(county)
    for name in COUNTIES:
        if _county_key(name) == key:
            return name
    close = difflib.get_close_matches(county.strip().title(), COUNTIES, n=3)
    hint = f" Did you mean {close}?" if close else ""
    raise ValueError(f"Unknown California county '{county}'.{hint}")


def county_slug(county: str) -> str:
    """Returns the folder name of a county on the Hub, e.g. `ContraCosta`."""
    return resolve_county(county).replace(" ", "")


def resolve_years(name: str, years: Union[None, int, Tuple[int, int], Iterable[int]]) -> List[int]:
    """Turns a year, an inclusive `(start, end)` range, or a list into the years a layer has.

    Raises an error for years the layer does not have, so a typo is not
    silently skipped. `None` means every year.
    """
    available = layer(name).years
    if years is None:
        return list(available)
    if isinstance(years, int):
        wanted = [years]
    elif isinstance(years, tuple) and len(years) == 2:
        wanted = [y for y in available if years[0] <= y <= years[1]]
    else:
        wanted = [int(y) for y in years]
    missing = [y for y in wanted if y not in available]
    if missing:
        raise ValueError(f"Layer '{name}' has no data for {missing}. It covers {available[0]}-{available[-1]} except 2012.")
    return wanted


def cdl_codes(crops: Union[str, int, Iterable[Union[str, int]], None]) -> List[int]:
    """Returns the CDL codes for class names or codes. `None` means every cropland class."""
    if crops is None:
        return [code for code, name in CDL_CLASSES.items() if name not in NON_CROP_CLASSES]
    if isinstance(crops, (str, int)):
        crops = [crops]
    by_name = {}
    for code, name in CDL_CLASSES.items():
        by_name.setdefault(name.lower(), []).append(code)
    codes = []
    for crop in crops:
        if isinstance(crop, int):
            codes.append(crop)
        elif crop.strip().lower() in by_name:
            codes.extend(by_name[crop.strip().lower()])
        else:
            close = difflib.get_close_matches(crop, sorted(set(CDL_CLASSES.values())), n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(f"Unknown CDL class '{crop}'.{hint}")
    return codes


def _county_key(name: str) -> str:
    name = name.strip().lower()
    name = name[: -len(" county")] if name.endswith(" county") else name
    return name.replace(" ", "")
