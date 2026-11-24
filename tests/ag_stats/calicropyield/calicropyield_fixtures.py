"""Values shared by the California Crop Yield Benchmark fixture dataset and the tests."""

import numpy as np

# The CDL pixel size, in degrees.
RES = 0.00026949458523585647

# CDL codes on the fixture grid: tomatoes (54), almonds (75), alfalfa (36), urban (121), water (141).
CDL = np.array(
    [
        [54, 54, 75, 75, 121, 141],
        [54, 54, 75, 75, 121, 141],
        [36, 36, 0, 0, 121, 141],
        [36, 36, 0, 0, 121, 141],
    ],
    dtype="uint8",
)

LANDSAT_BANDS = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7")
