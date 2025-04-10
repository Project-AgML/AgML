# Copyright 2021 UC Davis Plant AI and Biophysics Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO

import requests
import numpy as np
from tqdm import tqdm

import agml

import pycountry
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import DBSCAN


# Get all the available countries from AGML dataset
countries = []
for ds in agml.data.public_data_sources():
    country = ds.location.country
    if country is not None:
        countries.append(country.title())

# Convert country names to lowercase for case insensitivity
countries = [country.lower() for country in countries]

# Standardize country names to match GeoPandas dataset
country_mapping = {
    "usa": "united states of america",
    "uk": "united kingdom",
    "south korea": "korea, republic of",
    # Add more mappings as needed
}

# Define custom flag positions for specific countries
custom_positions = {
    "united states of america": (-98.5795, 39.8283)  # Center of continental US
}
countries = [country_mapping.get(country, country) for country in countries]

# Load world map
gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf = gdf[gdf.geometry.notnull()]

# Create a column to highlight relevant countries
gdf['highlight'] = gdf['name'].str.lower().apply(lambda x: x in countries)
gdf_highlighted = gdf[gdf['highlight']]

# Calculate centroids and areas
gdf_highlighted['centroid'] = gdf_highlighted.geometry.centroid
gdf_highlighted['area'] = gdf_highlighted.geometry.area


def get_country_code(country_name):
    """Convert country name to ISO 2-letter code for flag retrieval."""
    try:
        return pycountry.countries.lookup(country_name).alpha_2.lower()
    except LookupError:
        return None


def optimize_flag_positions(points, min_distance=5):
    """
    Optimize flag positions to avoid overlapping using DBSCAN clustering
    and smart repositioning within valid ocean spaces.
    """
    # Convert points to numpy array for clustering
    points_array = np.array([[p[0], p[1]] for p in points])

    # Use DBSCAN to identify clusters of close points
    clustering = DBSCAN(eps=min_distance, min_samples=2).fit(points_array)

    # Initialize optimized positions
    optimized_positions = points_array.copy()

    # Handle each cluster
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_mask = clustering.labels_ == cluster_id
        cluster_points = points_array[cluster_mask]

        # Calculate cluster center
        center = cluster_points.mean(axis=0)

        # Spread points around center
        n_points = len(cluster_points)
        if n_points > 1:
            radius = min_distance
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

            for idx, angle in enumerate(angles):
                new_x = center[0] + radius * np.cos(angle)
                new_y = center[1] + radius * np.sin(angle)
                optimized_positions[cluster_mask][idx] = [new_x, new_y]

    return optimized_positions


# Collect all flag positions first
flag_positions = []
country_data = []

for _, row in gdf_highlighted.iterrows():
    # Use custom position if available, otherwise use centroid
    if row['name'].lower() in custom_positions:
        lon, lat = custom_positions[row['name'].lower()]
    else:
        lon, lat = row['centroid'].x, row['centroid'].y
    country_code = get_country_code(row['name'])

    if country_code:
        flag_positions.append([lon, lat])
        country_data.append({
            'name': row['name'],
            'code': country_code,
            'original_position': (lon, lat),
            'area': row['area']
        })

# Optimize flag positions
optimized_positions = optimize_flag_positions(flag_positions)
# Calculate bounds of all land masses
total_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
bounds_width = total_bounds[2] - total_bounds[0]
bounds_height = total_bounds[3] - total_bounds[1]

# Calculate optimal figure size maintaining aspect ratio
width = 24  # Base width
height = width * (bounds_height / bounds_width)  # Proportional height

# Set up the plot with calculated dimensions
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(width, height), facecolor='#F0F8FF')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Plot base map
gdf.plot(ax=ax, color='#b3ffd1', linewidth=0.5, zorder=1)
gdf_highlighted.plot(ax=ax, color='#09e35f', edgecolor='#2196F3', linewidth=0.8, zorder=2)

# Add flags with optimized positions
for i, country in enumerate(tqdm(country_data, desc="Adding flags")):
    opt_lon, opt_lat = optimized_positions[i]
    orig_lon, orig_lat = country['original_position']

    # Get high-resolution flag
    flag_url = f"https://flagcdn.com/w40/{country['code']}.png"  # Using smaller flags

    try:
        response = requests.get(flag_url, timeout=5)
        if response.status_code == 200:
            img = mpimg.imread(BytesIO(response.content))

            # Create flag image with enhanced quality
            imagebox = OffsetImage(img, zoom=0.5, interpolation='hanning')
            imagebox.image.axes = ax

            # Add connection line if flag position was adjusted
            if (opt_lon, opt_lat) != (orig_lon, orig_lat):
                ax.plot([orig_lon, opt_lon], [orig_lat, opt_lat],
                        color="#2196F3", linestyle="--", linewidth=0.8,
                        alpha=0.6, zorder=3)

            # Add flag
            ab = AnnotationBbox(
                imagebox, (opt_lon, opt_lat),
                frameon=False,  # Remove frame
                pad=0.1,  # Minimal padding
                zorder=4
            )
            ax.add_artist(ab)

            # Add small dot at country centroid
            ax.scatter(orig_lon, orig_lat, color='#1976D2',
                       s=20, zorder=3, alpha=0.7)

        else:
            print(f"❌ Flag not found for {country['name']} ({country['code']})")
    except requests.RequestException:
        print(f"❌ Could not retrieve flag for {country['name']}")

# Customize map appearance
ax.axis('off')
plt.tight_layout(pad=0)
fig = plt.gcf()
plt.show()
fig.savefig('world_map.png', dpi=300, bbox_inches='tight')

