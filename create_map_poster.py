#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, parks, and more.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from networkx import MultiDiGraph
from shapely.geometry import Point
from shapely.geometry import box as shapely_box
from shapely.ops import linemerge, polygonize, unary_union
from tqdm import tqdm

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "out"

FILE_ENCODING = "utf-8"

FONTS = load_fonts()


def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


def is_latin_script(text):
    if not text:
        return True
    latin_count = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            if ord(char) < 0x250:
                latin_count += 1
    if total_alpha == 0:
        return True
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(city, theme_name, output_format):
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            themes.append(file[:-5])
    return themes


def load_theme(theme_name="terracotta"):
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "forest": "#D8D4C0",
            "buildings": "#E0D4C8",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
            "railway": "#6B4F3A",
            "railway_dash": "#F5EDE4",
            "contour_minor": "#D9A08A",
            "contour_major": "#B8653A",
            "admin_boundary": "#C9846A",
            "poi": "#F75E00",
        }
    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


THEME = dict[str, str]()


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0
    custom_cmap = mcolors.ListedColormap(my_colors)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end
    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def create_vignette(ax, color, zorder=10):
    """
    Creates a radial vignette fade darkening all four edges toward the centre.
    More cinematic than top/bottom-only gradients.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx**2 + yy**2)
    alpha = np.clip((radius - 0.4) / 0.8, 0, 1)

    rgb = mcolors.to_rgb(color)
    vignette = np.zeros((size, size, 4))
    vignette[:, :, 0] = rgb[0]
    vignette[:, :, 1] = rgb[1]
    vignette[:, :, 2] = rgb[2]
    vignette[:, :, 3] = alpha

    ax.imshow(
        vignette,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="auto",
        zorder=zorder,
        origin="upper",
        interpolation="bilinear",
    )


def draw_compass_rose(ax, scale_factor, x=0.92, y=0.92, size=0.03):
    """
    Draws a minimal compass rose at a given axes-coordinate position.
    x, y: axes-fraction position (default: top-right)
    size: radius in axes fraction
    """
    color = THEME["text"]
    alpha = 0.75

    # N/S arrow (tall)
    ax.annotate(
        "",
        xy=(x, y + size * 1.6),
        xytext=(x, y - size * 1.6),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0 * scale_factor),
        zorder=12,
        alpha=alpha,
    )
    # E/W tick
    ax.annotate(
        "",
        xy=(x + size, y),
        xytext=(x - size, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8 * scale_factor),
        zorder=12,
        alpha=alpha,
    )
    # N label
    ax.text(
        x,
        y + size * 1.9,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=color,
        alpha=alpha,
        fontsize=6 * scale_factor,
        fontweight="bold",
        zorder=12,
    )


def draw_scale_bar(ax, crop_xlim, scale_factor, units_m=1000):
    """
    Draws a scale bar in the bottom-left corner.
    units_m: length of the bar in metres (default: 1km)
    """
    color = THEME["text"]
    alpha = 0.7

    x_range = crop_xlim[1] - crop_xlim[0]
    bar_fraction = units_m / x_range

    bar_x_start = 0.05
    bar_x_end = bar_x_start + bar_fraction
    bar_y = 0.045

    ax.plot(
        [bar_x_start, bar_x_end],
        [bar_y, bar_y],
        transform=ax.transAxes,
        color=color,
        linewidth=1.5 * scale_factor,
        alpha=alpha,
        solid_capstyle="butt",
        zorder=12,
    )
    # End ticks
    for x in [bar_x_start, bar_x_end]:
        ax.plot(
            [x, x],
            [bar_y - 0.005, bar_y + 0.005],
            transform=ax.transAxes,
            color=color,
            linewidth=1.0 * scale_factor,
            alpha=alpha,
            zorder=12,
        )
    # Label
    label = f"{units_m // 1000} km" if units_m >= 1000 else f"{units_m} m"
    ax.text(
        (bar_x_start + bar_x_end) / 2,
        bar_y + 0.01,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=color,
        alpha=alpha,
        fontsize=5 * scale_factor,
        zorder=12,
    )


def get_edge_colors_by_type(g):
    edge_colors = []
    for _u, _v, data in g.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"
        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME["road_default"]
        edge_colors.append(color)
    return edge_colors


def get_edge_widths_by_type(g):
    edge_widths = []
    for _u, _v, data in g.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"
        if highway in ["motorway", "motorway_link"]:
            width = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            width = 1.0
        elif highway in ["secondary", "secondary_link"]:
            width = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            width = 0.6
        else:
            width = 0.4
        edge_widths.append(width)
    return edge_widths


def get_coordinates(city, country):
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    lat, lon = center_lat_lon
    center = ox.projection.project_geometry(
        Point(lon, lat), crs="EPSG:4326", to_crs=g_proj.graph["crs"]
    )[0]
    center_x, center_y = center.x, center.y
    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height
    half_x = dist
    half_y = dist
    if aspect > 1:
        half_y = half_x / aspect
    else:
        half_x = half_y * aspect
    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def _is_land_polygon(polygon, coastline_geom):
    test_point = polygon.representative_point()
    if coastline_geom.geom_type == "MultiLineString":
        nearest_line = min(coastline_geom.geoms, key=lambda l: l.distance(test_point))
    elif coastline_geom.geom_type == "LineString":
        nearest_line = coastline_geom
    else:
        return False
    param = nearest_line.project(test_point)
    epsilon = 1.0
    p1 = nearest_line.interpolate(max(0, param - epsilon))
    p2 = nearest_line.interpolate(min(nearest_line.length, param + epsilon))
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    nearest_point = nearest_line.interpolate(param)
    cx = test_point.x - nearest_point.x
    cy = test_point.y - nearest_point.y
    cross = dx * cy - dy * cx
    return cross > 0


def build_sea_polygons(coastline_gdf, g_proj, crop_xlim, crop_ylim, center_lat_lon):
    if coastline_gdf is None or coastline_gdf.empty:
        return None
    crs = g_proj.graph["crs"]
    line_mask = coastline_gdf.geometry.type.isin(["LineString", "MultiLineString"])
    coast_lines = coastline_gdf[line_mask]
    if coast_lines.empty:
        return None
    try:
        coast_proj = ox.projection.project_gdf(coast_lines, to_crs=crs)
    except Exception:
        try:
            coast_proj = coast_lines.to_crs(crs)
        except Exception:
            return None
    viewport = shapely_box(crop_xlim[0], crop_ylim[0], crop_xlim[1], crop_ylim[1])
    merged = linemerge(list(coast_proj.geometry))
    clipped = merged.intersection(viewport)
    if clipped.is_empty:
        return None
    combined = unary_union([clipped, viewport.boundary])
    polygons = list(polygonize(combined))
    if not polygons:
        return None
    water_polys = [p for p in polygons if not _is_land_polygon(p, clipped)]
    if not water_polys:
        return None
    return GeoDataFrame(geometry=water_polys, crs=crs)


def fetch_graph(point, dist) -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)
    try:
        g = ox.graph_from_point(
            point,
            dist=dist,
            dist_type="bbox",
            network_type="all",
            truncate_by_edge=True,
        )
        time.sleep(0.5)
        try:
            cache_set(graph, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)
    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    country_label=None,
    name_label=None,
    display_city=None,
    display_country=None,
    poi_dict={},
    fonts=None,
    line_scale=1.0,
    # -------------------------
    # Feature toggles
    # -------------------------
    show_buildings=True,
    show_forest=True,
    show_waterways=True,
    show_railway=True,
    show_admin_boundary=True,
    show_districts=True,
    show_historic=True,
    show_cycle_routes=True,
    show_contours=False,  # Off by default — requires elevation data
    show_compass=True,
    show_scale_bar=True,
    use_vignette=False,  # Off by default — replaces top/bottom gradients
    subtitle_text=None,  # Optional third text line on the poster
):
    """
    Generate a complete map poster with roads, water, parks, and typography.
    """
    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    points_of_interest = None

    # Count fetch steps dynamically based on enabled features
    fetch_steps = 5  # base: graph, water, parks, coastline, railway
    if show_buildings:
        fetch_steps += 1
    if show_forest:
        fetch_steps += 1
    if show_waterways:
        fetch_steps += 1
    if show_admin_boundary:
        fetch_steps += 1
    if show_districts:
        fetch_steps += 1
    if show_historic:
        fetch_steps += 1
    if show_cycle_routes:
        fetch_steps += 1

    with tqdm(
        total=fetch_steps,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4

        # 1. Street network
        pbar.set_description("Downloading street network")
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        # 2. Water polygons
        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        # 3. Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

        # 4. Coastline
        pbar.set_description("Downloading coastline data")
        coastline = fetch_features(
            point,
            compensated_dist,
            tags={"natural": "coastline"},
            name="coastline",
        )
        pbar.update(1)

        # 5. Railway
        pbar.set_description("Downloading railway data")
        railway = fetch_features(
            point,
            compensated_dist,
            tags={
                "railway": [
                    "rail",
                    "subway",
                    "tram",
                    "light_rail",
                    "narrow_gauge",
                    "preserved",
                ]
            },
            name="railway",
        )
        pbar.update(1)

        # 6. Buildings
        buildings = None
        if show_buildings:
            pbar.set_description("Downloading building footprints")
            buildings = fetch_features(
                point,
                compensated_dist,
                tags={"building": True},
                name="buildings",
            )
            pbar.update(1)

        # 7. Forest / woodland
        forest = None
        if show_forest:
            pbar.set_description("Downloading woodland/forests")
            forest = fetch_features(
                point,
                compensated_dist,
                tags={"landuse": ["forest", "wood"], "natural": "wood"},
                name="forest",
            )
            pbar.update(1)

        # 8. Waterway lines (rivers, streams, canals)
        waterways = None
        if show_waterways:
            pbar.set_description("Downloading waterway lines")
            waterways = fetch_features(
                point,
                compensated_dist,
                tags={"waterway": ["river", "stream", "canal", "drain"]},
                name="waterways",
            )
            pbar.update(1)

        # 9. Admin boundaries
        admin_boundary = None
        if show_admin_boundary:
            pbar.set_description("Downloading admin boundaries")
            admin_boundary = fetch_features(
                point,
                compensated_dist,
                tags={"boundary": "administrative", "admin_level": "8"},
                name="admin_boundary",
            )
            pbar.update(1)

        # 10. Districts / neighbourhoods
        districts = None
        if show_districts:
            pbar.set_description("Downloading district labels")
            districts = fetch_features(
                point,
                compensated_dist,
                tags={
                    "place": ["suburb", "neighbourhood", "quarter", "village", "hamlet"]
                },
                name="districts",
            )
            pbar.update(1)

        # 11. Historic sites
        historic = None
        if show_historic:
            pbar.set_description("Downloading historic sites")
            historic = fetch_features(
                point,
                compensated_dist,
                tags={
                    "historic": [
                        "castle",
                        "monument",
                        "archaeological_site",
                        "memorial",
                        "ruins",
                    ]
                },
                name="historic",
            )
            pbar.update(1)

        # 12. Cycle routes
        cycle_routes = None
        if show_cycle_routes:
            pbar.set_description("Downloading cycle routes")
            cycle_routes = fetch_features(
                point,
                compensated_dist,
                tags={"route": "bicycle"},
                name="cycle_routes",
            )
            pbar.update(1)

        # POIs (no progress step — optional add-on)
        if len(poi_dict) > 0 and poi_dict[next(iter(poi_dict))] != []:
            points_of_interest = fetch_features(
                point,
                compensated_dist,
                tags=poi_dict,
                name="_".join(poi_dict[next(iter(poi_dict))]),
            )

    print("✓ All data retrieved successfully!")

    # -------------------------------------------------------------------------
    # Setup figure
    # -------------------------------------------------------------------------
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    g_proj = ox.project_graph(g)
    crs = g_proj.graph["crs"]

    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, compensated_dist)
    sea_polys = build_sea_polygons(coastline, g_proj, crop_xlim, crop_ylim, point)

    scale_factor = min(height, width) / 12.0

    # -------------------------------------------------------------------------
    # Layer 0: Sea / ocean
    # -------------------------------------------------------------------------
    if sea_polys is not None and not sea_polys.empty:
        sea_polys.plot(ax=ax, facecolor=THEME["water"], edgecolor="none", zorder=0.4)

    # -------------------------------------------------------------------------
    # Layer 1: Inland water polygons
    # -------------------------------------------------------------------------
    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(crs)
            water_polys.plot(
                ax=ax, facecolor=THEME["water"], edgecolor="none", zorder=0.5
            )

    # -------------------------------------------------------------------------
    # Layer 1a: Forest / woodland
    # -------------------------------------------------------------------------
    if show_forest and forest is not None and not forest.empty:
        forest_polys = forest[forest.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not forest_polys.empty:
            try:
                forest_polys = ox.projection.project_gdf(forest_polys)
            except Exception:
                forest_polys = forest_polys.to_crs(crs)
            forest_polys.plot(
                ax=ax,
                facecolor=THEME.get("forest", THEME["parks"]),
                edgecolor="none",
                zorder=0.7,
            )

    # -------------------------------------------------------------------------
    # Layer 1b: Parks
    # -------------------------------------------------------------------------
    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(crs)
            parks_polys.plot(
                ax=ax, facecolor=THEME["parks"], edgecolor="none", zorder=0.8
            )

    # -------------------------------------------------------------------------
    # Layer 1c: Building footprints
    # -------------------------------------------------------------------------
    if show_buildings and buildings is not None and not buildings.empty:
        building_polys = buildings[
            buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
        ]
        if not building_polys.empty:
            try:
                building_polys = ox.projection.project_gdf(building_polys)
            except Exception:
                building_polys = building_polys.to_crs(crs)
            building_polys.plot(
                ax=ax,
                facecolor=THEME.get("buildings", THEME["road_residential"]),
                edgecolor="none",
                zorder=0.9,
            )

    # -------------------------------------------------------------------------
    # Layer 1d: Admin boundary (dashed overlay)
    # -------------------------------------------------------------------------
    if show_admin_boundary and admin_boundary is not None and not admin_boundary.empty:
        admin_lines = admin_boundary[
            admin_boundary.geometry.type.isin(
                ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]
            )
        ]
        if not admin_lines.empty:
            try:
                admin_lines = ox.projection.project_gdf(admin_lines)
            except Exception:
                admin_lines = admin_lines.to_crs(crs)
            admin_lines.plot(
                ax=ax,
                facecolor="none",
                edgecolor=THEME.get("admin_boundary", THEME["road_tertiary"]),
                linewidth=0.6 * line_scale,
                linestyle="dashed",
                zorder=1.0,
            )

    # -------------------------------------------------------------------------
    # Layer 2: Roads
    # -------------------------------------------------------------------------
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = [w * line_scale for w in get_edge_widths_by_type(g_proj)]

    ox.plot_graph(
        g_proj,
        ax=ax,
        bgcolor=THEME["bg"],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    # -------------------------------------------------------------------------
    # Layer 2a: Waterway lines (rivers, streams, canals)
    # -------------------------------------------------------------------------
    if show_waterways and waterways is not None and not waterways.empty:
        waterway_lines = waterways[
            waterways.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if not waterway_lines.empty:
            try:
                waterway_lines = ox.projection.project_gdf(waterway_lines)
            except Exception:
                waterway_lines = waterway_lines.to_crs(crs)
            waterway_lines.plot(
                ax=ax,
                color=THEME["water"],
                linewidth=1.0 * line_scale,
                zorder=1.2,
            )

    # -------------------------------------------------------------------------
    # Layer 2b: Cycle routes
    # -------------------------------------------------------------------------
    if show_cycle_routes and cycle_routes is not None and not cycle_routes.empty:
        cycle_lines = cycle_routes[
            cycle_routes.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if not cycle_lines.empty:
            try:
                cycle_lines = ox.projection.project_gdf(cycle_lines)
            except Exception:
                cycle_lines = cycle_lines.to_crs(crs)
            cycle_lines.plot(
                ax=ax,
                color=THEME.get("cycle_route", THEME["parks"]),
                linewidth=0.8 * line_scale,
                linestyle=(0, (4, 2)),
                zorder=1.3,
            )

    # -------------------------------------------------------------------------
    # Layer 2c: Railway lines (solid + sleeper dash)
    # -------------------------------------------------------------------------
    if show_railway and railway is not None and not railway.empty:
        railway_lines = railway[
            railway.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if not railway_lines.empty:
            try:
                railway_lines = ox.projection.project_gdf(railway_lines)
            except Exception:
                railway_lines = railway_lines.to_crs(crs)
            railway_lines.plot(
                ax=ax,
                color=THEME.get("railway", THEME["road_primary"]),
                linewidth=1.2 * line_scale,
                linestyle="solid",
                zorder=1.5,
            )
            railway_lines.plot(
                ax=ax,
                color=THEME.get("railway_dash", THEME["bg"]),
                linewidth=0.4 * line_scale,
                linestyle=(0, (2, 4)),
                zorder=1.6,
            )

    # -------------------------------------------------------------------------
    # Layer 3: Historic site POI dots
    # -------------------------------------------------------------------------
    if show_historic and historic is not None and not historic.empty:
        try:
            historic_proj = ox.projection.project_gdf(historic)
        except Exception:
            historic_proj = historic.to_crs(crs)
        for geom in historic_proj.geometry:
            center = (
                (geom.x, geom.y)
                if geom.geom_type == "Point"
                else (geom.centroid.x, geom.centroid.y)
            )
            ax.add_patch(
                Circle(
                    center,
                    radius=15 * scale_factor,
                    facecolor=THEME.get("historic", THEME["road_secondary"]),
                    edgecolor=THEME.get("bg", "#F5EDE4"),
                    linewidth=0.5 * scale_factor,
                    alpha=0.9,
                    zorder=8,
                )
            )

    # -------------------------------------------------------------------------
    # Layer 3a: POI dots (user-specified)
    # -------------------------------------------------------------------------
    if (
        len(poi_dict) > 0
        and poi_dict[next(iter(poi_dict))] != []
        and points_of_interest is not None
        and not points_of_interest.empty
    ):
        try:
            points_of_interest = ox.projection.project_gdf(points_of_interest)
        except Exception:
            points_of_interest = points_of_interest.to_crs(crs)
        for poi in points_of_interest.geometry:
            center = (
                (poi.x, poi.y)
                if poi.geom_type == "Point"
                else (poi.centroid.x, poi.centroid.y)
            )
            ax.add_patch(
                Circle(
                    center,
                    radius=12 * scale_factor,
                    facecolor=THEME.get("poi", THEME["text"]),
                    edgecolor=THEME.get("poi", THEME["text"]),
                    linewidth=1,
                    alpha=1,
                    zorder=9,
                )
            )

    # -------------------------------------------------------------------------
    # Layer 4: Gradients or vignette
    # -------------------------------------------------------------------------
    if use_vignette:
        create_vignette(ax, THEME["gradient_color"], zorder=10)
    else:
        create_gradient_fade(ax, THEME["gradient_color"], location="bottom", zorder=10)
        create_gradient_fade(ax, THEME["gradient_color"], location="top", zorder=10)

    # -------------------------------------------------------------------------
    # Layer 5: District / neighbourhood labels
    # -------------------------------------------------------------------------
    if show_districts and districts is not None and not districts.empty:
        try:
            districts_proj = ox.projection.project_gdf(districts)
        except Exception:
            districts_proj = districts.to_crs(crs)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for _, row in districts_proj.iterrows():
            name = row.get("name")
            if not name:
                continue
            geom = row.geometry
            cx = geom.x if geom.geom_type == "Point" else geom.centroid.x
            cy = geom.y if geom.geom_type == "Point" else geom.centroid.y
            # Only label points within the current viewport
            if not (xlim[0] < cx < xlim[1] and ylim[0] < cy < ylim[1]):
                continue
            ax.text(
                cx,
                cy,
                name.upper(),
                color=THEME["text"],
                alpha=0.45,
                fontsize=4.5 * scale_factor,
                ha="center",
                va="center",
                fontweight="normal",
                fontstyle="italic",
                zorder=11,
                clip_on=True,
            )

    # -------------------------------------------------------------------------
    # Compass rose & scale bar
    # -------------------------------------------------------------------------
    if show_compass:
        draw_compass_rose(ax, scale_factor)

    if show_scale_bar:
        draw_scale_bar(ax, crop_xlim, scale_factor, units_m=1000)

    # -------------------------------------------------------------------------
    # Typography
    # -------------------------------------------------------------------------
    base_main = 60
    base_sub = 22
    base_coords = 14

    active_fonts = fonts or FONTS
    if active_fonts:
        font_sub = FontProperties(
            fname=active_fonts["light"], size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            fname=active_fonts["regular"], size=base_coords * scale_factor
        )
    else:
        font_sub = FontProperties(
            family="monospace", weight="normal", size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            family="monospace", size=base_coords * scale_factor
        )

    if is_latin_script(display_city):
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        spaced_city = display_city

    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(
            fname=active_fonts["bold"], size=adjusted_font_size
        )
    else:
        font_main_adjusted = FontProperties(
            family="monospace", weight="bold", size=adjusted_font_size
        )

    # City name
    ax.text(
        0.5,
        0.14,
        spaced_city,
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_main_adjusted,
        zorder=12,
    )

    # Country name
    ax.text(
        0.5,
        0.10,
        display_country.upper(),
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_sub,
        zorder=12,
    )

    # Optional subtitle line
    if subtitle_text:
        font_subtitle = FontProperties(
            fname=active_fonts["light"] if active_fonts else None,
            family="monospace" if not active_fonts else None,
            size=base_coords * scale_factor,
        )
        ax.text(
            0.5,
            0.075,
            subtitle_text,
            transform=ax.transAxes,
            color=THEME["text"],
            alpha=0.8,
            ha="center",
            fontproperties=font_subtitle,
            zorder=12,
        )

    # Coordinates
    lat, lon = point
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    coords_y = 0.055 if subtitle_text else 0.07
    ax.text(
        0.5,
        coords_y,
        coords,
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.7,
        ha="center",
        fontproperties=font_coords,
        zorder=12,
    )

    # Divider line
    ax.plot(
        [0.4, 0.6],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=THEME["text"],
        linewidth=1 * scale_factor,
        zorder=12,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"Saving to {output_file}...")
    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05)
    if fmt == "png":
        save_kwargs["dpi"] = 300
    plt.savefig(output_file, format=fmt, **save_kwargs)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  python create_map_poster.py -c "St. Neots" -C "United Kingdom" -t terracotta -d 16000
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000
  python create_map_poster.py --list-themes

Distance guide:
  4000-6000m   Small/dense cities
  8000-12000m  Medium cities
  15000-20000m Large metros
""")


def list_themes():
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get("name", theme_name)
                description = theme_data.get("description", "")
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument(
        "--latitude", "-lat", dest="latitude", type=str, help="Override latitude"
    )
    parser.add_argument(
        "--longitude", "-long", dest="longitude", type=str, help="Override longitude"
    )
    parser.add_argument(
        "--country-label",
        dest="country_label",
        type=str,
        help="Override country text on poster",
    )
    parser.add_argument(
        "--theme", "-t", type=str, default="terracotta", help="Theme name"
    )
    parser.add_argument(
        "--all-themes", "--All-themes", dest="all_themes", action="store_true"
    )
    parser.add_argument(
        "--distance", "-d", type=int, default=18000, help="Map radius in metres"
    )
    parser.add_argument(
        "--width", "-W", type=float, default=12, help="Width in inches (max 20)"
    )
    parser.add_argument(
        "--height", "-H", type=float, default=16, help="Height in inches (max 20)"
    )
    parser.add_argument("--list-themes", action="store_true")
    parser.add_argument("--display-city", "-dc", type=str)
    parser.add_argument("--display-country", "-dC", type=str)
    parser.add_argument("--font-family", type=str)
    parser.add_argument("--line-scale", "-ls", type=float, default=1.0)
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--output", "-o", type=str, default="posters")
    parser.add_argument("--points-of-interest-key", "-poiK", type=str, default="")
    parser.add_argument(
        "--points-of-interest-val", "-poiV", type=str, nargs="+", default=[]
    )
    parser.add_argument(
        "--subtitle", type=str, default=None, help="Optional subtitle line on poster"
    )

    # Feature toggles
    parser.add_argument("--no-buildings", dest="show_buildings", action="store_false")
    parser.add_argument("--no-forest", dest="show_forest", action="store_false")
    parser.add_argument("--no-waterways", dest="show_waterways", action="store_false")
    parser.add_argument("--no-railway", dest="show_railway", action="store_false")
    parser.add_argument(
        "--no-admin-boundary", dest="show_admin_boundary", action="store_false"
    )
    parser.add_argument("--no-districts", dest="show_districts", action="store_false")
    parser.add_argument("--no-historic", dest="show_historic", action="store_false")
    parser.add_argument(
        "--no-cycle-routes", dest="show_cycle_routes", action="store_false"
    )
    parser.add_argument(
        "--vignette",
        dest="use_vignette",
        action="store_true",
        help="Use radial vignette instead of top/bottom gradients",
    )
    parser.add_argument("--no-compass", dest="show_compass", action="store_false")
    parser.add_argument("--no-scale-bar", dest="show_scale_bar", action="store_false")

    parser.set_defaults(
        show_buildings=True,
        show_forest=True,
        show_waterways=True,
        show_railway=True,
        show_admin_boundary=True,
        show_districts=True,
        show_historic=True,
        show_cycle_routes=True,
        use_vignette=False,
        show_compass=True,
        show_scale_bar=True,
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    if args.list_themes:
        list_themes()
        sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    if args.width > 20:
        print(f"⚠ Width clamped to 20.")
        args.width = 20.0
    if args.height > 20:
        print(f"⚠ Height clamped to 20.")
        args.height = 20.0

    if args.output:
        POSTERS_DIR = args.output

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    try:
        if args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = [lat, lon]
            print(f"✓ Coordinates: {', '.join([str(i) for i in coords])}")
        else:
            coords = get_coordinates(args.city, args.country)

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(
                args.city,
                args.country,
                coords,
                args.distance,
                output_file,
                args.format,
                args.width,
                args.height,
                country_label=args.country_label,
                display_city=args.display_city,
                display_country=args.display_country,
                poi_dict={args.points_of_interest_key: args.points_of_interest_val},
                fonts=custom_fonts,
                line_scale=args.line_scale,
                show_buildings=args.show_buildings,
                show_forest=args.show_forest,
                show_waterways=args.show_waterways,
                show_railway=args.show_railway,
                show_admin_boundary=args.show_admin_boundary,
                show_districts=args.show_districts,
                show_historic=args.show_historic,
                show_cycle_routes=args.show_cycle_routes,
                show_compass=args.show_compass,
                show_scale_bar=args.show_scale_bar,
                use_vignette=args.use_vignette,
                subtitle_text=args.subtitle,
            )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
