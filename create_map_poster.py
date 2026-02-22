#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, parks, and more.
"""

import argparse
import asyncio
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def format_coords_dms(lat: float, lon: float) -> str:
    def to_dms(value: float, pos: str, neg: str) -> str:
        direction = pos if value >= 0 else neg
        value = abs(value)
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = round((value - degrees - minutes / 60) * 3600)
        return f"{degrees}°{minutes:02d}′{seconds:02d}″{direction}"

    return f"{to_dms(lat, 'N', 'S')} / {to_dms(lon, 'E', 'W')}"


def format_coords_decimal(lat: float, lon: float) -> str:
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")
    return coords


def rgb_to_cmyk_safe(hex_color: str) -> str:
    rgb = mcolors.to_rgb(hex_color)
    r, g, b = rgb
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    factor = 0.85
    r = r * factor + grey * (1 - factor)
    g = g * factor + grey * (1 - factor)
    b = b * factor + grey * (1 - factor)
    max_val = max(r, g, b)
    if max_val > 0.92:
        scale = 0.92 / max_val
        r, g, b = r * scale, g * scale, b * scale
    return mcolors.to_hex((r, g, b))


def apply_cmyk_safe(theme: dict) -> dict:
    safe = {}
    for k, v in theme.items():
        if isinstance(v, str) and v.startswith("#"):
            try:
                safe[k] = rgb_to_cmyk_safe(v)
            except ValueError:
                safe[k] = v
        else:
            safe[k] = v
    return safe


def generate_output_filename(
    city: str, theme_name: str, output_format: str, output_dir: str | None = None
) -> str:
    out_dir = output_dir or POSTERS_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    return os.path.join(out_dir, f"{city_slug}_{theme_name}_{timestamp}.{ext}")


def get_available_themes() -> list[str]:
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    return [f[:-5] for f in sorted(os.listdir(THEMES_DIR)) if f.endswith(".json")]


def load_theme(theme_name: str = "terracotta") -> dict:
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
            "parks": "#D8E8D0",
            "forest": "#C8D8B8",
            "buildings": "#DDD0C4",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
            "railway": "#6B4F3A",
            "railway_dash": "#F5EDE4",
            "cycle_route": "#8B9E6A",
            "admin_boundary": "#C9846A",
            "historic": "#A0522D",
            "poi": "#F75E00",
        }
    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


def load_preset(preset_path: str) -> dict:
    with open(preset_path, "r", encoding=FILE_ENCODING) as f:
        return json.load(f)


def load_font_pair(
    font_family: str | None,
    body_font_family: str | None,
) -> tuple[dict | None, dict | None]:
    """
    Load display and body font dicts.
    Returns (display_fonts, body_fonts) — either may be None if loading fails.
    body_fonts falls back to display_fonts if no separate body family is given.
    """
    display_fonts: dict | None = None
    body_fonts: dict | None = None

    if font_family:
        display_fonts = load_fonts(font_family)
        if not display_fonts:
            print(
                f"⚠ Failed to load display font '{font_family}', falling back to Roboto"
            )

    if body_font_family:
        body_fonts = load_fonts(body_font_family)
        if not body_fonts:
            print(
                f"⚠ Failed to load body font '{body_font_family}', falling back to display font"
            )
            body_fonts = None  # will fall back to display_fonts in create_poster

    return display_fonts, body_fonts


THEME: dict = {}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def create_gradient_fade(
    ax, color: str, location: str = "bottom", zorder: float = 10
) -> None:
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]
    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start, extent_y_end = 0, 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start, extent_y_end = 0.75, 1.0
    custom_cmap = mcolors.ListedColormap(my_colors)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    ax.imshow(
        gradient,
        extent=[
            xlim[0],
            xlim[1],
            ylim[0] + y_range * extent_y_start,
            ylim[0] + y_range * extent_y_end,
        ],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def create_vignette(ax, color: str, zorder: float = 10) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
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


def draw_compass_rose(
    ax, scale_factor: float, x: float = 0.92, y: float = 0.92, size: float = 0.03
) -> None:
    color = THEME["text"]
    alpha = 0.75
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


def draw_scale_bar(
    ax, crop_xlim: tuple, scale_factor: float, units_m: int = 1000
) -> None:
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


def apply_road_casing(ax, g_proj, edge_widths: list[float], line_scale: float) -> None:
    casing_colors = []
    for _u, _v, data in g_proj.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"
        if highway in ["motorway", "motorway_link"]:
            base = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            base = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            base = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            base = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            base = THEME["road_residential"]
        else:
            base = THEME["road_default"]
        rgb = mcolors.to_rgb(base)
        darker = tuple(max(0.0, c * 0.75) for c in rgb)
        casing_colors.append(mcolors.to_hex(darker))
    casing_widths = [w * 1.4 for w in edge_widths]
    ox.plot_graph(
        g_proj,
        ax=ax,
        bgcolor=THEME["bg"],
        node_size=0,
        edge_color=casing_colors,
        edge_linewidth=casing_widths,
        show=False,
        close=False,
    )


def apply_paper_texture(ax, crop_xlim, crop_ylim, opacity=0.07):
    rng = np.random.default_rng(seed=42)
    size = 2048  # much larger — closer to actual print resolution
    noise = rng.standard_normal((size, size))

    # Add a second coarser noise layer for visible grain
    coarse = rng.standard_normal((size // 8, size // 8))
    coarse = np.kron(coarse, np.ones((8, 8)))  # upscale without smoothing

    combined = noise * 0.6 + coarse * 0.4
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    texture = np.ones((size, size, 4))
    texture[:, :, 3] = combined * opacity

    ax.imshow(
        texture,
        extent=[crop_xlim[0], crop_xlim[1], crop_ylim[0], crop_ylim[1]],
        aspect="auto",
        zorder=10.5,
        interpolation="nearest",  # don't smooth — keep grain sharp
        origin="upper",
    )


# ---------------------------------------------------------------------------
# Graph / feature helpers
# ---------------------------------------------------------------------------


def get_edge_colors_by_type(g) -> list[str]:
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


def get_edge_widths_by_type(g) -> list[float]:
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


def get_coordinates(city: str, country: str) -> tuple[float, float]:
    coords_key = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords_key)
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
        print(
            f"✓ Found: {addr}" if addr else "✓ Found location (address not available)"
        )
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords_key, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_crop_limits(g_proj, center_lat_lon: tuple, fig, dist: float) -> tuple:
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


def _is_land_polygon(polygon, coastline_geom) -> bool:
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
    dx, dy = p2.x - p1.x, p2.y - p1.y
    nearest_point = nearest_line.interpolate(param)
    cx, cy = test_point.x - nearest_point.x, test_point.y - nearest_point.y
    return (dx * cy - dy * cx) > 0


def build_sea_polygons(
    coastline_gdf, g_proj, crop_xlim: tuple, crop_ylim: tuple, center_lat_lon: tuple
) -> GeoDataFrame | None:
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


def fetch_graph(point: tuple, dist: float) -> MultiDiGraph | None:
    lat, lon = point
    graph_key = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph_key)
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
            cache_set(graph_key, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(
    point: tuple, dist: float, tags: dict, name: str
) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    feat_key = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(feat_key)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)
    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(feat_key, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def get_waterway_width(row, line_scale: float) -> float:
    osm_width = row.get("width") if hasattr(row, "get") else None
    waterway_type = row.get("waterway") if hasattr(row, "get") else None
    if osm_width:
        try:
            w = float(str(osm_width).replace("m", "").strip())
            return max(0.3, min(w / 10.0, 4.0)) * line_scale
        except ValueError:
            pass
    type_widths = {"river": 1.8, "canal": 1.4, "stream": 0.6, "drain": 0.4}
    return type_widths.get(waterway_type, 0.6) * line_scale


# ---------------------------------------------------------------------------
# Main poster generation
# ---------------------------------------------------------------------------


def create_poster(
    city: str,
    country: str,
    point: tuple,
    dist: int,
    output_file: str,
    output_format: str,
    width: float = 12,
    height: float = 16,
    country_label: str | None = None,
    name_label: str | None = None,
    display_city: str | None = None,
    display_country: str | None = None,
    poi_dict: dict | None = None,
    fonts: dict | None = None,
    body_fonts: dict | None = None,
    line_scale: float = 1.0,
    # Feature toggles
    show_buildings: bool = True,
    show_forest: bool = True,
    show_waterways: bool = True,
    show_railway: bool = True,
    show_admin_boundary: bool = True,
    show_districts: bool = True,
    show_historic: bool = True,
    show_cycle_routes: bool = True,
    show_compass: bool = True,
    show_scale_bar: bool = True,
    use_vignette: bool = False,
    road_casing: bool = False,
    paper_texture: bool = False,
    paper_texture_opacity: float = 0.07,
    cmyk_safe: bool = False,
    text_position: str = "bottom",
    coord_format: str = "decimal",
    subtitle_text: str | None = None,
    edition_text: str | None = None,
    dpi: int = 300,
) -> None:
    if poi_dict is None:
        poi_dict = {}

    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    points_of_interest = None

    fetch_steps = 5
    for flag in [
        show_buildings,
        show_forest,
        show_waterways,
        show_admin_boundary,
        show_districts,
        show_historic,
        show_cycle_routes,
    ]:
        if flag:
            fetch_steps += 1

    with tqdm(
        total=fetch_steps,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4

        pbar.set_description("Downloading street network")
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

        pbar.set_description("Downloading coastline data")
        coastline = fetch_features(
            point, compensated_dist, tags={"natural": "coastline"}, name="coastline"
        )
        pbar.update(1)

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

        buildings: GeoDataFrame | None = None
        if show_buildings:
            pbar.set_description("Downloading building footprints")
            buildings = fetch_features(
                point, compensated_dist, tags={"building": True}, name="buildings"
            )
            pbar.update(1)

        forest: GeoDataFrame | None = None
        if show_forest:
            pbar.set_description("Downloading woodland/forests")
            forest = fetch_features(
                point,
                compensated_dist,
                tags={"landuse": ["forest", "wood"], "natural": "wood"},
                name="forest",
            )
            pbar.update(1)

        waterways: GeoDataFrame | None = None
        if show_waterways:
            pbar.set_description("Downloading waterway lines")
            waterways = fetch_features(
                point,
                compensated_dist,
                tags={"waterway": ["river", "stream", "canal", "drain"]},
                name="waterways",
            )
            pbar.update(1)

        admin_boundary: GeoDataFrame | None = None
        if show_admin_boundary:
            pbar.set_description("Downloading admin boundaries")
            admin_boundary = fetch_features(
                point,
                compensated_dist,
                tags={"boundary": "administrative", "admin_level": "8"},
                name="admin_boundary",
            )
            pbar.update(1)

        districts: GeoDataFrame | None = None
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

        historic: GeoDataFrame | None = None
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

        cycle_routes: GeoDataFrame | None = None
        if show_cycle_routes:
            pbar.set_description("Downloading cycle routes")
            cycle_routes = fetch_features(
                point, compensated_dist, tags={"route": "bicycle"}, name="cycle_routes"
            )
            pbar.update(1)

        if poi_dict and poi_dict.get(next(iter(poi_dict), ""), []) != []:
            points_of_interest = fetch_features(
                point,
                compensated_dist,
                tags=poi_dict,
                name="_".join(poi_dict[next(iter(poi_dict))]),
            )

    print("✓ All data retrieved successfully!")

    global THEME
    if cmyk_safe:
        print("Applying CMYK-safe colour adjustments...")
        THEME = apply_cmyk_safe(THEME)

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
    # Layers
    # -------------------------------------------------------------------------

    # Layer 0: Sea
    if sea_polys is not None and not sea_polys.empty:
        sea_polys.plot(ax=ax, facecolor=THEME["water"], edgecolor="none", zorder=0.4)

    # Layer 1: Inland water polygons
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

    # Layer 1a: Forest
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

    # Layer 1b: Parks
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

    # Layer 1c: Buildings
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

    # Layer 1d: Admin boundary
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

    # Layer 2: Road casing pass (optional)
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = [w * line_scale for w in get_edge_widths_by_type(g_proj)]

    if road_casing:
        apply_road_casing(ax, g_proj, edge_widths, line_scale)

    # Layer 2: Roads
    print("Applying road hierarchy colors...")
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

    # Layer 2a: Waterway lines
    if show_waterways and waterways is not None and not waterways.empty:
        waterway_lines = waterways[
            waterways.geometry.type.isin(["LineString", "MultiLineString"])
        ]
        if not waterway_lines.empty:
            try:
                waterway_lines = ox.projection.project_gdf(waterway_lines)
            except Exception:
                waterway_lines = waterway_lines.to_crs(crs)
            for _, row in waterway_lines.iterrows():
                try:
                    lw = get_waterway_width(row, line_scale)
                    GeoDataFrame([row], crs=waterway_lines.crs).plot(
                        ax=ax, color=THEME["water"], linewidth=lw, zorder=1.2
                    )
                except Exception:
                    pass

    # Layer 2b: Cycle routes
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

    # Layer 2c: Railway
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

    # Layer 3: Historic dots
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
                    edgecolor=THEME["bg"],
                    linewidth=0.5 * scale_factor,
                    alpha=0.9,
                    zorder=8,
                )
            )

    # Layer 3a: User POI dots
    if (
        poi_dict
        and poi_dict.get(next(iter(poi_dict), ""), []) != []
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

    # Layer 4: Vignette or gradients
    if use_vignette:
        create_vignette(ax, THEME["gradient_color"], zorder=10)
    else:
        create_gradient_fade(ax, THEME["gradient_color"], location="bottom", zorder=10)
        create_gradient_fade(ax, THEME["gradient_color"], location="top", zorder=10)

    # Layer 4.5: Paper texture
    if paper_texture:
        apply_paper_texture(ax, crop_xlim, crop_ylim, opacity=paper_texture_opacity)

    # Layer 5: District labels
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

    if show_compass:
        draw_compass_rose(ax, scale_factor)
    if show_scale_bar:
        draw_scale_bar(ax, crop_xlim, scale_factor, units_m=1000)

    # -------------------------------------------------------------------------
    # Typography
    # -------------------------------------------------------------------------
    if text_position != "none":
        base_main = 60
        base_sub = 22
        base_coords = 14

        # Display font — city name only
        active_display_fonts = fonts or FONTS
        # Body font — country, coords, subtitle, edition
        # Falls back to display font if no separate body font was provided
        active_body_fonts = body_fonts or active_display_fonts

        if active_body_fonts:
            font_sub = FontProperties(
                fname=active_body_fonts["light"], size=base_sub * scale_factor
            )
            font_coords = FontProperties(
                fname=active_body_fonts["regular"], size=base_coords * scale_factor
            )
            font_light = FontProperties(
                fname=active_body_fonts["light"], size=base_coords * scale_factor
            )
            font_small = FontProperties(
                fname=active_body_fonts["light"], size=6 * scale_factor
            )
        else:
            font_sub = FontProperties(family="monospace", size=base_sub * scale_factor)
            font_coords = FontProperties(
                family="monospace", size=base_coords * scale_factor
            )
            font_light = FontProperties(
                family="monospace", size=base_coords * scale_factor
            )
            font_small = FontProperties(family="monospace", size=6 * scale_factor)

        spaced_city = (
            "  ".join(list(display_city.upper()))
            if is_latin_script(display_city)
            else display_city
        )

        base_adjusted_main = base_main * scale_factor
        city_char_count = len(display_city)
        if city_char_count > 10:
            length_factor = 10 / city_char_count
            adjusted_font_size = max(
                base_adjusted_main * length_factor, 10 * scale_factor
            )
        else:
            adjusted_font_size = base_adjusted_main

        if active_display_fonts:
            font_main_adjusted = FontProperties(
                fname=active_display_fonts["bold"], size=adjusted_font_size
            )
        else:
            font_main_adjusted = FontProperties(
                family="monospace", weight="bold", size=adjusted_font_size
            )

        if text_position == "bottom":
            y_city = 0.14
            y_country = 0.10
            y_subtitle = 0.075
            y_coords = 0.055 if subtitle_text else 0.07
            y_divider = 0.125
        else:  # top
            y_city = 0.88
            y_country = 0.84
            y_subtitle = 0.815
            y_coords = 0.795 if subtitle_text else 0.81
            y_divider = 0.875

        coords_str = (
            format_coords_dms(point[0], point[1])
            if coord_format == "dms"
            else format_coords_decimal(point[0], point[1])
        )

        ax.text(
            0.5,
            y_city,
            spaced_city,
            transform=ax.transAxes,
            color=THEME["text"],
            ha="center",
            fontproperties=font_main_adjusted,
            zorder=12,
        )

        ax.text(
            0.5,
            y_country,
            display_country.upper(),
            transform=ax.transAxes,
            color=THEME["text"],
            ha="center",
            fontproperties=font_sub,
            zorder=12,
        )

        if subtitle_text:
            ax.text(
                0.5,
                y_subtitle,
                subtitle_text,
                transform=ax.transAxes,
                color=THEME["text"],
                alpha=0.8,
                ha="center",
                fontproperties=font_light,
                zorder=12,
            )

        ax.text(
            0.5,
            y_coords,
            coords_str,
            transform=ax.transAxes,
            color=THEME["text"],
            alpha=0.7,
            ha="center",
            fontproperties=font_coords,
            zorder=12,
        )

        ax.plot(
            [0.4, 0.6],
            [y_divider, y_divider],
            transform=ax.transAxes,
            color=THEME["text"],
            linewidth=1 * scale_factor,
            zorder=12,
        )

        if edition_text:
            ax.text(
                0.97,
                0.025,
                edition_text,
                transform=ax.transAxes,
                color=THEME["text"],
                alpha=0.6,
                ha="right",
                va="bottom",
                fontproperties=font_small,
                zorder=12,
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"Saving to {output_file}...")
    fmt = output_format.lower()
    save_kwargs: dict = dict(
        facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05
    )
    if fmt == "png":
        save_kwargs["dpi"] = dpi
    plt.savefig(output_file, format=fmt, **save_kwargs)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def run_batch(
    batch_file: str,
    global_args: argparse.Namespace,
    custom_fonts: dict | None = None,
    custom_body_fonts: dict | None = None,
) -> None:
    print(f"\nRunning batch from {batch_file}...")
    with open(batch_file, "r", encoding=FILE_ENCODING) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} entries.\n")

    for i, row in enumerate(rows, 1):
        city = row.get("city", "").strip()
        country = row.get("country", "").strip()
        if not city or not country:
            print(f"⚠ Row {i}: missing city or country, skipping.")
            continue

        theme_name = row.get("theme", global_args.theme).strip()
        dist = int(row.get("distance", global_args.distance))
        display_city = row.get("display_city", "").strip() or None
        display_country = row.get("display_country", "").strip() or None
        subtitle = row.get("subtitle", "").strip() or None
        edition = row.get("edition", "").strip() or None

        print(f"\n[{i}/{len(rows)}] {city}, {country} — theme: {theme_name}")

        available_themes = get_available_themes()
        if theme_name not in available_themes:
            print(f"  ⚠ Theme '{theme_name}' not found, skipping.")
            continue

        global THEME
        THEME = load_theme(theme_name)
        output_file = generate_output_filename(
            city, theme_name, global_args.format, global_args.output
        )

        try:
            coords = get_coordinates(city, country)
            create_poster(
                city=city,
                country=country,
                point=coords,
                dist=dist,
                output_file=output_file,
                output_format=global_args.format,
                width=global_args.width,
                height=global_args.height,
                display_city=display_city,
                display_country=display_country,
                fonts=custom_fonts,
                body_fonts=custom_body_fonts,
                line_scale=global_args.line_scale,
                show_buildings=global_args.show_buildings,
                show_forest=global_args.show_forest,
                show_waterways=global_args.show_waterways,
                show_railway=global_args.show_railway,
                show_admin_boundary=global_args.show_admin_boundary,
                show_districts=global_args.show_districts,
                show_historic=global_args.show_historic,
                show_cycle_routes=global_args.show_cycle_routes,
                show_compass=global_args.show_compass,
                show_scale_bar=global_args.show_scale_bar,
                use_vignette=global_args.use_vignette,
                road_casing=global_args.road_casing,
                paper_texture=global_args.paper_texture,
                paper_texture_opacity=global_args.paper_texture_opacity,
                cmyk_safe=global_args.cmyk_safe,
                text_position=global_args.text_position,
                coord_format=global_args.coord_format,
                subtitle_text=subtitle,
                edition_text=edition,
                dpi=global_args.dpi,
            )
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("✓ Batch complete!")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def print_examples() -> None:
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  python create_map_poster.py -c "St. Neots" -C "United Kingdom" -t terracotta -d 16000
  python create_map_poster.py -c "St. Neots" -C "United Kingdom" -t terracotta -d 16000 \\
      --font-family "Playfair Display" --body-font-family "Raleway" \\
      --road-casing --paper-texture --coord-format dms --subtitle "Est. 917 AD"
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000 --road-casing
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000 --text-position top
  python create_map_poster.py --batch cities.csv -t terracotta
  python create_map_poster.py --preset my_preset.json
  python create_map_poster.py --list-themes

Batch CSV format (cities.csv):
  city,country,theme,distance,display_city,display_country,subtitle,edition
  St. Neots,United Kingdom,terracotta,16000,,,Est. 917 AD,No. 1 of 10
  London,United Kingdom,noir,15000,,,The Capital,
""")


def list_themes() -> None:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    parser.add_argument("--city", "-c", type=str)
    parser.add_argument("--country", "-C", type=str)
    parser.add_argument("--latitude", "-lat", dest="latitude", type=str)
    parser.add_argument("--longitude", "-long", dest="longitude", type=str)
    parser.add_argument("--country-label", dest="country_label", type=str)
    parser.add_argument("--theme", "-t", type=str, default="terracotta")
    parser.add_argument(
        "--all-themes", "--All-themes", dest="all_themes", action="store_true"
    )
    parser.add_argument("--distance", "-d", type=int, default=18000)
    parser.add_argument("--width", "-W", type=float, default=12)
    parser.add_argument("--height", "-H", type=float, default=16)
    parser.add_argument("--list-themes", action="store_true")
    parser.add_argument("--display-city", "-dc", type=str)
    parser.add_argument("--display-country", "-dC", type=str)
    parser.add_argument(
        "--font-family",
        type=str,
        default=None,
        help='Display font for city name (e.g. "Playfair Display")',
    )
    parser.add_argument(
        "--body-font-family",
        type=str,
        default=None,
        help='Body font for country/coords/subtitle (e.g. "Raleway"). Defaults to --font-family.',
    )
    parser.add_argument("--line-scale", "-ls", type=float, default=1.0)
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--output", "-o", type=str, default="out")
    parser.add_argument("--points-of-interest-key", "-poiK", type=str, default="")
    parser.add_argument(
        "--points-of-interest-val", "-poiV", type=str, nargs="+", default=[]
    )

    # Style / layout
    parser.add_argument("--subtitle", type=str, default=None)
    parser.add_argument(
        "--edition",
        type=str,
        default=None,
        help='Edition text in bottom-right corner (e.g. "No. 1 of 10")',
    )
    parser.add_argument(
        "--text-position", type=str, default="bottom", choices=["top", "bottom", "none"]
    )
    parser.add_argument(
        "--coord-format", type=str, default="decimal", choices=["decimal", "dms"]
    )
    parser.add_argument("--dpi", type=int, default=300, choices=[72, 150, 300, 600])
    parser.add_argument("--road-casing", dest="road_casing", action="store_true")
    parser.add_argument("--paper-texture", dest="paper_texture", action="store_true")
    parser.add_argument(
        "--paper-texture-opacity",
        dest="paper_texture_opacity",
        type=float,
        default=0.07,
        help="Paper texture opacity (default: 0.07 for print, try 0.15 for screen proofing)",
    )
    parser.add_argument("--cmyk-safe", dest="cmyk_safe", action="store_true")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--batch", type=str, default=None)

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
    parser.add_argument("--vignette", dest="use_vignette", action="store_true")
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
        road_casing=False,
        paper_texture=False,
        cmyk_safe=False,
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Preset overrides all args
    if args.preset:
        preset = load_preset(args.preset)
        for k, v in preset.items():
            setattr(args, k.replace("-", "_"), v)

    # Load fonts once, before batch or single run
    custom_fonts, custom_body_fonts = load_font_pair(
        getattr(args, "font_family", None),
        getattr(args, "body_font_family", None),
    )

    # Batch mode
    if args.batch:
        run_batch(
            args.batch,
            args,
            custom_fonts=custom_fonts,
            custom_body_fonts=custom_body_fonts,
        )
        sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    if args.width > 20:
        print("⚠ Width clamped to 20.")
        args.width = 20.0
    if args.height > 20:
        print("⚠ Height clamped to 20.")
        args.height = 20.0

    if args.output:
        POSTERS_DIR = args.output

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    themes_to_generate = available_themes if args.all_themes else [args.theme]
    if not args.all_themes and args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        sys.exit(1)

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    try:
        if args.latitude and args.longitude:
            coords = (parse(args.latitude), parse(args.longitude))
            print(f"✓ Coordinates: {coords[0]}, {coords[1]}")
        else:
            coords = get_coordinates(args.city, args.country)

        poi_dict: dict = (
            {args.points_of_interest_key: args.points_of_interest_val}
            if args.points_of_interest_key
            else {}
        )

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(
                args.city, theme_name, args.format, args.output
            )
            create_poster(
                city=args.city,
                country=args.country,
                point=coords,
                dist=args.distance,
                output_file=output_file,
                output_format=args.format,
                width=args.width,
                height=args.height,
                country_label=args.country_label,
                display_city=args.display_city,
                display_country=args.display_country,
                poi_dict=poi_dict,
                fonts=custom_fonts,
                body_fonts=custom_body_fonts,
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
                road_casing=args.road_casing,
                paper_texture=args.paper_texture,
                paper_texture_opacity=args.paper_texture_opacity,
                cmyk_safe=args.cmyk_safe,
                text_position=args.text_position,
                coord_format=args.coord_format,
                subtitle_text=args.subtitle,
                edition_text=args.edition,
                dpi=args.dpi,
            )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
