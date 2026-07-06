# maptoposter

Generates print-ready map posters for any city by fetching OpenStreetMap data via OSMnx, applying a colour theme, and exporting PNG/PDF/SVG.

## Running

```bash
# Single preset
uv run ./create_map_poster.py --preset presets/london.json

# All presets
for f in presets/*.json; do uv run ./create_map_poster.py --preset "$f"; done

# Override format
uv run ./create_map_poster.py --preset presets/london.json --format pdf

# Render one preset in every theme
uv run ./create_map_poster.py --preset presets/london.json --all-themes

# List all themes
uv run ./create_map_poster.py --list-themes
```

Output goes to `out/` with a timestamped filename.

## Project structure

- `create_map_poster.py` — main entrypoint; all rendering logic lives here
- `font_management.py` — font loading helper imported by the main script
- `presets/` — JSON files that define a city render (see below)
- `themes/` — JSON files that define colour palettes
- `fonts/` — font files; any font installed here can be used via `font_family`/`body_font_family`
- `cache/` — OSMnx data cached as `.pkl` files to avoid repeated API calls (set `CACHE_DIR` env var to override)
- `out/` — rendered output images

## Presets

The two required fields are `city` and `country`. Everything else is optional.

`presets/barcelona_full.json` lists every available option and is the canonical reference.

Key preset fields:

| Field | Default | Notes |
|---|---|---|
| `city`, `country` | required | Used for geocoding |
| `display_city`, `display_country` | city/country | Text shown on the poster |
| `subtitle`, `edition` | — | Extra text lines on the poster |
| `theme` | `"terracotta"` | Must match a file in `themes/` |
| `distance` | — | Radius in metres around city centre |
| `latitude`, `longitude` | geocoded | Override the map centre |
| `width`, `height` | — | Figure size in inches |
| `dpi` | 300 | 150 for proofing, 300 for print, 600 for large-format |
| `format` | `"png"` | `png`, `pdf`, or `svg` |
| `font_family` | — | City name font (must be in `fonts/`) |
| `body_font_family` | — | Country/subtitle font |
| `text_position` | `"bottom"` | `"top"` or `"bottom"` |
| `coord_format` | — | `"dms"` or `"decimal"` |
| `border` | false | `border_style`: `"single"` or `"double"` |
| `road_glow` | false | `road_glow_intensity`: 0–1; best on dark themes |
| `road_casing` | false | Darker outline around roads |
| `directional_roads` | false | Colour roads by compass bearing |
| `paper_texture` | false | `paper_texture_opacity`: 0.05–0.15 |
| `use_vignette` | false | Radial fade instead of gradient |
| `gradient_intensity` | 1.0 | Strength of the top/bottom gradient fade, 0–1; lower it if the fade feels too heavy |
| `cmyk_safe` | false | Desaturates for commercial print |
| `line_scale` | 1.0 | Scale all line widths |
| `show_buildings` | false | |
| `show_forest` | false | |
| `show_waterways` | false | |
| `show_railway` | false | |
| `show_admin_boundary` | false | |
| `show_districts` | false | |
| `show_historic` | false | |
| `show_cycle_routes` | false | |
| `show_compass` | false | |
| `show_scale_bar` | false | |
| `route` | — | List of `[lat, lon]` pairs to highlight as a route (a run, hike, walk) |
| `route_file` | — | GPX file path, alternative to inline `route` |
| `route_color` | theme's `route`/`poi` | Override hex colour for the route line |
| `route_width` | 2.5 | Route line width |
| `route_style` | `"solid"` | `"solid"` or `"dashed"` |
| `route_glow` | false | Soft bloom around the route line |
| `marker` | — | `[lat, lon]` pin for a single custom location (e.g. "our house") |
| `marker_label` | — | Text shown next to the marker |
| `marker_color` | theme's `marker`/`poi` | Override hex colour for the marker |
| `marker_style` | `"star"` | `"star"`, `"dot"`, `"diamond"`, or `"pin"` |

## Themes

Themes are JSON files in `themes/`. Available: `terracotta`, `noir`, `neon_cyberpunk`, `midnight_blue`, `ocean`, `pastel_dream`, `blueprint`, `japanese_ink`, `monochrome_blue`, `warm_beige`, `emerald`, `forest`, `sunset`, `autumn`, `copper_patina`, `contrast_zones`, `gradient_roads`, `rose_gold`, `sepia`, `arctic`, `violet_dusk`, `sage_linen`, `coral`.

## Dependencies

Managed with `uv` via `pyproject.toml`. Python >= 3.11 required. Key libraries: `osmnx`, `matplotlib`, `geopandas`, `shapely`, `geopy`.
