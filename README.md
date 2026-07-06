# City Map Poster Generator

Generate beautiful, minimalist map posters for any city in the world. Fetches real OpenStreetMap data, applies a colour theme, and exports a print-ready PNG/PDF/SVG.

---

## How to use

### 1. Run a preset

The simplest workflow — pick a preset and run it:

```bash
uv run ./create_map_poster.py --preset presets/london.json
```

Output is saved automatically to the `out/` folder with a timestamped filename.

### 2. Run all presets in one go

```bash
for f in presets/*.json; do uv run ./create_map_poster.py --preset "$f"; done
```

### 3. Export as PDF or SVG instead of PNG

Override the format on the fly:

```bash
uv run ./create_map_poster.py --preset presets/london.json --format pdf
uv run ./create_map_poster.py --preset presets/london.json --format svg
```

### 4. Generate one city in every theme at once

```bash
uv run ./create_map_poster.py --preset presets/london.json --all-themes
```

---

## Available presets

| Preset | City | Theme | Highlights |
|--------|------|-------|------------|
| `presets/london.json` | London, UK | Noir | Road glow, paper texture, historic sites, compass |
| `presets/paris.json` | Paris, France | Pastel Dream | Double border, district labels, text at top |
| `presets/tokyo.json` | Tokyo, Japan | Neon Cyberpunk | Max-intensity road glow, buildings, railway |
| `presets/new_york.json` | New York, USA | Midnight Blue | Directional roads (shows the Manhattan grid), glow |
| `presets/amsterdam.json` | Amsterdam, Netherlands | Ocean | Cycle routes, waterways, double border |
| `presets/reykjavik.json` | Reykjavik, Iceland | Ocean | Directional roads, double border, text at top |
| `presets/st_neots.json` | St. Neots, UK | Terracotta | Road casing, DMS coordinates |
| `presets/barcelona_full.json` | Barcelona, Spain | Terracotta | **Every available option** — use as a reference |

---

## Creating your own preset

Copy any existing preset and edit it. The two required fields are `city` and `country` — everything else is optional and will fall back to a sensible default.

```json
{
    "city": "Edinburgh",
    "country": "Scotland",
    "theme": "noir",
    "distance": 12000,
    "display_city": "Edinburgh",
    "display_country": "Scotland",
    "subtitle": "Athens of the North",
    "border": true,
    "border_style": "double",
    "road_glow": true,
    "road_glow_intensity": 0.6,
    "dpi": 300
}
```

See [`presets/barcelona_full.json`](presets/barcelona_full.json) for a preset that lists every available option with explanatory context.

---

## Visual features at a glance

| Feature | Key in preset | Notes |
|---------|---------------|-------|
| Decorative border | `"border": true` | `"border_style": "single"` or `"double"` |
| Road glow / bloom | `"road_glow": true` | Best on dark themes. `"road_glow_intensity"`: 0–1 |
| Directional road colour | `"directional_roads": true` | Roads coloured by compass bearing — spectacular on grid cities |
| Paper texture | `"paper_texture": true` | `"paper_texture_opacity"`: 0.05–0.15 |
| Road casing | `"road_casing": true` | Adds a darker outline around each road |
| Vignette | `"use_vignette": true` | Radial fade instead of top/bottom gradient |
| CMYK-safe colours | `"cmyk_safe": true` | Desaturates slightly for commercial print |
| Custom route | `"route": [[lat, lon], ...]` or `"route_file"` | Highlight a run, hike, or walk. See [Custom routes and markers](#custom-routes-and-markers) |
| Custom marker | `"marker": [lat, lon]` | Pin a single location with a label. See [Custom routes and markers](#custom-routes-and-markers) |

---

## Custom routes and markers

Two ways to personalise a poster beyond the standard city render: a highlighted route (a run, hike, first-date walk, marathon course) and a single pinned location (your house, where you got engaged, a favourite bar).

### Highlighting a route

Give it either an inline list of `[lat, lon]` points, or a GPX file (track or route points are both supported):

```json
{
    "city": "Boston",
    "country": "USA",
    "theme": "noir",

    "route": [
        [42.3467, -71.0972],
        [42.3398, -71.0892],
        [42.3355, -71.0745]
    ],
    "route_color": "#FF3B30",
    "route_width": 2.5,
    "route_style": "solid",
    "route_glow": true
}
```

Or from the command line, with a GPX file exported from Strava/Garmin/etc.:

```bash
uv run ./create_map_poster.py -c "Boston" -C "USA" -t noir -d 8000 \
    --route-file marathon.gpx --route-glow --route-color "#FF3B30"
```

| Field | CLI flag | Default | Notes |
|-------|----------|---------|-------|
| `route` | — | — | Inline list of `[lat, lon]` points |
| `route_file` | `--route-file` | — | Path to a `.gpx` file; alternative to `route` |
| `route_color` | `--route-color` | theme's `route`/`poi` colour | Any hex colour |
| `route_width` | `--route-width` | `2.5` | Line width |
| `route_style` | `--route-style` | `"solid"` | `"solid"` or `"dashed"` |
| `route_glow` | `--route-glow` | `false` | Adds a soft bloom around the line |

### Pinning a custom marker

```json
{
    "marker": [41.4036, 2.1744],
    "marker_label": "Sagrada Família",
    "marker_color": null,
    "marker_style": "star"
}
```

Or from the command line:

```bash
uv run ./create_map_poster.py -c "Paris" -C "France" -t rose_gold -d 6000 \
    --marker-lat 48.8584 --marker-lon 2.2945 --marker-label "Where we got engaged"
```

| Field | CLI flag | Default | Notes |
|-------|----------|---------|-------|
| `marker` | `--marker-lat` / `--marker-lon` | — | `[lat, lon]` (preset) or two separate floats (CLI) |
| `marker_label` | `--marker-label` | — | Text shown next to the marker |
| `marker_color` | `--marker-color` | theme's `marker`/`poi` colour | Any hex colour |
| `marker_style` | `--marker-style` | `"star"` | `"star"`, `"dot"`, `"diamond"`, or `"pin"` |

See [`presets/barcelona_full.json`](presets/barcelona_full.json) for both in use together.

---

## Available themes

| Theme | Description |
|-------|-------------|
| `terracotta` | Mediterranean warmth — burnt orange and clay on cream |
| `noir` | Pure black with white/grey roads — gallery aesthetic |
| `neon_cyberpunk` | Dark background with electric pink/cyan |
| `midnight_blue` | Deep navy with gold/copper roads — luxury atlas |
| `ocean` | Various blues and teals — perfect for coastal cities |
| `pastel_dream` | Soft muted pastels with dusty blues and mauves |
| `blueprint` | Classic architectural blueprint — technical drawing |
| `japanese_ink` | Minimal ink-wash aesthetic |
| `monochrome_blue` | Single-hue blue tones |
| `warm_beige` | Warm neutral tones |
| `emerald` | Rich greens |
| `forest` | Deep woodland palette |
| `sunset` | Warm oranges and pinks |
| `autumn` | Russet and amber tones |
| `copper_patina` | Aged copper greens and browns |
| `contrast_zones` | High-contrast zone-based colouring |
| `gradient_roads` | Roads rendered as a colour gradient |
| `rose_gold` | Blush pinks and rose gold roads on soft ivory — elegant and romantic |
| `sepia` | Aged paper and burnt umber tones — antique map from a dusty archive |
| `arctic` | Crisp ice whites and polar blues — clean Scandinavian winter clarity |
| `violet_dusk` | Deep midnight purples with lavender roads — twilight hour mystique |
| `sage_linen` | Muted sage greens on natural linen — quiet, earthy, modern minimal |
| `coral` | Warm coral roads and turquoise water on white — vivid tropical coastal energy |

List all themes with descriptions:

```bash
uv run ./create_map_poster.py --list-themes
```

---

## Tips

- **Distance** (`"distance"`) controls the radius in metres around the city centre. Smaller cities need smaller values (5000–8000); major cities work well at 10000–16000.
- **Coordinate override** — set `"latitude"` and `"longitude"` to pin the exact centre of the map rather than relying on geocoding.
- **Fonts** — `"font_family"` controls the city name typeface; `"body_font_family"` controls country/coordinates/subtitle. Both accept any font name installed in the `fonts/` folder.
- **DPI** — use `150` for quick proofing, `300` for standard print, `600` for large-format printing.
- **Paper sizes** — set `width` and `height` in inches to match your print size:

  | Size | Portrait (w × h) | Landscape (w × h) |
  |------|-----------------|-------------------|
  | A5   | `5.83 × 8.27`   | `8.27 × 5.83`     |
  | A4   | `8.27 × 11.69`  | `11.69 × 8.27`    |
  | A3   | `11.69 × 16.54` | `16.54 × 11.69`   |
  | A3+  | `12.95 × 19.02` | `19.02 × 12.95`   |
