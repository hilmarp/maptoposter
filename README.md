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
