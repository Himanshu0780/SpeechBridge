"""
app/map_visualization.py
--------------------------
Interactive India language map using Plotly.

Fixes vs original:
- Removed scope="asia"  — conflicts with center/lataxis/lonaxis in Plotly 5.x,
  causing a blank render. Use scope="world" + explicit lat/lon ranges instead.
- Changed projection to "mercator" — most reliable for regional maps.
- Replaced transparent paper_bgcolor with solid dark colour — transparent was
  causing the map to disappear on some Streamlit/browser combos.
- Removed unused `import plotly.express as px`.
- Moved LANGUAGE_INFO into this file (was duplicated in main.py).
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Colours ───────────────────────────────────────────────────────────────────
SOURCE_COLOR  = "#FF6B35"
TARGET_COLOR  = "#4ECDC4"
OVERLAP_COLOR = "#9B59B6"
NEUTRAL_COLOR = "rgba(80, 100, 160, 0.4)"

# ── Language → states ─────────────────────────────────────────────────────────
LANGUAGE_STATES: Dict[str, list] = {
    "Hindi":     ["Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan",
                  "Uttarakhand", "Himachal Pradesh", "Jharkhand", "Chhattisgarh",
                  "Haryana", "Delhi"],
    "Bengali":   ["West Bengal", "Tripura"],
    "Tamil":     ["Tamil Nadu", "Puducherry"],
    "Telugu":    ["Andhra Pradesh", "Telangana"],
    "Marathi":   ["Maharashtra"],
    "Gujarati":  ["Gujarat", "Dadra and Nagar Haveli"],
    "Kannada":   ["Karnataka"],
    "Malayalam": ["Kerala", "Lakshadweep"],
    "Punjabi":   ["Punjab"],
    "Odia":      ["Odisha"],
    "Assamese":  ["Assam"],
    "English":   ["All States"],
    "Urdu":      ["Jammu and Kashmir", "Uttar Pradesh", "Telangana"],
}

# ── State centroids (lat, lon) ────────────────────────────────────────────────
STATE_CENTROIDS: Dict[str, tuple] = {
    "Uttar Pradesh":          (26.8, 80.9),
    "Bihar":                  (25.1, 85.3),
    "Madhya Pradesh":         (23.5, 77.5),
    "Rajasthan":              (27.0, 74.2),
    "Uttarakhand":            (30.1, 79.3),
    "Himachal Pradesh":       (31.9, 77.1),
    "Jharkhand":              (23.6, 85.3),
    "Chhattisgarh":           (21.3, 81.9),
    "Haryana":                (29.1, 76.1),
    "Delhi":                  (28.7, 77.1),
    "West Bengal":            (22.9, 87.9),
    "Tripura":                (23.9, 91.7),
    "Tamil Nadu":             (11.1, 78.7),
    "Puducherry":             (11.9, 79.8),
    "Andhra Pradesh":         (15.9, 79.7),
    "Telangana":              (18.1, 79.0),
    "Maharashtra":            (19.7, 75.7),
    "Gujarat":                (22.3, 72.6),
    "Dadra and Nagar Haveli": (20.2, 73.0),
    "Karnataka":              (15.3, 75.7),
    "Kerala":                 (10.9, 76.3),
    "Lakshadweep":            (10.6, 72.6),
    "Punjab":                 (31.1, 75.3),
    "Odisha":                 (20.9, 85.1),
    "Assam":                  (26.2, 92.9),
    "Jammu and Kashmir":      (33.7, 76.9),
    "Sikkim":                 (27.5, 88.5),
    "Arunachal Pradesh":      (28.2, 94.7),
    "Nagaland":               (26.2, 94.6),
    "Manipur":                (24.7, 93.9),
    "Mizoram":                (23.2, 92.9),
    "Meghalaya":              (25.5, 91.4),
    "Goa":                    (15.3, 74.1),
    "Andaman and Nicobar":    (11.7, 92.7),
}

# ── Region / speaker info ─────────────────────────────────────────────────────
LANGUAGE_INFO: Dict[str, tuple] = {
    "Hindi":     ("North & Central India", "~600 million speakers"),
    "Bengali":   ("Eastern India",         "~100 million speakers"),
    "Tamil":     ("South India",           "~75 million speakers"),
    "Telugu":    ("South India",           "~85 million speakers"),
    "Marathi":   ("Western India",         "~83 million speakers"),
    "Gujarati":  ("Western India",         "~57 million speakers"),
    "Kannada":   ("South India",           "~45 million speakers"),
    "Malayalam": ("South India",           "~38 million speakers"),
    "Punjabi":   ("North India",           "~33 million speakers"),
    "Odia":      ("Eastern India",         "~38 million speakers"),
    "Assamese":  ("Northeast India",       "~15 million speakers"),
    "English":   ("Pan-India",             "~125 million speakers"),
    "Urdu":      ("Pan-India",             "~70 million speakers"),
}


def _resolve_states(language: Optional[str]) -> set:
    """Expand a language name to its set of state names."""
    if not language:
        return set()
    raw = LANGUAGE_STATES.get(language, [])
    if "All States" in raw:
        return set(STATE_CENTROIDS.keys())
    return set(raw)


def create_india_language_map(
    source_language: Optional[str] = None,
    target_language: Optional[str] = None,
    title: str = "🇮🇳 Indian Language Regions",
    height: int = 520,
):
    """
    Return a Plotly Figure of India with colour-coded language region bubbles.

    source_language / target_language: display names, e.g. "Hindi", "Tamil".
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly not installed. Run: pip install plotly>=5.17.0")

    source_states  = _resolve_states(source_language)
    target_states  = _resolve_states(target_language)
    overlap_states = source_states & target_states

    lats, lons, colors, sizes, hovers = [], [], [], [], []

    for state, (lat, lon) in STATE_CENTROIDS.items():
        lats.append(lat)
        lons.append(lon)
        if state in overlap_states:
            colors.append(OVERLAP_COLOR); sizes.append(22)
            hovers.append(f"<b>{state}</b><br>Both languages")
        elif state in source_states:
            colors.append(SOURCE_COLOR);  sizes.append(20)
            hovers.append(f"<b>{state}</b><br>{source_language} region")
        elif state in target_states:
            colors.append(TARGET_COLOR);  sizes.append(20)
            hovers.append(f"<b>{state}</b><br>{target_language} region")
        else:
            colors.append(NEUTRAL_COLOR); sizes.append(9)
            hovers.append(f"<b>{state}</b>")

    traces = [
        go.Scattergeo(
            lat=lats, lon=lons,
            hovertext=hovers, hoverinfo="text",
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.88,
                        line=dict(width=0.8, color="rgba(255,255,255,0.25)")),
            showlegend=False,
        )
    ]

    # Legend ghost points
    if source_language and source_language != target_language:
        traces.append(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            marker=dict(size=12, color=SOURCE_COLOR),
            name=f"🎤 {source_language}", showlegend=True,
        ))
    if target_language:
        traces.append(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            marker=dict(size=12, color=TARGET_COLOR),
            name=f"🔁 {target_language}", showlegend=True,
        ))
    if overlap_states and source_language and target_language:
        traces.append(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            marker=dict(size=12, color=OVERLAP_COLOR),
            name="Both", showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#D0D8F0"),
                   x=0.5, xanchor="center", y=0.97),
        height=height,
        # ── FIXED: solid dark bg — transparent caused blank renders ──────────
        paper_bgcolor="#0A1020",
        plot_bgcolor="#0A1020",
        geo=dict(
            # ── FIXED: scope="world" + explicit ranges ────────────────────────
            # scope="asia" in Plotly 5.x ignores center/lataxis/lonaxis and
            # clips to a preset region that doesn't match India well.
            scope="world",
            projection_type="mercator",   # most reliable for regional maps
            showland=True,      landcolor="#1C2848",
            showocean=True,     oceancolor="#0A1632",
            showcoastlines=True, coastlinecolor="rgba(100,150,200,0.55)",
            showcountries=True, countrycolor="rgba(140,170,220,0.4)",
            showsubunits=True,  subunitcolor="rgba(120,160,220,0.35)",
            showframe=False,    showlakes=False,
            bgcolor="#0A1020",
            center=dict(lat=22.5, lon=82.0),
            lataxis=dict(range=[5, 40]),
            lonaxis=dict(range=[65, 100]),
        ),
        legend=dict(
            font=dict(color="#B8C0D8", size=11),
            bgcolor="rgba(15,22,48,0.85)",
            bordercolor="rgba(80,110,200,0.3)", borderwidth=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        hoverlabel=dict(
            bgcolor="#1C2848", bordercolor="rgba(100,150,220,0.4)",
            font=dict(color="#E0E8FF", size=12),
        ),
    )
    return fig


def get_language_region_info(language: str) -> Dict:
    """Return display info dict for a language."""
    states = LANGUAGE_STATES.get(language, [])
    region, speakers = LANGUAGE_INFO.get(language, ("India", "Data unavailable"))
    return {
        "language":    language,
        "states":      ["All Indian States"] if "All States" in states else states,
        "region":      region,
        "speakers":    speakers,
        "state_count": len(states),
    }
