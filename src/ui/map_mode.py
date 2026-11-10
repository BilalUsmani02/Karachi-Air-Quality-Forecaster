"""Whether the sensor map should use offline-safe styling (review item 3).

`style="open-street-map"` needs internet tiles; offline, the dots draw on a blank background.
This module decides the *default* for the sidebar's "Offline map (no background tiles)"
toggle; the actual tile-host socket probe (needs sockets) and the plotly figure itself live in
`app/components/map.py`, which is not Streamlit-free.
"""

from __future__ import annotations

OFFLINE_STYLE = "white-bg"  # a tile-free plotly map style — always renders, online or not
ONLINE_STYLE = "open-street-map"


def default_offline_map(tiles_reachable: bool | None) -> bool:
    """True to default the "offline map" toggle on: only when the tile probe failed.

    The data source is no guide — the app normally loads from the cache even with internet —
    so the default follows the once-per-session socket probe; no probe result means online.
    """
    return tiles_reachable is False


def map_style(offline: bool) -> str:
    """The plotly map style name for the "offline map" toggle's current value."""
    return OFFLINE_STYLE if offline else ONLINE_STYLE
