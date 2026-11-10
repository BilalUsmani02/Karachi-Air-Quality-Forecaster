"""Whether the sensor map should use offline-safe styling (review item 3).

`style="open-street-map"` needs internet tiles; offline, the dots draw on a blank background.
This module decides the *default* for the sidebar's "Offline map (no background tiles)"
toggle; the actual tile-host socket probe (needs sockets) and the plotly figure itself live in
`app/components/map.py`, which is not Streamlit-free.
"""

from __future__ import annotations

OFFLINE_STYLE = "white-bg"  # a tile-free plotly map style — always renders, online or not
ONLINE_STYLE = "open-street-map"


def default_offline_map(source: str, tiles_reachable: bool | None) -> bool:
    """True to default the "offline map" toggle on.

    A snapshot that isn't "live" (cache or demo) already means today's forecast likely came
    from a fallback path with no guarantee of a live connection, so default to the safe,
    tile-free style without needing the probe. For a live snapshot, trust the cheap socket
    probe when one was run (`tiles_reachable` is `None` only if it wasn't).
    """
    if source != "live":
        return True
    if tiles_reachable is None:
        return False
    return not tiles_reachable


def map_style(offline: bool) -> str:
    """The plotly map style name for the "offline map" toggle's current value."""
    return OFFLINE_STYLE if offline else ONLINE_STYLE
